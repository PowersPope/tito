import argparse
from datetime import timedelta
import os
from pathlib import Path
import sys

import numpy as np
import torch
import lightning as pl
from lightning.pytorch.strategies import DDPStrategy
from torch_geometric.loader import DataLoader as GeometricDataLoader


from tito import DEVICE
from tito.utils.data import get_dataset
import tito.models.model as model
import tito.models.velocity as velocity
from tito.mlops import get_wandb_logger, get_profiler
import tito.mlops as mlops 

from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import Data

from tito.data.timewarp import LaggedTimewarp
from tito.data.octopeptide import LaggedOctopeptide

from scripts.iteration_tools import validation_subset, ThroughputBenchmark

class CommonTrainingSchema(Dataset):
    """Fields needed by the current model and active losses."""


    def __init__(self, dataset):
        self.dataset = dataset
        self.graph_keys = (
                "x", "node_type", 
                "node_residue_id", "medium_atom_type",
                "bond_index", "bond_type", "angle_index",
                )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        result = {"lag": item["lag"]}

        for role in ("cond", "target"):
            original = item[role]

            fields = {}
            for key in self.graph_keys:
                if key not in original:
                    raise ValueError(f"{type(self.dataset).__name__}: "
                                     f"{role} is missing {key}")
                fields[key] = original[key]

            if role == "target":
                if "xbase" not in original:
                    raise ValueError("target is missing xbase")
                fields["xbase"] = original.xbase

            n = fields["x"].shape[0]
            for key in ("node_type", "node_residue_id", "medium_atom_type"):
                if fields[key].shape != (n,):
                    raise ValueError(
                            f"{role}.{key}: {tuple(fields[key].shape)} "
                            f"does not match {n} atoms"
                            )
                result[role] = Data(**fields)

        return result

def build_combined_datasets(args):
    if not args.timewarp_path or not args.octopep_path:
        raise ValueError("Combined training requires --timewarp-path and --octopep-path")

    if args.max_lag < max(args.timewarp_max_lag, args.octopep_max_lag):
        raise ValueError("--max-lag should cover both dataset lag ranges")

    def build_split(split):
        datasets = []

        for dataset_class, path, max_lag in (
                (
                    LaggedTimewarp,
                    args.timewarp_path,
                    args.timewarp_max_lag,
                ),
                (
                    LaggedOctopeptide,
                    args.octopep_path,
                    args.octopep_max_lag,
                ),
                ):
            dataset = dataset_class(
                    path=path,
                    split=split,
                    max_lag=max_lag,
                    fixed_lag=False,
                    normalize=True,
                    lazy_load=True,
                    ot_coupling=not args.no_ot,
                    condition_base_std=args.condition_base_std,
                    )

            dataset.uniform = not args.no_uniform_lag

            if len(dataset) == 0:
                raise ValueError(
                        f"{dataset_class.__name__}/{split}: no valid pairs; "
                        "check trajectory lengths and max_lag"
                        )

            print(f"{dataset_class.__name__}/{split}: {len(dataset)} pairs"
                  f"frame_dt={dataset.frame_dt_ps:g} ps, "
                  f"requested max_lag={max_lag:g} ps", flush=True)

            datasets.append(dataset)

        if not np.isclose(
                datasets[0].scaling_factor,
                datasets[1].scaling_factor,
                ):
            raise ValueError(
                    "Datasets use different coordinate scaling factors; "
                    "harmonize them before combined training"
                    )

        return ConcatDataset([
            CommonTrainingSchema(dataset)
            for dataset in datasets
            ])

    return build_split("train"), build_split("val")

def train_model(args):
    """
    Placeholder function for training a model.
    Replace this with your actual training logic.
    """
    torch.set_float32_matmul_precision('medium')
    print(f"Training model with the following parameters:")
    print(f"Data set: {args.data_set}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    if args.multigpu:
        print(
            "training in multi-gpu mode"
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if DEVICE == "cpu":
        num_workers = 0
    else:
        if args.num_workers is not None:
            num_workers = args.num_workers
        else:
            num_workers = len(os.sched_getaffinity(0))
    persistent_workers = True if num_workers > 0 else False

    strategy = DDPStrategy(find_unused_parameters=True) if args.multigpu else "auto"

    profiler = get_profiler(args)

    if args.data_set == "combined":
        if args.from_checkpoint_id is not None:
            raise ValueError("This needs to be trained from scratch for combined.")
        train_dataset, val_dataset = build_combined_datasets(args)
    else:
        train_dataset, val_dataset = get_dataset(args)

    benchmark = args.benchmark_steps > 0
    if min(args.val_per_dataset, args.benchmark_steps, args.benchmark_warmup) < 0:
        raise ValueError("Validation size and benchmark counts must be nonnegative.")
    if args.val_every_steps <= 0:
        raise ValueError("--val-every-steps must be positive.")

    if not benchmark:
        val_dataset = validation_subset(val_dataset, args.val_per_dataset, args.seed)
        print(f"Validation examples: {len(val_dataset)}", flush=True)


    #NOTE: we are doing ot in the get_item and not in the collate function now. Collate function not used here therefore

    train_dataloader = GeometricDataLoader(
        train_dataset, batch_size = args.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True if num_workers>0 else False, persistent_workers=persistent_workers,
    )
    
    val_dataloader = GeometricDataLoader(
        val_dataset, batch_size = args.val_batch_size or args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if num_workers>0 else False, persistent_workers=persistent_workers,
    )
    
    #instantiate the model
    if hasattr(args, 'from_checkpoint_id') and args.from_checkpoint_id is not None:
        print(f"Loading model from checkpoint {args.from_checkpoint_id} ...")
        project = args.data_set + "-tito"
        ckpt = mlops.get_checkpoint(args.entity, project, args.from_checkpoint_id, tag=args.checkpoint_tag)
        cfm = model.CFM.load_from_checkpoint(checkpoint_path=ckpt)
    else:
        print("Creating new model ...")
        vf = velocity.PainnCondVelocity(n_features=args.n_features, model_layers=args.n_model_layers, 
                                        embedding_layers=args.n_embedding_layers, length_scale=args.length_scale,
                                        n_reduced_features=args.n_reduced_features, max_lag=args.max_lag,
                                        cutoff=args.radius_cutoff, k=args.k, virtual_clusters=args.virtual_nodes,
                                        cluster_ratio=args.cluster_ratio, k_meta=args.k_meta,
                                        virtual_to_virtual_hop=args.virtual_virtual_hop)
        cfm = model.CFM(vf, lr=args.learning_rate)

    root_dir = Path(__file__).resolve().parents[1]
    wandblogger = get_wandb_logger(args, root_dir, num_workers=num_workers)

    callbacks = []

    if benchmark:
        callbacks.append(ThroughputBenchmark(args.benchmark_warmup))
    else:
        callbacks.append(
            pl.pytorch.callbacks.ModelCheckpoint(
                        monitor="train/loss" if args.no_evaluate else "valid/loss",
                        filename=f"{args.data_set}-{{epoch:02d}}-{{step}}",
                        mode="min",
                        save_top_k=2,
                        save_last=True,
                        every_n_epochs=1,
                        save_on_train_epoch_end=args.no_evaluate,
                        )
                )

#     monitor = "valid/loss" if not args.no_evaluate else "train/loss"
#     model_callback = pl.pytorch.callbacks.ModelCheckpoint(
#             monitor=monitor, 
#             filename=f"{args.data_set}-{{epoch:02d}}-{{step}}",
#             mode="min",
#             save_top_k=2,
#             save_last=True,
#             every_n_epochs=1,
#             )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_steps=(
            args.benchmark_warmup + args.benchmark_steps
            if benchmark else args.max_steps
            ),
        accelerator=DEVICE,
        logger=wandblogger,
        strategy=strategy,
        enable_checkpointing=not benchmark,
        enable_progress_bar=not benchmark,
        num_sanity_val_steps=0,
        limit_val_batches=0 if benchmark or args.no_evaluate else 1.0,
        val_check_interval=args.val_every_steps,
        check_val_every_n_epoch=None,
        #precision="bf16" if DEVICE == "gpu" else 32,
        #auto_select_gpus=multigpu,
        devices="auto",#4 if args.multigpu else 1,
        callbacks=callbacks,
        
        #profiler=profiler,
    )
    if benchmark or args.no_evaluate:
        trainer.fit(cfm, train_dataloader)
    else:
        trainer.fit(cfm, train_dataloader, val_dataloader)

def main():
    parser = argparse.ArgumentParser(description="TITO training script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_set', type=str, default="mdqm9", required=False, help="Dataset options: [ala2, mdqm9, timewarp]")
    parser.add_argument("--data_path", type=str, required=False, help="Path to dataset directory.")
    parser.add_argument('--sub_sampling_strategy', type=str, default="None", required=False, help="Sub-sampling strategy: [none, indices]")
    parser.add_argument('--sub_sampling_train_indices', type=str, default=None, required=False, help="Path to sub-sampling indices (.npy file=).")
    parser.add_argument('--sub_sampling_val_indices', type=str, default=None, required=False, help="Path to sub-sampling indices (.npy file).")
    parser.add_argument('--epochs', type=int, default=100000, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--no_evaluate', action='store_true', help="If set, do not run validation.")

    parser.add_argument('--timewarp-path', type=str)
    parser.add_argument('--octopep-path', type=str)
    parser.add_argument('--timewarp-max-lag', type=float, default=1000.0, help="Timewarp maximum sampled lag in ps.")
    parser.add_argument('--octopep-max-lag', type=float, default=10000.0, help="Octopeptide maximum sampled lag in ps.")
    parser.add_argument('--val-per-dataset', type=int, default=0, help="Validation examples per source; 0 means full validation.")
    parser.add_argument('--val-batch-size', type=int, default=None)
    parser.add_argument('--val-every-steps', type=int, default=5000)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--benchmark-steps', type=int, default=0, help="Measured steps after warm-up; 0 disables benchmarking.")
    parser.add_argument('--benchmark-warmup', type=int, default=100)

    parser.add_argument('--from_checkpoint_id', type=str, help="Wandb run id of checkpoint to start training from.")
    parser.add_argument('--entity', type=str, default="apowers4-vanderbilt-university", help="Wandb entity of checkpoint to start training from.")
    parser.add_argument('--checkpoint_tag', type=str, default="latest", help="Wandb tag of checkpoint to start training from.")
    parser.add_argument('--n_features', type=int, default=64, help="Number of hidden features.")
    parser.add_argument('--n_model_layers', type=int, default=5, help="Number of score layers.")
    parser.add_argument('--n_embedding_layers', type=int, default=2, help="Number of embedding layers.")
    parser.add_argument('--n_reduced_features', type=int, default=0, help="Number of reduced features.")
    parser.add_argument('--max_lag', type=float, default=100., help="Maximum lag for the dataset in ps.") #1000 mdqm9, 2000 timewarp
    parser.add_argument('--length_scale', type=float, default=10.0, help="Length scale for the model.")
    parser.add_argument('--distinguish_atoms', action='store_true', help="If set, distinguish atoms in the dataset.")
    parser.add_argument('--no_uniform_lag', action='store_true', help="Do not use uniform lag (default: False).")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.") #3072 mdqm9,128 for timewarp
    parser.add_argument('--seed', type=int, default=808313, help="Random seed for reproducibility.")
    parser.add_argument('--multigpu', action='store_true', help='Enable multi-GPU training')
    parser.add_argument("--save_freq", type=int, default=30, help="Frequency of saving the model in minutes.")
    parser.add_argument('--num_workers', type=int, help="Number workers (GPU Training only).")
    parser.add_argument('--no_ot', action='store_true', help='Disable optimal transport')
    parser.add_argument("--radius_cutoff", type=float, default=None, help="Specify a radius cutoff for our Graph being build within PainnCondVelocity.")
    parser.add_argument("--k", type=int, default=None, help="K-Nearest neighbors to include, if None (default) RadiusGraph is used instead.")
    parser.add_argument("--virtual-nodes", action="store_true", help="Train where centroids are calculated with k-nn graphs are computed around the centroids and then a sparse k-nn of clusters pass infromation between each other.")
    parser.add_argument("--k-meta", type=int, default=3, help="Number of centoids make up the sparse subgraph in --virtual-nodes")
    parser.add_argument("--cluster-ratio", type=float, default=0.3, help="The percentage of centroids to make given N nodes in a graph.")
    parser.add_argument("--virtual-virtual-hop", action="store_true", help="Whether to pass information along virtual-virtual nodes in a 1-hop manner.")

    parser.add_argument("--condition-base-std", type=float, default=0.1, help="Amount of noise used to create our structure-conditioned noised prior.")

    args = parser.parse_args()
    args.mode = "train"  #for compatibility with the dataset loading function

    train_model(args)
if __name__ == "__main__":
    main()
