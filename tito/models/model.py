from tqdm import tqdm
import torch
import torch_geometric as geom
import lightning as pl

from tito.models import utils
from tito.data.datasets import BaseDensity
from tito.models.loss_terms import (
        bond_length_loss, 
        circular_bond_angle_loss,
        circular_backbone_torsion_loss,
        )
from tito.models.rama import (
        ResidueVonMisesMixture,
        class_conditioned_rama_mmd,
        rama_angles_from_coordinates,
        )


class CFM(pl.pytorch.LightningModule):
    def __init__(
            self,
            score,
            lr=1e-3,
            rama_prior=None,
            lambda_rama=0.0,
            rollout_rama_weight=0.0,
            rollout_ode_steps=8,
            rollout_nested_steps=1,
            rollout_train_every_n_steps=32,
            rollout_val_batches=0,
            ):
        super().__init__()
        self.score = score 
        self.sigma = 0.001
        self.save_hyperparameters(ignore=["rama_prior"])
        self.learning_rate = lr
        self.lambda_bond = 1.0
        self.lambda_angle = 1.0
        self.lambda_torsion = 0.25
        self.lambda_rama = float(lambda_rama)
        self.rama_prior = rama_prior
        self.rollout_rama_weight = float(rollout_rama_weight)
        self.rollout_ode_steps = int(rollout_ode_steps)
        self.rollout_nested_steps = int(rollout_nested_steps)
        self.rollout_train_every_n_steps = int(rollout_train_every_n_steps)
        self.rollout_val_batches = int(rollout_val_batches)

        if self.rollout_ode_steps < 1:
            raise ValueError("rollout_ode_steps must be at least 1")
        if self.rollout_nested_steps < 1:
            raise ValueError("rollout_nested_steps must be at least 1")
        if self.rollout_train_every_n_steps < 1:
            raise ValueError("rollout_train_every_n_steps must be at least 1")
        if self.rollout_val_batches < 0:
            raise ValueError("rollout_val_batches cannot be negative")


    def training_step(self, batch, batch_idx):
        bs = batch['cond'].num_graphs
        t = torch.rand(len(batch['cond'])).type_as(batch['cond'].x)
        losses = self.get_loss(t, batch)
        self.log("train/loss", losses["loss"], prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("train/loss_flow", losses["flow"], batch_size=bs, sync_dist=True)
        self.log("train/loss_bond", losses["bond"], batch_size=bs, sync_dist=True)
        self.log("train/loss_angle", losses["angle"], batch_size=bs, sync_dist=True)

        self.log("train/loss_torsion", losses["torsion"], batch_size=bs, sync_dist=True)
        self.log("train/loss_phi", losses["phi"], batch_size=bs, sync_dist=True)
        self.log("train/loss_psi", losses["psi"], batch_size=bs, sync_dist=True)
        self.log("train/loss_rama_nll", losses["rama_nll"], batch_size=bs, sync_dist=True)
        self.log("train/rama_count", losses["rama_count"].float(), batch_size=bs, sync_dist=True)

        should_train_rollout = (
            self.rollout_rama_weight > 0.0
            and self.global_step % self.rollout_train_every_n_steps == 0
        )
        if should_train_rollout:
            rollout = self.get_rollout_rama_losses(batch)
            losses["loss"] = (
                losses["loss"]
                + self.rollout_rama_weight * rollout["rama_mmd"]
            )
            self.log(
                "train/loss_rollout_rama_mmd",
                rollout["rama_mmd"],
                on_step=True,
                on_epoch=False,
                batch_size=bs,
                sync_dist=True,
            )
            self.log(
                "train/loss_with_rollout",
                losses["loss"],
                on_step=True,
                on_epoch=False,
                batch_size=bs,
                sync_dist=True,
            )
        return losses
    
    def validation_step(self, batch, batch_idx):
        bs = batch['cond'].num_graphs
        t = torch.rand(len(batch['cond'])).type_as(batch['cond'].x)
        losses = self.get_loss(t, batch)
        self.log("valid/loss", losses["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("valid/loss_flow", losses["flow"], on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("valid/loss_bond", losses["bond"], on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("valid/loss_angle", losses["angle"], batch_size=bs, sync_dist=True, on_step=False, on_epoch=True)

        self.log("valid/loss_torsion", losses["torsion"], on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("valid/loss_phi", losses["phi"], on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("valid/loss_psi", losses["psi"], on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("valid/loss_rama_nll", losses["rama_nll"], on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("valid/rama_count", losses["rama_count"].float(), on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)

        if batch_idx < self.rollout_val_batches:
            rollout = self.get_rollout_rama_losses(batch)
            self.log(
                "valid/loss_rollout_rama_mmd",
                rollout["rama_mmd"],
                on_step=False,
                on_epoch=True,
                batch_size=bs,
                sync_dist=True,
            )
            self.log(
                "valid/loss_rollout_rama_nll",
                rollout["rama_nll"],
                on_step=False,
                on_epoch=True,
                batch_size=bs,
                sync_dist=True,
            )
            for step, value in enumerate(rollout["rama_mmd_by_step"], start=1):
                self.log(
                    f"valid/rollout_rama_mmd_step_{step}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                    sync_dist=True,
                )
            for step, value in enumerate(rollout["rama_nll_by_step"], start=1):
                self.log(
                    f"valid/rollout_rama_nll_step_{step}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                    sync_dist=True,
                )
        return losses

    def _clone_batch(self, batch):
        return {
            key: value.clone() if hasattr(value, "clone") else value
            for key, value in batch.items()
        }

    def rollout_endpoints(self, batch):
        """Differentiably reproduce the Euler/nested path used by sample()."""
        rollout_batch = self._clone_batch(batch)
        rollout_batch["corr"] = rollout_batch["target"].clone()
        # Sampling starts from an uncoupled base-density draw. Do not reuse
        # target.xbase here: the training dataset OT-couples that tensor to the
        # target, whereas scripts/sample.py draws independent Gaussian noise.
        x = torch.randn_like(rollout_batch["target"].x)
        x = utils.center_coordinates_batch(x, rollout_batch["cond"].batch)
        endpoints = []
        dt = 1.0 / self.rollout_ode_steps

        for nested_step in range(self.rollout_nested_steps):
            rollout_batch["corr"].x = x
            rand_eq_node_feats = self.sample_equivariant_features(rollout_batch)
            sample_forward = SampleHandler(self._forward, rand_eq_node_feats)

            for ode_step in range(self.rollout_ode_steps):
                t = x.new_tensor([ode_step * dt])
                velocity = sample_forward(t, rollout_batch)
                x = x + dt * velocity
                rollout_batch["corr"].x = x

            endpoints.append(x)
            rollout_batch["cond"].x = x

            if nested_step < self.rollout_nested_steps - 1:
                x = torch.randn_like(x)
                x = utils.center_coordinates_batch(
                    x, rollout_batch["cond"].batch
                )

        return endpoints

    def get_rollout_rama_losses(self, batch):
        target_phi, target_psi = rama_angles_from_coordinates(
            batch["target"].x,
            batch["target"].rama_phi_index,
            batch["target"].rama_psi_index,
        )
        mmd_by_step = []
        nll_by_step = []

        for endpoint in self.rollout_endpoints(batch):
            pred_phi, pred_psi = rama_angles_from_coordinates(
                endpoint,
                batch["target"].rama_phi_index,
                batch["target"].rama_psi_index,
            )
            mmd_by_step.append(
                class_conditioned_rama_mmd(
                    pred_phi,
                    pred_psi,
                    target_phi,
                    target_psi,
                    batch["target"].rama_class,
                )
            )
            if self.rama_prior is not None:
                nll, _ = self.rama_prior.nll_from_coordinates(
                    endpoint,
                    batch["target"].rama_phi_index,
                    batch["target"].rama_psi_index,
                    batch["target"].rama_class,
                )
            else:
                nll = endpoint.new_zeros(())
            nll_by_step.append(nll)

        return {
            "rama_mmd": torch.stack(mmd_by_step).mean(),
            "rama_nll": torch.stack(nll_by_step).mean(),
            "rama_mmd_by_step": mmd_by_step,
            "rama_nll_by_step": nll_by_step,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _forward(self, t, batch, rand_eq_node_feats):
        batch["t_diff"] = t
        return self.score(t, batch, rand_eq_node_feats)

    def get_loss(self, t, batch):
        batch['corr'] = batch['cond'].clone() # clone batch for interpolated coordinates 

        x0 = batch['target'].xbase #self.__basedistribution.sample_as(batch['cond']) # sample from base distribution
        x1 = batch['target'].x # set target interpolation coordinates 

        t_batch = t[batch['cond'].batch] # associate coordinates with sampled times

        xt = self.sample_conditional_pt(t_batch, x0, x1, batch=batch['cond'].batch) # compute interpolated coordinates
 
        batch['corr'].x = xt # inject interpolated coordinates into batch
        ut = self.compute_conditional_vector_field(x0, x1) # compute vector field

        rand_eq_node_feats = self.sample_equivariant_features(batch)
        vt = self._forward(t, batch, rand_eq_node_feats) # predict vector field from model

        loss_flow = ((vt - ut).pow(2).sum(dim=-1)).mean()

        x1_pred = x0 + vt

        loss_bond = bond_length_loss(x1_pred, x1, batch["target"].bond_index)
        loss_angle = circular_bond_angle_loss(x1_pred, x1, batch["target"].angle_index)
        loss_torsion, loss_phi, loss_psi = circular_backbone_torsion_loss(
                x1_pred, x1, batch["target"].phi_index, batch["target"].psi_index,
                )

        if self.rama_prior is not None and self.lambda_rama > 0.0:
            loss_rama_nll, rama_count = self.rama_prior.nll_from_coordinates(
                    x1_pred,
                    batch["target"].rama_phi_index,
                    batch["target"].rama_psi_index,
                    batch["target"].rama_class,
                    )
        else:
            loss_rama_nll = x1_pred.new_zeros(())
            rama_count = x1_pred.new_zeros((), dtype=torch.long)

        loss = (
                loss_flow
                + self.lambda_bond * loss_bond
                + self.lambda_angle * loss_angle
                + self.lambda_torsion * loss_torsion
                + self.lambda_rama * loss_rama_nll
                )
        return {"loss": loss, 
                "flow": loss_flow, 
                "bond": loss_bond, 
                "angle": loss_angle,
                "torsion": loss_torsion,
                "phi": loss_phi,
                "psi": loss_psi,
                "rama_nll": loss_rama_nll,
                "rama_count": rama_count,
                }


    def sample_conditional_pt(self, t, x0, x1, batch):
        epsilon = torch.normal(0, 1, size=x0.shape, device=x0.device)
        epsilon = utils.center_coordinates_batch(epsilon, batch) 
        mu_t = (t * x1.T + (1 - t) * x0.T).T
        return mu_t + self.sigma * epsilon

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def sample_equivariant_features(self, batch):
        cond = batch["cond"]
        return torch.randn(cond.node_type.size(0), self.score.n_features, 3,
                           device=cond.x.device, dtype=cond.x.dtype)

    def sample(self, batch, ode_steps=50, nested_samples=1, base_distribution=BaseDensity(std=1.0)):
        self.eval()

        with torch.no_grad():
            device = next(self.parameters()).device
            self.score.eval()
            
            x0 = batch['corr'].x
            dt = 1.0 / ode_steps
            traj = [batch['cond'].x.clone()]

            for i_nested in tqdm(range(nested_samples)):
                rand_eq_node_feats = self.sample_equivariant_features(batch)
                sh = SampleHandler(self._forward, rand_eq_node_feats)

                for i_ode in range(ode_steps): # simple Forward Euler solver
                    #print(f'Sampling step {i_ode}...', end='\r')
                    t = torch.tensor([i_ode * dt], device=device,
                                     dtype=x0.dtype)
                    velocity = sh(t, batch)
                    x0 = x0 + dt * velocity
                    batch['corr'].x = x0

                traj.append(x0.clone())
                batch["cond"].x = x0.clone() # update condition with last step

                if i_nested < nested_samples - 1:
                    x0 = base_distribution.sample_as(batch["cond"].x)
                    batch["corr"].x = x0.clone()

            batch["traj"] = batch["cond"].clone()
            batch["traj"].x = torch.stack(traj, dim=0) # store trajectory
            print("Done!")
            return batch

    
class SampleHandler:
    def __init__(self, sample_forward, rand_eq_node_feats):
        self.sample_forward = sample_forward
        self.rand_eq_node_feats = rand_eq_node_feats

    def __call__(self, t, batch):
        num_graphs = batch["cond"].num_graphs
        graph_t = t.expand(num_graphs)
        return self.sample_forward(graph_t, batch, self.rand_eq_node_feats)
