# # pylint: disable=not-callable

import numpy as np
import torch

from tito.models import device

FRAME_EDGE_TYPE = 13
NUM_EDGE_TYPES = 14

def build_node_residue_frames(x, frame_atom_index, frame_valid):
    """
    Build a right-handed N-CA-C frame for every atom.
    Atoms in the same residdue receive the same frame.
    """
    n_idx, ca_idx, c_idx = frame_atom_index.long()

    x_n = x[n_idx]
    x_ca = x[ca_idx]
    x_c = x[c_idx]

    # CA -> C
    axis_1_raw = x_c - x_ca
    axis_1_norm = torch.linalg.vector_norm(
            axis_1_raw, dim=-1, keepdim=True,
            )
    axis_1 = axis_1_raw / axis_1_norm.clamp_min(1e-8)

    # CA -> N
    n_dir = x_n - x_ca
    n_parallel = (n_dir * axis_1).sum(dim=-1, keepdim=True)
    axis_2_raw = n_dir - n_parallel
    axis_2_norm = torch.linalg.vector_norm(
            axis_2_raw, dim=-1, keepdim=True,
            )
    axis_2 = axis_2_raw / axis_2_norm.clamp_min(1e-8)

    axis_3 = torch.cross(axis_1, axis_2, dim=-1)
    axis_3 = F.normalize(axis_3, dim=-1, eps=1e-8)

    frames = torch.stack([axis_1, axis_2, axis_3], dim=-1)

    geometry_valid = (
            (axis_1_norm.squeeze(-1) > 1e-8)
            & (axis_2_norm.squeeze(-1) > 1e-8)
            )

    valid = frame_valid.bool() & geometry_valid

    return x_ca, frames, valid

class MLP(device.Module):
    def __init__(self, f_in, f_hidden, f_out, skip=False):
        super().__init__()

        self.skip = skip
        self.f_out = f_out

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(f_in, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_out),
        )

    def forward(self, x):
        if self.skip:
            return x[:, : self.f_out] + self.mlp(x)
        return self.mlp(x)


class InvariantFeatures(device.Module):
    """
    Implement embedding in child class
    All features that will be embedded should be in the batch
    """

    def __init__(self, feature_name, feature_type="node"):
        super().__init__()
        self.feature_name = feature_name
        self.feature_type = feature_type

    def forward(self, batch):
        batch = batch.clone()
        embedded_features = self.embedding(batch[self.feature_name])

        name = f"invariant_{self.feature_type}_features"
        if hasattr(batch, name):
            batch[name] = torch.cat([batch[name], embedded_features], dim=-1)
        else:
            batch[name] = embedded_features

        return batch


class NominalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, n_types, feature_type="node"):
        super().__init__(feature_name, feature_type)
        self.embedding = torch.nn.Embedding(n_types, n_features)


class PositionalEncoder(device.DeviceTracker):
    def __init__(self, dim, max_length):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"

        self.dim = dim
        self.max_length = max_length
        self.max_rank = dim // 2

    def forward(self, x):
        encodings = [self.positional_encoding(x, rank) for rank in range(1, self.max_rank + 1)]
        encodings = torch.cat(
            encodings,
            axis=1,
        )
        return encodings

    def positional_encoding(self, x, rank):
        sin = torch.sin(x / self.max_length * rank * np.pi)
        cos = torch.cos(x / self.max_length * rank * np.pi)
        assert cos.device == self.device, f"batch device {cos.device} != model device {self.device}"
        return torch.stack((cos, sin), axis=1)


class PositionalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, length):
        super().__init__(feature_name)
        assert n_features % 2 == 0, "n_features must be even"
        self.rank = n_features // 2
        self.embedding = PositionalEncoder(n_features, length)


class CombineInvariantFeatures(device.Module):
    def __init__(self, n_features_in, n_features_out, skip=False):
        super().__init__()
        self.n_features_out = n_features_out
        self.mlp = MLP(f_in=n_features_in, f_hidden=n_features_out, f_out=n_features_out, skip=skip)

    def forward(self, batch):
        invariant_node_features = self.mlp(batch.invariant_node_features)
        batch.invariant_node_features = invariant_node_features
        return batch


class EdgeEmbedding(NominalEmbedding):
    def __init__(self, n_features):
        super().__init__(feature_name="edge_type", n_features=n_features, n_types=NUM_EDGE_TYPES, feature_type="edge")


class NodeEmbedding(NominalEmbedding):
    def __init__(self, n_features):
        super().__init__(feature_name="node_type", n_features=n_features, n_types=167, feature_type="node")


class AddEquivariantFeatures(device.DeviceTracker):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def forward(self, batch):
        eq_features = torch.zeros(
            batch.node_type.shape[0],
            self.n_features,
            3,
        )
        batch.equivariant_node_features = eq_features.to(self.device)
        return batch


class EmbedGraph(device.Module):
    def __init__(self, n_features):
        super().__init__()
        self.embedding = torch.nn.Sequential(
            NodeEmbedding(n_features=n_features),
            EdgeEmbedding(n_features=n_features),
            AddEquivariantFeatures(n_features=n_features),
        )

    def forward(self, batch):
        return self.embedding(batch)

class ResidueEmbedding(NominalEmbedding):
    def __init__(self, n_features):
        super().__init__(feature_name="node_residue_type", n_features=n_features, n_types=2, feature_type="node")

class ResidueRamaEmbedding(NominalEmbedding):
    def __init__(self, n_features):
        super().__init__(feature_name="node_rama_class", n_features=n_features, n_types=21, feature_type="node")

class RelativeResidueFrameEdgeEmbedding(device.Module):
    """
    Add R_i^T R_j and local CA displacement to explict frame edges

    Frame features are nonzero only for edges whose edge type is FRAME_EDGE_TYPE
    """
    def __init__(self, n_features, coordinate_scale=10.0):
        super().__init__()

        self.coordinate_scale = float(coordinate_scaled)

        # 9 feats from R_i^T R_j
        # 3 entries from R_i^T (Ca_j - Ca_i)
        self.frame_mlp = MLP(f_in=12, f_hidden=n_features, f_out=n_features)

    def forward(self, batch):
        batch = batch.clone()

        origins, frames, node_frame_valid = build_node_residue_frames(
                x=batch.x,
                frame_atom_index=batch.frame_atom_index,
                frame_valid=batch.frame_valid,
                )

        src, dst = batch.edge_index

        frame_src_t = frames[src].transpose(-1, -2)
        frame_dst = frames[dst]

        # Relative residue-frame rotation
        relative_rotation = torch.matmul(frame_src_t, frame_dst)

        # Ca_j - Ca_i represented in residue i's frame
        origin_displacement = origins[dst] - origins[src]
        local_displacement = torch.matmul(frame_src_t, origin_displacement.unsqueeze(-1)).squeeze(-1)

        # Keep translation inputs near a convenient numerical scale
        local_displacement = local_displacement / self.coordinate_scale

        frame_features = torch.cat([relative_rotation.reshape(-1, 9), local_displacement], dim=-1)
        is_frame_edge = batch.edge_index == FRAME_EDGE_TYPE

        valid_edge = (node_frame_valid[src] & node_frame_valid[dst] & is_frame_edge)

        frame_embedding = self.frame_mlp(frame_features)
        frame_embedding = frame_embedding * valid_edge.to(batch.x.dtype).unsqueeze(-1)

        batch.invariant_edge_features = batch.invariant_edge_features + frame_embedding

        # Possible debug infor
        batch.residue_frame = frames
        batch.residue_frame_origin = origins

        return batch
