from warnings import warn
from typing import Optional, Tuple

# import numpy as np
import torch
from torch import Tensor
import torch_geometric as geom
from torch_geometric.nn import fps, knn, knn_graph
from torch_geometric.data import Batch
from torch_scatter import scatter

from tito import utils
from tito.models import device


class AddEdges(device.DeviceTracker):
    def __init__(self):
        super().__init__()

    def get_edges(self, batch):
        raise NotImplementedError()

    def forward(self, batch):
        batch = batch.clone()
        device = batch.x.device
        edge_index, edge_type = self.get_edges(batch)

        if batch.edge_index is not None:
            edge_index = torch.cat([batch.edge_index, edge_index], dim=1)
            edge_type = torch.cat([batch.edge_type, edge_type], dim=0)

        batch.edge_index = edge_index.to(device)
        batch.edge_type = edge_type.to(device)

        return batch


class AddRadiusGraph(AddEdges):
    def __init__(self, cutoff=None, edge_type=0):
        super().__init__()
        self.cutoff = float("inf") if cutoff is None else cutoff
        self.edge_type = edge_type

    def get_edges(self, batch):
        if self.cutoff == float("inf"):
            return get_fully_connected(batch, self.edge_type)

        device = batch.x.device

        edge_index = geom.nn.radius_graph(batch.x, r=self.cutoff, batch=batch.batch, max_num_neighbors=1000)
        edge_type = torch.ones(edge_index.shape[1], dtype=torch.long, device=device) * self.edge_type

        return edge_index, edge_type

class AddKnnGraph(AddEdges):
    def __init__(self, k: int = 32, edge_type: int = 0) -> None:
        super().__init__()
        self.k = k
        self.edge_type = edge_type

    def get_edges(self, batch):
        device = batch.x.device

        edge_index = geom.nn.knn_graph(batch.x, k=self.k, batch=batch.batch)
        edge_type = torch.ones(edge_index.shape[1], dtype=torch.long, device=device) * self.edge_type

        return edge_index, edge_type


class AddFullyConnectedGraph(AddEdges):
    def __init__(self, edge_type=1):
        super().__init__()
        self.edge_type = edge_type

    def get_edges(self, batch):
        return get_fully_connected(batch, self.edge_type)


def get_fully_connected(batch, edge_type=0):
    src = []
    dst = []
    for i in range(len(batch)):
        if batch.batch is None:
            nodes = torch.arange(len(batch.x), device=batch.x.device)
        else:
            nodes = torch.where(batch.batch == i)[0]
        src_, dst_ = torch.meshgrid(nodes, nodes, indexing="ij")
        src.append(src_.flatten())
        dst.append(dst_.flatten())

    src = torch.cat(src)
    dst = torch.cat(dst)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    edge_index = torch.stack([src, dst])
    edge_type = torch.ones(edge_index[0].shape, dtype=torch.long, device=dst.device) * edge_type

    return edge_index, edge_type


class AddBondGraph(AddEdges):
    def get_edges(self, batch):
        return batch.bond_index, batch.bond_type


class AddRandomGraph(AddEdges):
    def __init__(self, n_edges, edge_type=3):
        super().__init__()
        self.edge_type = edge_type
        self.n_edges = n_edges

    def get_edges(self, batch):
        random_nodes = []

        for pt_start, pt_end in zip(batch.ptr, batch.ptr[1:]):
            n_nodes = pt_end - pt_start
            random = torch.randint(pt_start, pt_end, (n_nodes * self.n_edges,))
            random_nodes.append(random)

        random_nodes = torch.cat(random_nodes)
        sorted_nodes = torch.arange(len(batch.node_type), device=batch.x.device).repeat_interleave(
            self.n_edges
        )

        edge_index = torch.stack([random_nodes, sorted_nodes], dim=0)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        return edge_index, torch.zeros_like(edge_index[0])


class ConnectVirtual(AddEdges):
    def __init__(self, edge_type=4):
        super().__init__()
        self.edge_type = edge_type

    def get_edges(self, batch):
        device = batch.x.device
        edge_index = geom.nn.radius_graph(batch.x, r=float("inf"), batch=batch.batch)

        virtual_edge_mask = torch.logical_or(
            batch.atoms[edge_index[0]] == 0,
            batch.atoms[edge_index[1]] == 0,
        )
        edge_index = edge_index[:, virtual_edge_mask]
        edge_type = torch.ones(edge_index.shape[1], dtype=torch.long, device=device) * self.edge_type

        return edge_index, edge_type


class Coalesce(device.DeviceTracker):
    def forward(self, batch):
        batch = batch.clone()
        assert hasattr(batch, "edge_type"), "Edge types have not been defined on batch"

        batch.edge_index, batch.edge_type = geom.utils.coalesce(
            batch.edge_index,
            batch.edge_type,
            reduce="max",
        )
        #  batch.edge_index, batch.edge_type = coalesce(
        #      batch.edge_index,
        #      batch.edge_type,
        #      reduce="max",
        #  )
        return batch


def coalesce(edge_index, edge_type, reduce="max"):
    idxs = torch.argsort(edge_index[1] + edge_index[0] * edge_index.size(1))
    #  idxs = np.lexsort((edge_index[1], edge_index[0]))
    edge_index = edge_index[:, idxs]
    edge_type = edge_type[idxs]
    edge_index, inverse_indices = torch.unique(edge_index, dim=1, return_inverse=True)
    edge_type = scatter(edge_type, inverse_indices, dim=0, reduce=reduce)
    return edge_index, edge_type


class AddVirtualNode(device.DeviceTracker):
    def forward(self, data):
        if isinstance(data, geom.data.Batch):
            return add_virtual_node_batch(data)
        elif isinstance(data, geom.data.Data):
            return add_virtual_node_data(data)
        else:
            raise ValueError(f"Data must be of type Batch or Data but got {type(data)}")


def add_virtual_node_batch(batch):
    warn("Adding virtual node to batch, this is extremely slow")
    batch = geom.data.Batch.from_data_list([add_virtual_node_data(data) for data in batch.to_data_list()])
    return batch


def add_virtual_node_data(data):
    data = data.clone()
    device = data.x.device
    data.x = utils.center_coordinates(data.x)
    virtual_nodes_x = torch.zeros(1, 3, device=device)
    virtual_nodes = torch.zeros(1, dtype=torch.long, device=device)
    data.x = torch.cat([data.x, virtual_nodes_x], dim=0)
    data.node_type = torch.cat([data.node_type, virtual_nodes], dim=0)
    return data


class AddVirtualGraph(AddEdges):
    def __init__(self, edge_type=4):
        super().__init__()
        self.edge_type = edge_type

    def get_edges(self, batch):
#         biggest_graph = max(np.diff(batch.ptr.cpu().numpy())) # Andrew Note: Why would you convert this to cpu? It is already loaded on the GPU
        biggest_graph = torch.max(batch.ptr[:, 1:] - batch.ptr[:, :-1])
        device = batch.x.device
        edge_index = geom.nn.radius_graph(
            batch.x, r=float(100), batch=batch.batch, max_num_neighbors=biggest_graph
        )

        virtual_edge_mask = torch.logical_or(
            batch.node_type[edge_index[0]] == 0,
            batch.node_type[edge_index[1]] == 0,
        )
        edge_index = edge_index[:, virtual_edge_mask]
        edge_type = torch.ones(edge_index.shape[1], dtype=torch.long, device=device) * self.edge_type

        return edge_index, edge_type


#  class ConnectVirtualNode(AddEdges):
#      def __init__(self, edge_type=4):
#          super().__init__()
#          self.edge_type = edge_type
#
#      def get_edges(self, batch):
#          virtual_indices = torch.nonzero(batch.atoms == 0).squeeze(dim=-1)
#          non_virtual_indices = torch.nonzero(batch.atoms != 0).squeeze(dim=-1)
#          virtual_batches = batch.batch[virtual_indices]
#          non_virtual_batches = batch.batch[non_virtual_indices]
#
#          batch_mask = virtual_batches.unsqueeze(-1) == non_virtual_batches
#          repeat_virtual_indices = virtual_indices.repeat_interleave(batch_mask.sum(dim=1))
#
#          edge_index_1 = torch.cat([repeat_virtual_indices, non_virtual_indices])
#          edge_index_2 = torch.cat([non_virtual_indices, repeat_virtual_indices])
#          edge_index = torch.stack([edge_index_1, edge_index_2], dim=0)
#
#          edge_type = torch.ones(edge_index.shape[1], dtype=torch.long, device=batch.x.device) * self.edge_type
#
#          return edge_index, edge_type


class AddSpatialFeatures(device.Module):
    def forward(self, batch):
        batch = batch.clone()
        assert hasattr(batch, "edge_type"), "Edge types have not been defined on batch"

        r = batch.x[batch.edge_index[0]] - batch.x[batch.edge_index[1]]
        edge_dist = r.norm(dim=-1)
        edge_dir = r / (1 + edge_dist.unsqueeze(-1))

        batch.edge_dist = edge_dist
        batch.edge_dir = edge_dir

        return batch


class AddGraph(device.Module):
    def __init__(self, cutoff=None, virtual_node=False):
        super().__init__()
        self.create_graph = torch.nn.Sequential(
            *[
                AddVirtualNode() if virtual_node else torch.nn.Identity(),
                AddBondGraph(),
                AddRadiusGraph(cutoff=cutoff),
                ConnectVirtual() if virtual_node else torch.nn.Identity(),
                Coalesce(),
                AddSpatialFeatures(),
            ]
        )

    def forward(self, batch):
        return self.create_graph(batch)


class AddVirtualNodeToConnectClusters(AddEdges):
    def __init__(self) -> None:
        super().__init__()

    def add_virtual_nodes(self, batch: Batch, ratio: float = 0.5, k: int = 3) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Add a virtual node that is connecting clusters betwen each other, so that messages can be passed
        between clusters after aggregation.
        :param batch: Our batch of nodes
        :param ratio: The ratio of nodes within a graph to create centroids
        :param k: The number of connections between centroids to form our virtual edges

        :return h_cluster: (N*ratio per graph, x.size(1))
        :return virtual_edge_index: (2, E) where this is the centroid subgraph edges
        :return cluster_idx: (N,) what cluster N_i is a part of
        """
        # Build centroid cluster centers and then subgraphs from centroids
        cluster_idx, centroid_pos, centroid_batch = self.build_cluster_points(batch.x, batch.batch, ratio=ratio)

        # Pool feature per cluster 
        h_cluster = scatter(batch.x, cluster_idx, dim=0, dim_size=centroid_pos.size(0), reduce="mean")

        # 3. Build k-NN graphs from the cluster centers to connect virtual points
        virtual_edge_index = self.build_meta_edges_knn(centroid_pos, centroid_batch, k)

        return h_cluster, virtual_edge_index, cluster_idx

    def build_cluster_points(self, pos: Tensor, batch_idx: Optional[Tensor] = None, ratio: float) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform FPS to build centroid cluster points -> Build k-NN graphs around the cluster centers

        :param pos: (N, 3) xyz position graphs
        :param batch_idx: (N,) a tensor indicating which graph N_i belongs to
        :param ratio: The ratio of centroids from N nodes.

        :return: cluster_idx (N,), centroid_pos (N*ratio, 3), centroid_batch: (N*ratio,)
        """
        # random_start might be played with but if False then we start with the first node of X
        centroid_node_idx = fps(x=pos, batch=batch_idx, ratio=ratio, random_start=False)
        centroid_pos = pos[centroid_node_idx]
        centroid_batch = batch_idx[centroid_node_idx] if batch_idx is not None else None

        # build our k-nn clusters and assign them for indexing later
        subgraph_edge_index = knn(centroid_pos, pos, k=1, batch_x=centroid_batch, batch_y=batch_idx)
        cluster_idx = torch.empty(pos.size(0), dtype=torch.long, device=pos.device)
        cluster_idx[subraph_edge_index[0]] = subgraph_edge_index[1]
        
        return cluster_idx, cluster_pos, centroid_batch

    def build_meta_edges_knn(self, centroid_pos: Tensor, centroid_batch: Optional[Tensor], k_meta: int) -> Tensor:
        """
        Sparse cluster-cluster adjacency via k-NN over centroids.
        This does not guarantee triangular tesselation.
        Though it is fast.
        """
        ei = knn_graph(centroid_pos, k=k_meta, batch=centroid_batch, loop=False)
        return dedupe_undirected(ei)

    def dedupe_undirected(self, edge_index: Tensor) -> Tensor:
        """
        Take a graph with duplicate edges (i, j) and (j, i) and collapse them
        into a single canonical edge i < j.
        """
        src, dst = edge_index
        low = torch.minimum(src, dst)
        high = torch.maximum(src, dst)
        stacked_lo_hi = torch.stacked([low, high], dim=0)
        uniq = torch.unique(stacked_lo_hi, dim=-1)
        return uniq


