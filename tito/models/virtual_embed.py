from typing import Optional

from torch.nn import Sequential, Linear, ReLU, Module, Sigmoid
from torch import Tensor
from torch_geometric.utils import scatter

class CentroidVirtualMsgPass(Module):
    def __init__(
            self,
            hidden_dim: int,
            k_meta: int = 3,
            virtual_to_virtual_hop: bool = False,
            ) -> None:
        super().__init__()
        self.k_meta = k_meta
        self.virtual_to_virtual_hop = virtual_to_virtual_hop

        # cluster nodes pass to virtual nodes
        self.cluster_to_virtual = Sequential(
                Linear(2 * hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                )

        # virtual node refinement by passing in intra triangle information
        if virtual_to_virtual_hop:
            self.virtual_to_virtual_update = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                    )

        # virtual node back to the cluster centroids
        self.virtual_to_cluster_update = Sequential(
                Linear(k_meta-1 * hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                )

        # cluster back to the nodes within the cluster
        self.cluster_to_nodes_gate = Sequential(
                Linear(2 * hidden_dim, hidden_dim),
                Sigmoid(),
                )


    def forward(
            self, 
            x: Tensor, 
            pos: Tensor,
            h_cluster: Tensor,
            virtual_edges: Tensor,
            cluster_idx: Tensor,
            batch: Optional[Tensor] = None) -> Tensor:
        """
        :param x: (N, hidden_dim) node features
        :param pos: (N, 3) node positions
        :param h_cluster: (N, hidden_dim) Cluster reduce information
        :param virtual_edges: (2, E) Edge index connecting centroids into subgraphs
        :param cluster_idx: (N,) The idex for referencing which node goes to what cluster
        :param batch: (N,) graph id per node

        :return x_update: (N, hidden_dim) residual-updated node features
        """
        src, dst = virtual_edges
        num_virtual = virtual_edges.size(1)

        v = self.cluster_to_virtual(torch.cat([h_cluster[src], h_cluster[dst]], dim=-1))

        if self.virtual_to_virtual_hop:
            # (E*2, )
            endpoints = torch.cat([src, dst])
            v_idx = torch.cat([torch.arange(num_virtual, device=v.device)] * 2)
            # aggregate virtual node information
            per_cluster_v = scatter(
                    v[v_idx], endpoints, dim=0, dim_size=h_cluster.size(0), reduce="mean",
                    )
            # Add the src and dst information together and reduce it
            triangle_context = (per_cluster_v[src] + per_cluster_v[dst]) * 0.5
            v = v + self.virtual_to_virtual_update(triangle_context)

        endpoints = torch.cat([src, dst])
        v_idx = torch.cat([torch.arange(num_virtual, device=v.device)] * 2)
        cluster_msg = scatter(
                v[v_idx], endpoints, dim=0, dim_size=h_cluster.size(0), reduce="mean",
                )
        h_cluster_new = h_cluster + self.virtual_to_cluster_update(
                torch.cat([h_cluster, cluster_msg], dim=-1)
                )

        broadcast = h_cluster_new[cluster_idx]
        gate = self.cluster_to_nodes_gate(torch.cat[x, broadcast], dim=-1)
        x_update = x + gate * broadcast

        return x_update

