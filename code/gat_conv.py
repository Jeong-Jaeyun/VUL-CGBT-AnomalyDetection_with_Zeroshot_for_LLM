from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, softmax


def add_self_loops_with_relation(
    edge_index: Tensor,
    edge_type: Tensor,
    num_nodes: int,
    self_loop_relation: int,
) -> Tuple[Tensor, Tensor]:
    edge_index, edge_type = remove_self_loops(edge_index, edge_type)
    loop_index = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)
    loop_edge_index = torch.stack((loop_index, loop_index), dim=0)
    loop_edge_type = edge_type.new_full((num_nodes,), self_loop_relation)
    edge_index = torch.cat((edge_index, loop_edge_index), dim=1)
    edge_type = torch.cat((edge_type, loop_edge_type), dim=0)
    return edge_index, edge_type


class RelationAwareGATConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        heads: int = 4,
        concat: bool = False,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.use_self_loops = add_self_loops
        self.self_loop_relation = num_relations

        projected_dim = heads * out_channels
        relation_count = num_relations + 1 if add_self_loops else num_relations

        self.node_proj = nn.Linear(in_channels, projected_dim, bias=False)
        self.relation_proj = nn.Embedding(relation_count, projected_dim)
        self.attention = nn.Parameter(torch.empty(1, heads, out_channels))
        self.relation_attention = nn.Parameter(torch.empty(1, heads, out_channels))
        if residual:
            residual_dim = projected_dim if concat else out_channels
            self.residual_proj = nn.Linear(in_channels, residual_dim, bias=False)
        else:
            self.residual_proj = None
        self.bias = nn.Parameter(torch.zeros(projected_dim if concat else out_channels))
        self.reset_parameters()
        self._alpha: Optional[Tensor] = None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.xavier_uniform_(self.relation_proj.weight)
        nn.init.xavier_uniform_(self.attention)
        nn.init.xavier_uniform_(self.relation_attention)
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Optional[Tensor] = None,
        return_attention_weights: bool = False,
    ):
        num_nodes = x.size(0)
        edge_index = edge_index.long()
        projected = self.node_proj(x).view(num_nodes, self.heads, self.out_channels)
        if edge_type is None:
            edge_type = edge_index.new_zeros(edge_index.size(1))
        edge_type = edge_type.long()

        if self.use_self_loops:
            edge_index, edge_type = add_self_loops_with_relation(
                edge_index=edge_index,
                edge_type=edge_type,
                num_nodes=num_nodes,
                self_loop_relation=self.self_loop_relation,
            )

        out = self.propagate(edge_index, x=projected, edge_type=edge_type, size=None)
        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        out = out + self.bias

        attention = self._alpha
        self._alpha = None
        if return_attention_weights:
            return out, (edge_index, edge_type, attention)
        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_type: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        relation = self.relation_proj(edge_type).view(-1, self.heads, self.out_channels)
        logits = F.leaky_relu(x_i + x_j + relation, negative_slope=self.negative_slope)
        alpha = (logits * self.attention).sum(dim=-1)
        alpha = alpha + (relation * self.relation_attention).sum(dim=-1)
        alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (x_j + relation) * alpha.unsqueeze(-1)
