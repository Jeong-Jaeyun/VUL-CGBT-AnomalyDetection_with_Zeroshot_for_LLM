import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.utils import softmax as segment_softmax

from gat_conv import RelationAwareGATConv


def _num_graphs(batch: Tensor) -> int:
    if batch.numel() == 0:
        return 1
    return int(batch.max().item()) + 1


def scatter_sum(x: Tensor, batch: Tensor, dim_size: int | None = None) -> Tensor:
    if dim_size is None:
        dim_size = _num_graphs(batch)
    out = x.new_zeros((dim_size, x.size(-1)))
    if batch.numel():
        out.index_add_(0, batch, x)
    return out


def scatter_mean(x: Tensor, batch: Tensor, dim_size: int | None = None) -> Tensor:
    if dim_size is None:
        dim_size = _num_graphs(batch)
    summed = scatter_sum(x, batch, dim_size=dim_size)
    counts = x.new_zeros((dim_size, 1))
    if batch.numel():
        ones = x.new_ones((x.size(0), 1))
        counts.index_add_(0, batch, ones)
    return summed / counts.clamp(min=1.0)


def masked_mean_pool(x: Tensor, batch: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return scatter_mean(x, batch)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    dim_size = _num_graphs(batch)
    if mask.numel() == 0:
        return x.new_zeros((dim_size, x.size(-1)))
    weight = mask.float().unsqueeze(-1)
    summed = scatter_sum(x * weight, batch, dim_size=dim_size)
    counts = scatter_sum(weight, batch, dim_size=dim_size)
    return summed / counts.clamp(min=1.0)


class AttentionalGraphPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        scores = self.gate(x).squeeze(-1)
        graph_count = _num_graphs(batch)
        if scores.numel() == 0:
            return x.new_zeros((graph_count, x.size(-1)))
        weights = segment_softmax(scores, batch)
        return scatter_sum(x * weights.unsqueeze(-1), batch, dim_size=graph_count)


class RelationAwareVulGAT(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_relations: int = 8,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        num_node_types: int | None = None,
        node_type_dim: int = 32,
        flag_dim: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.heads = heads
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        self.head_dim = hidden_dim // heads
        self.dropout = nn.Dropout(dropout)
        self.node_type_embedding = nn.Embedding(num_node_types, node_type_dim) if num_node_types else None

        combined_input_dim = input_dim + flag_dim + (node_type_dim if self.node_type_embedding is not None else 0)
        self.input_projection = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList(
            [
                RelationAwareGATConv(
                    in_channels=hidden_dim,
                    out_channels=self.head_dim,
                    num_relations=num_relations,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    residual=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_projections = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.global_pool = AttentionalGraphPool(hidden_dim)
        self.readout_norms = nn.ModuleDict(
            {
                "global": nn.LayerNorm(hidden_dim),
                "source": nn.LayerNorm(hidden_dim),
                "sink": nn.LayerNorm(hidden_dim),
                "direct_sink": nn.LayerNorm(hidden_dim),
            }
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _compose_node_inputs(
        self,
        x: Tensor,
        node_flags: Tensor | None,
        node_type_ids: Tensor | None,
    ) -> Tensor:
        components = [x]
        if node_flags is None:
            node_flags = x.new_zeros((x.size(0), 3))
        components.append(node_flags.float())
        if self.node_type_embedding is not None and node_type_ids is not None:
            components.append(self.node_type_embedding(node_type_ids.long()))
        return torch.cat(components, dim=-1)

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor,
        node_flags: Tensor | None = None,
        node_type_ids: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        hidden = self.input_projection(self._compose_node_inputs(x, node_flags, node_type_ids))
        attentions: list[dict[str, Tensor]] = []

        for layer, projection, norm in zip(self.layers, self.layer_projections, self.norms):
            if return_attention:
                updated, (att_edge_index, att_edge_type, alpha) = layer(
                    hidden,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    return_attention_weights=True,
                )
                attentions.append({"edge_index": att_edge_index, "edge_type": att_edge_type, "alpha": alpha})
            else:
                updated = layer(hidden, edge_index=edge_index, edge_type=edge_type)
            updated = projection(updated)
            hidden = norm(hidden + self.dropout(F.gelu(updated)))

        return hidden, attentions

    def readout(self, hidden: Tensor, batch: Tensor, node_flags: Tensor | None = None) -> Tensor:
        if batch is None:
            batch = hidden.new_zeros(hidden.size(0), dtype=torch.long)
        if node_flags is None:
            node_flags = hidden.new_zeros((hidden.size(0), 3))
        global_features = self.readout_norms["global"](self.global_pool(hidden, batch))
        source_features = self.readout_norms["source"](masked_mean_pool(hidden, batch, node_flags[:, 0]))
        sink_features = self.readout_norms["sink"](masked_mean_pool(hidden, batch, node_flags[:, 1]))
        direct_sink_features = self.readout_norms["direct_sink"](masked_mean_pool(hidden, batch, node_flags[:, 2]))
        return torch.cat(
            (global_features, source_features, sink_features, direct_sink_features),
            dim=-1,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor,
        node_flags: Tensor | None = None,
        node_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        hidden, attentions = self.encode(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            batch=batch,
            node_flags=node_flags,
            node_type_ids=node_type_ids,
            return_attention=return_attention,
        )
        graph_features = self.readout(hidden, batch=batch, node_flags=node_flags)
        logits = self.classifier(graph_features).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        output: dict[str, Tensor | list[dict[str, Tensor]]] = {
            "logits": logits,
            "probabilities": probabilities,
            "node_embeddings": hidden,
            "graph_embeddings": graph_features,
        }
        if labels is not None:
            output["loss"] = F.binary_cross_entropy_with_logits(logits, labels.float())
        if return_attention:
            output["attentions"] = attentions
        return output
