import argparse
import json
import logging
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from gat_model import RelationAwareVulGAT
from metrics import compute_binary_classification_metrics


logger = logging.getLogger(__name__)
VOCAB_UNK = "__UNK__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-graphs", type=Path, default=None)
    parser.add_argument("--train-embeddings", type=Path, default=None)
    parser.add_argument("--valid-graphs", type=Path, default=None)
    parser.add_argument("--valid-embeddings", type=Path, default=None)
    parser.add_argument("--test-graphs", type=Path, default=None)
    parser.add_argument("--test-embeddings", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-cache-size", type=int, default=8)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint-metric", type=str, default="balanced_accuracy")
    parser.add_argument("--tune-threshold-on-valid", action="store_true")
    parser.add_argument("--threshold-metric", type=str, default="balanced_accuracy")
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--save-explanations", action="store_true")
    parser.add_argument("--topk-explanation-edges", type=int, default=10)
    parser.add_argument("--max-path-depth", type=int, default=8)
    parser.add_argument("--relation-vocab-path", type=Path, default=None)
    parser.add_argument("--node-type-vocab-path", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path, max_samples: int | None = None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_vocab(path: Path) -> Dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): int(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return {str(value): index for index, value in enumerate(payload)}
    raise ValueError(f"Unsupported vocabulary payload: {path}")


def save_vocab(path: Path, vocab: Dict[str, int]) -> None:
    ordered = OrderedDict(sorted(vocab.items(), key=lambda item: item[1]))
    write_json(path, ordered)


def build_vocab_from_graphs(graph_paths: List[Path], field: str) -> Dict[str, int]:
    values = {VOCAB_UNK}
    for graph_path in graph_paths:
        for record in read_jsonl(graph_path):
            if field == "edge_type":
                for edge in record.get("edges", []):
                    values.add(str(edge.get("edge_type", VOCAB_UNK)))
            elif field == "node_type":
                for node in record.get("nodes", []):
                    values.add(str(node.get("node_type", VOCAB_UNK)))
            else:
                raise ValueError(f"Unsupported vocab field: {field}")
    ordered_values = [VOCAB_UNK] + sorted(value for value in values if value != VOCAB_UNK)
    return {value: index for index, value in enumerate(ordered_values)}


class EmbeddingShardCache:
    def __init__(self, cache_dir: Path, max_shards: int = 4) -> None:
        self.cache_dir = cache_dir
        self.max_shards = max_shards
        self._cache: "OrderedDict[str, Dict[str, object]]" = OrderedDict()

    def get_record(self, shard_path: str, record_index: int) -> Dict[str, object]:
        if shard_path not in self._cache:
            abs_path = self.cache_dir / shard_path
            shard_payload = torch.load(abs_path, map_location="cpu", weights_only=False)
            self._cache[shard_path] = shard_payload
            while len(self._cache) > self.max_shards:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(shard_path)
        return self._cache[shard_path]["records"][record_index]


def load_embedding_manifest(cache_dir: Path) -> Dict[str, Dict[str, object]]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing embedding manifest: {manifest_path}")
    manifest = {}
    for record in read_jsonl(manifest_path):
        manifest[str(record["idx"])] = record
    return manifest


def ensure_graph_nodes(record: Dict[str, object]) -> List[Dict[str, object]]:
    nodes = record.get("nodes")
    if isinstance(nodes, list) and nodes:
        return nodes
    fallback_text = str(record.get("code") or record.get("func") or "")
    return [
        {
            "id": 0,
            "text": fallback_text,
            "node_type": "METHOD",
            "line": 1,
            "is_source": False,
            "is_sink": False,
            "is_direct_sink": False,
        }
    ]


class GraphEmbeddingDataset(Dataset):
    def __init__(
        self,
        graph_path: Path,
        embedding_dir: Path,
        relation_vocab: Dict[str, int],
        node_type_vocab: Dict[str, int],
        max_samples: int | None = None,
        shard_cache_size: int = 4,
    ) -> None:
        self.graph_path = graph_path.resolve()
        self.embedding_dir = embedding_dir.resolve()
        self.relation_vocab = relation_vocab
        self.node_type_vocab = node_type_vocab
        self.records = read_jsonl(self.graph_path, max_samples=max_samples)
        self.embedding_manifest = load_embedding_manifest(self.embedding_dir)
        self.shard_cache = EmbeddingShardCache(self.embedding_dir, max_shards=shard_cache_size)
        self.labels = [int(record.get("target", 0)) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        sample_id = str(record["idx"])
        if sample_id not in self.embedding_manifest:
            raise KeyError(f"Embedding cache does not contain sample: {sample_id}")

        manifest_record = self.embedding_manifest[sample_id]
        shard_record = self.shard_cache.get_record(
            shard_path=str(manifest_record["shard_path"]),
            record_index=int(manifest_record["record_index"]),
        )

        node_ids = [int(value) for value in shard_record["node_ids"]]
        embeddings = shard_record["embeddings"].float()
        node_flags = shard_record["node_flags"].float()

        graph_nodes = ensure_graph_nodes(record)
        node_by_id = {int(node.get("id", idx)): node for idx, node in enumerate(graph_nodes)}
        ordered_nodes = [node_by_id[node_id] for node_id in node_ids if node_id in node_by_id]
        if len(ordered_nodes) != len(node_ids):
            missing_ids = [node_id for node_id in node_ids if node_id not in node_by_id]
            raise ValueError(f"Graph node ids missing from graph JSON for sample {sample_id}: {missing_ids[:5]}")
        if embeddings.size(0) != len(ordered_nodes):
            raise ValueError(
                f"Embedding node count mismatch for sample {sample_id}: "
                f"{embeddings.size(0)} embeddings vs {len(ordered_nodes)} graph nodes"
            )

        remap = {node_id: offset for offset, node_id in enumerate(node_ids)}
        edge_rows: List[List[int]] = []
        edge_types: List[int] = []
        original_edges: List[Dict[str, object]] = []
        for edge in record.get("edges", []):
            src = int(edge.get("src", -1))
            dst = int(edge.get("dst", -1))
            if src not in remap or dst not in remap:
                continue
            relation_name = str(edge.get("edge_type", VOCAB_UNK))
            edge_rows.append([remap[src], remap[dst]])
            edge_types.append(self.relation_vocab.get(relation_name, self.relation_vocab[VOCAB_UNK]))
            original_edges.append(
                {
                    "src": remap[src],
                    "dst": remap[dst],
                    "edge_type": relation_name,
                }
            )

        if edge_rows:
            edge_index = torch.tensor(edge_rows, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0,), dtype=torch.long)

        node_type_ids = torch.tensor(
            [
                self.node_type_vocab.get(str(node.get("node_type", VOCAB_UNK)), self.node_type_vocab[VOCAB_UNK])
                for node in ordered_nodes
            ],
            dtype=torch.long,
        )
        label = torch.tensor([float(record.get("target", 0))], dtype=torch.float32)

        data = Data(
            x=embeddings,
            edge_index=edge_index,
            edge_type=edge_type,
            node_flags=node_flags,
            node_type_ids=node_type_ids,
            y=label,
        )
        metadata = {
            "idx": sample_id,
            "target": int(record.get("target", 0)),
            "dataset": str(record.get("dataset", "")),
            "cwe_id": str(record.get("cwe_id", "")),
            "subset_name": str(record.get("subset_name", "")),
            "slice_strategy": str(record.get("slice_strategy", "")),
            "source_tags": list(record.get("source_tags", [])),
            "sink_tags": list(record.get("sink_tags", [])),
            "direct_sink_tags": list(record.get("direct_sink_tags", [])),
            "node_ids": node_ids,
            "node_texts": [str(node.get("text", "")) for node in ordered_nodes],
            "node_types": [str(node.get("node_type", "")) for node in ordered_nodes],
            "node_lines": [int(node.get("line")) if node.get("line") is not None else None for node in ordered_nodes],
            "node_flags": [[int(value) for value in row] for row in node_flags.tolist()],
            "edges": original_edges,
        }
        return {"data": data, "metadata": metadata}


def collate_graph_batch(items: List[Dict[str, object]]) -> Tuple[Batch, List[Dict[str, object]]]:
    batch = Batch.from_data_list([item["data"] for item in items])
    metadata = [item["metadata"] for item in items]
    return batch, metadata


def compute_pos_weight(labels: List[int]) -> float:
    positives = sum(int(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0:
        return 1.0
    return negatives / positives


def select_metric(metrics: Dict[str, float | int], metric_name: str) -> float:
    metric_value = metrics.get(metric_name)
    if metric_value is None:
        raise KeyError(f"Metric '{metric_name}' not available in metrics: {sorted(metrics)}")
    return float(metric_value)


def metric_rank(metrics: Dict[str, float | int], metric_name: str) -> Tuple[float, float, float, float, float]:
    primary = select_metric(metrics, metric_name)
    alternate = float(metrics.get("balanced_accuracy" if metric_name != "balanced_accuracy" else "f1", 0.0))
    predicted_positive_rate = float(metrics.get("predicted_positive_rate", 0.5))
    actual_positive_rate = float(metrics.get("actual_positive_rate", 0.5))
    prevalence_alignment = -abs(predicted_positive_rate - actual_positive_rate)
    loss = -float(metrics.get("loss", 0.0)) if metrics.get("loss") is not None else 0.0
    threshold_distance = -abs(float(metrics.get("threshold", 0.5)) - 0.5)
    return primary, alternate, prevalence_alignment, loss, threshold_distance


def evaluate_threshold(labels: List[int], scores: List[float], threshold: float) -> Dict[str, float | int]:
    predictions = [1 if score >= threshold else 0 for score in scores]
    metrics = compute_binary_classification_metrics(labels, predictions)
    metrics["threshold"] = float(threshold)
    return metrics


def tune_threshold(labels: List[int], scores: List[float], metric_name: str) -> Dict[str, float | int]:
    best_threshold = 0.5
    best_metrics = evaluate_threshold(labels, scores, threshold=best_threshold)
    best_rank = metric_rank(best_metrics, metric_name)

    for candidate in np.linspace(0.05, 0.95, 91):
        metrics = evaluate_threshold(labels, scores, threshold=float(candidate))
        current_rank = metric_rank(metrics, metric_name)
        if current_rank > best_rank:
            best_threshold = float(candidate)
            best_metrics = metrics
            best_rank = current_rank
    return best_metrics


def build_prediction_records(
    metadata_records: List[Dict[str, object]],
    labels: List[int],
    scores: List[float],
    predictions: List[int],
    threshold: float,
) -> List[Dict[str, object]]:
    output = []
    for meta, label, score, pred in zip(metadata_records, labels, scores, predictions):
        output.append(
            {
                "idx": meta["idx"],
                "dataset": meta["dataset"],
                "cwe_id": meta["cwe_id"],
                "target": int(label),
                "score": float(score),
                "predicted_label": int(pred),
                "threshold": float(threshold),
                "slice_strategy": meta.get("slice_strategy", ""),
                "source_tags": meta.get("source_tags", []),
                "sink_tags": meta.get("sink_tags", []),
                "direct_sink_tags": meta.get("direct_sink_tags", []),
            }
        )
    return output


def find_best_source_sink_path(
    edge_items: List[Dict[str, object]],
    node_flags: List[List[int]],
    max_depth: int,
) -> Dict[str, object] | None:
    source_nodes = [index for index, flags in enumerate(node_flags) if flags[0] == 1]
    sink_nodes = {index for index, flags in enumerate(node_flags) if flags[1] == 1 or flags[2] == 1}
    if not source_nodes or not sink_nodes:
        return None

    adjacency: Dict[int, List[Dict[str, object]]] = {}
    for edge in edge_items:
        adjacency.setdefault(int(edge["src"]), []).append(edge)

    best_path: Dict[str, object] | None = None
    best_score = float("-inf")

    def dfs(node: int, visited: set[int], path_edges: List[Dict[str, object]], score_sum: float) -> None:
        nonlocal best_path, best_score
        if len(path_edges) > max_depth:
            return
        if node in sink_nodes and path_edges:
            average_score = score_sum / len(path_edges)
            if average_score > best_score:
                best_score = average_score
                best_path = {
                    "average_attention": float(average_score),
                    "length": len(path_edges),
                    "edges": [
                        {
                            "src": int(edge["src"]),
                            "dst": int(edge["dst"]),
                            "edge_type": edge["edge_type"],
                            "attention": float(edge["attention"]),
                        }
                        for edge in path_edges
                    ],
                    "node_sequence": [int(path_edges[0]["src"])] + [int(edge["dst"]) for edge in path_edges],
                }
        for edge in adjacency.get(node, []):
            dst = int(edge["dst"])
            if dst in visited:
                continue
            dfs(dst, visited | {dst}, path_edges + [edge], score_sum + float(edge["attention"]))

    for source_node in source_nodes:
        dfs(source_node, {source_node}, [], 0.0)
    return best_path


def extract_sample_explanation(
    sample_metadata: Dict[str, object],
    node_start: int,
    node_end: int,
    attentions: List[Dict[str, torch.Tensor]],
    top_k: int,
    relation_id_to_name: Dict[int, str],
    max_path_depth: int,
) -> Dict[str, object]:
    if not attentions:
        return {"idx": sample_metadata["idx"], "top_edges": [], "critical_path": None}

    last_layer = attentions[-1]
    edge_index = last_layer["edge_index"].detach().cpu()
    edge_type = last_layer["edge_type"].detach().cpu()
    alpha = last_layer["alpha"].detach().cpu()
    if alpha.ndim == 2:
        alpha = alpha.mean(dim=1)

    mask = (
        (edge_index[0] >= node_start)
        & (edge_index[0] < node_end)
        & (edge_index[1] >= node_start)
        & (edge_index[1] < node_end)
        & (edge_index[0] != edge_index[1])
    )

    local_edges: List[Dict[str, object]] = []
    node_texts = sample_metadata["node_texts"]
    node_lines = sample_metadata["node_lines"]
    for src, dst, rel_id, score in zip(
        edge_index[0][mask].tolist(),
        edge_index[1][mask].tolist(),
        edge_type[mask].tolist(),
        alpha[mask].tolist(),
    ):
        local_src = int(src - node_start)
        local_dst = int(dst - node_start)
        if not (0 <= local_src < len(node_texts) and 0 <= local_dst < len(node_texts)):
            continue
        local_edges.append(
            {
                "src": local_src,
                "dst": local_dst,
                "edge_type": relation_id_to_name.get(int(rel_id), VOCAB_UNK),
                "attention": float(score),
                "src_text": node_texts[local_src],
                "dst_text": node_texts[local_dst],
                "src_line": node_lines[local_src],
                "dst_line": node_lines[local_dst],
            }
        )

    local_edges.sort(key=lambda item: item["attention"], reverse=True)
    critical_path = find_best_source_sink_path(local_edges, sample_metadata["node_flags"], max_depth=max_path_depth)
    return {
        "idx": sample_metadata["idx"],
        "target": sample_metadata["target"],
        "dataset": sample_metadata["dataset"],
        "cwe_id": sample_metadata["cwe_id"],
        "top_edges": local_edges[:top_k],
        "critical_path": critical_path,
    }


def train_one_epoch(
    model: RelationAwareVulGAT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
    pos_weight: float | None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch, _metadata in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            node_flags=batch.node_flags,
            node_type_ids=batch.node_type_ids,
        )
        labels = batch.y.view(-1).float()
        if pos_weight is not None:
            weight_tensor = torch.tensor([pos_weight], device=device, dtype=outputs["logits"].dtype)
            loss = F.binary_cross_entropy_with_logits(outputs["logits"], labels, pos_weight=weight_tensor)
        else:
            loss = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return {
        "loss": (total_loss / total_examples) if total_examples else 0.0,
        "num_examples": total_examples,
    }


def evaluate_model(
    model: RelationAwareVulGAT,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    pos_weight: float | None = None,
    relation_id_to_name: Dict[int, str] | None = None,
    export_explanations: bool = False,
    top_k_edges: int = 10,
    max_path_depth: int = 8,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels: List[int] = []
    scores: List[float] = []
    metadata_records: List[Dict[str, object]] = []
    explanation_records: List[Dict[str, object]] = []

    with torch.inference_mode():
        for batch, metadata in dataloader:
            batch = batch.to(device)
            outputs = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_type=batch.edge_type,
                batch=batch.batch,
                node_flags=batch.node_flags,
                node_type_ids=batch.node_type_ids,
                return_attention=export_explanations,
            )
            batch_labels = batch.y.view(-1).float()
            if pos_weight is not None:
                weight_tensor = torch.tensor([pos_weight], device=device, dtype=outputs["logits"].dtype)
                loss = F.binary_cross_entropy_with_logits(outputs["logits"], batch_labels, pos_weight=weight_tensor)
            else:
                loss = F.binary_cross_entropy_with_logits(outputs["logits"], batch_labels)

            batch_scores = outputs["probabilities"].detach().cpu().tolist()
            total_loss += float(loss.item()) * batch_labels.size(0)
            total_examples += batch_labels.size(0)
            labels.extend(int(value) for value in batch_labels.cpu().tolist())
            scores.extend(float(value) for value in batch_scores)
            metadata_records.extend(metadata)

            if export_explanations:
                ptr = batch.ptr.detach().cpu().tolist()
                attentions = outputs.get("attentions", [])
                for sample_index, sample_metadata in enumerate(metadata):
                    explanation_records.append(
                        extract_sample_explanation(
                            sample_metadata=sample_metadata,
                            node_start=int(ptr[sample_index]),
                            node_end=int(ptr[sample_index + 1]),
                            attentions=attentions,
                            top_k=top_k_edges,
                            relation_id_to_name=relation_id_to_name or {},
                            max_path_depth=max_path_depth,
                        )
                    )

    metrics = evaluate_threshold(labels, scores, threshold=threshold)
    metrics["loss"] = (total_loss / total_examples) if total_examples else 0.0
    predictions = [1 if score >= threshold else 0 for score in scores]
    prediction_records = build_prediction_records(metadata_records, labels, scores, predictions, threshold)
    return {
        "metrics": metrics,
        "labels": labels,
        "scores": scores,
        "predictions": predictions,
        "prediction_records": prediction_records,
        "explanations": explanation_records,
    }


def prepare_dataloader(
    graph_path: Path | None,
    embedding_dir: Path | None,
    relation_vocab: Dict[str, int],
    node_type_vocab: Dict[str, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    shard_cache_size: int,
    max_samples: int | None,
) -> tuple[GraphEmbeddingDataset | None, DataLoader | None]:
    if graph_path is None or embedding_dir is None:
        return None, None
    dataset = GraphEmbeddingDataset(
        graph_path=graph_path,
        embedding_dir=embedding_dir,
        relation_vocab=relation_vocab,
        node_type_vocab=node_type_vocab,
        max_samples=max_samples,
        shard_cache_size=shard_cache_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_graph_batch,
    )
    return dataset, dataloader


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    device = resolve_device(args.device)

    graph_paths = [
        path.resolve()
        for path in [args.train_graphs, args.valid_graphs, args.test_graphs]
        if path is not None
    ]
    if not graph_paths:
        raise ValueError("At least one graph dataset path must be provided.")

    relation_vocab = load_vocab(args.relation_vocab_path.resolve()) if args.relation_vocab_path else build_vocab_from_graphs(graph_paths, "edge_type")
    node_type_vocab = load_vocab(args.node_type_vocab_path.resolve()) if args.node_type_vocab_path else build_vocab_from_graphs(graph_paths, "node_type")
    relation_vocab_path = args.output_dir / "relation_vocab.json"
    node_type_vocab_path = args.output_dir / "node_type_vocab.json"
    save_vocab(relation_vocab_path, relation_vocab)
    save_vocab(node_type_vocab_path, node_type_vocab)
    relation_id_to_name = {value: key for key, value in relation_vocab.items()}

    train_dataset, train_loader = prepare_dataloader(
        graph_path=args.train_graphs.resolve() if args.train_graphs else None,
        embedding_dir=args.train_embeddings.resolve() if args.train_embeddings else None,
        relation_vocab=relation_vocab,
        node_type_vocab=node_type_vocab,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        shard_cache_size=args.shard_cache_size,
        max_samples=args.max_train_samples,
    )
    valid_dataset, valid_loader = prepare_dataloader(
        graph_path=args.valid_graphs.resolve() if args.valid_graphs else None,
        embedding_dir=args.valid_embeddings.resolve() if args.valid_embeddings else None,
        relation_vocab=relation_vocab,
        node_type_vocab=node_type_vocab,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        shard_cache_size=args.shard_cache_size,
        max_samples=args.max_valid_samples,
    )
    test_dataset, test_loader = prepare_dataloader(
        graph_path=args.test_graphs.resolve() if args.test_graphs else None,
        embedding_dir=args.test_embeddings.resolve() if args.test_embeddings else None,
        relation_vocab=relation_vocab,
        node_type_vocab=node_type_vocab,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        shard_cache_size=args.shard_cache_size,
        max_samples=args.max_test_samples,
    )

    if train_dataset is None and args.checkpoint_path is None:
        raise ValueError("Training data or --checkpoint-path is required.")

    if train_dataset is not None:
        input_dim = int(train_dataset[0]["data"].x.size(-1))
    elif valid_dataset is not None:
        input_dim = int(valid_dataset[0]["data"].x.size(-1))
    elif test_dataset is not None:
        input_dim = int(test_dataset[0]["data"].x.size(-1))
    else:
        raise ValueError("Could not infer input dimension from datasets.")

    model = RelationAwareVulGAT(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_relations=len(relation_vocab),
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        num_node_types=len(node_type_vocab),
    ).to(device)

    if args.checkpoint_path is not None and args.checkpoint_path.exists():
        checkpoint_payload = torch.load(args.checkpoint_path.resolve(), map_location=device, weights_only=False)
        state_dict = checkpoint_payload["model_state"] if "model_state" in checkpoint_payload else checkpoint_payload
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint: %s", args.checkpoint_path)

    config_payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config_payload["device_resolved"] = str(device)
    config_payload["input_dim"] = input_dim
    config_payload["num_relations"] = len(relation_vocab)
    config_payload["num_node_types"] = len(node_type_vocab)
    write_json(args.output_dir / "run_config.json", config_payload)

    pos_weight = None
    if train_dataset is not None and not args.disable_pos_weight:
        pos_weight = compute_pos_weight(train_dataset.labels)

    best_checkpoint_path = args.output_dir / "best_model.pt"
    best_metric_value = float("-inf")
    best_metric_rank: Tuple[float, float, float, float, float] | None = None
    best_epoch = None
    history: List[Dict[str, object]] = []

    if train_loader is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                max_grad_norm=args.max_grad_norm,
                pos_weight=pos_weight,
            )
            epoch_payload: Dict[str, object] = {"epoch": epoch, "train": train_stats}

            if valid_loader is not None:
                valid_result = evaluate_model(
                    model=model,
                    dataloader=valid_loader,
                    device=device,
                    threshold=args.prediction_threshold,
                    pos_weight=pos_weight,
                    relation_id_to_name=relation_id_to_name,
                    export_explanations=False,
                )
                epoch_payload["valid"] = valid_result["metrics"]
                metric_value = select_metric(valid_result["metrics"], args.checkpoint_metric)
                current_rank = metric_rank(valid_result["metrics"], args.checkpoint_metric)
                if best_metric_rank is None or current_rank > best_metric_rank:
                    best_metric_value = metric_value
                    best_metric_rank = current_rank
                    best_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "metric_name": args.checkpoint_metric,
                            "metric_value": metric_value,
                        },
                        best_checkpoint_path,
                    )
            elif epoch == args.epochs:
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_checkpoint_path)

            history.append(epoch_payload)
            logger.info("epoch=%s train_loss=%.4f", epoch, train_stats["loss"])

        write_json(args.output_dir / "training_history.json", {"epochs": history})
        if best_checkpoint_path.exists():
            checkpoint_payload = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint_payload["model_state"])

    selected_threshold = float(args.prediction_threshold)
    threshold_payload: Dict[str, object] = {
        "selected_threshold": selected_threshold,
        "source": "argument",
        "metric": args.threshold_metric,
    }

    if args.tune_threshold_on_valid and valid_loader is not None:
        valid_result = evaluate_model(
            model=model,
            dataloader=valid_loader,
            device=device,
            threshold=args.prediction_threshold,
            pos_weight=pos_weight,
            relation_id_to_name=relation_id_to_name,
            export_explanations=False,
        )
        tuned = tune_threshold(valid_result["labels"], valid_result["scores"], metric_name=args.threshold_metric)
        selected_threshold = float(tuned["threshold"])
        threshold_payload = {
            "selected_threshold": selected_threshold,
            "source": "validation_sweep",
            "metric": args.threshold_metric,
            "validation_metrics": tuned,
        }
    write_json(args.output_dir / "selected_threshold.json", threshold_payload)

    summary: Dict[str, object] = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
        "selected_threshold": selected_threshold,
        "pos_weight": pos_weight,
    }

    if valid_loader is not None:
        valid_result = evaluate_model(
            model=model,
            dataloader=valid_loader,
            device=device,
            threshold=selected_threshold,
            pos_weight=pos_weight,
            relation_id_to_name=relation_id_to_name,
            export_explanations=args.save_explanations,
            top_k_edges=args.topk_explanation_edges,
            max_path_depth=args.max_path_depth,
        )
        summary["valid_metrics"] = valid_result["metrics"]
        write_json(args.output_dir / "valid_metrics.json", valid_result["metrics"])
        write_jsonl(args.output_dir / "valid_predictions.jsonl", valid_result["prediction_records"])
        if args.save_explanations:
            write_jsonl(args.output_dir / "valid_explanations.jsonl", valid_result["explanations"])

    if test_loader is not None:
        test_result = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            threshold=selected_threshold,
            pos_weight=pos_weight,
            relation_id_to_name=relation_id_to_name,
            export_explanations=args.save_explanations,
            top_k_edges=args.topk_explanation_edges,
            max_path_depth=args.max_path_depth,
        )
        summary["test_metrics"] = test_result["metrics"]
        write_json(args.output_dir / "test_metrics.json", test_result["metrics"])
        write_jsonl(args.output_dir / "test_predictions.jsonl", test_result["prediction_records"])
        if args.save_explanations:
            write_jsonl(args.output_dir / "test_explanations.jsonl", test_result["explanations"])

    write_json(args.output_dir / "run_summary.json", summary)


if __name__ == "__main__":
    main()
