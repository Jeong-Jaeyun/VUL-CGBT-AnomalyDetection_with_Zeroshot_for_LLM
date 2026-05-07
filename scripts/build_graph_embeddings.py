import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "configs" / "source_sink_rules_v1.json"
DEFAULT_MODEL_NAME = os.environ.get("GRAPHCODEBERT_DIR", "microsoft/graphcodebert-base")

C_KEYWORDS = {
    "alignas",
    "alignof",
    "asm",
    "auto",
    "bool",
    "break",
    "case",
    "char",
    "const",
    "constexpr",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "false",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "NULL",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "size_t",
    "static",
    "struct",
    "switch",
    "true",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
}

IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
CALL_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()")
HELPER_NAME_RE = re.compile(r"^(good|bad)[A-Za-z0-9_]*$", re.IGNORECASE)
SOURCE_SINK_NAME_RE = re.compile(r".*(source|sink)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input graph JSONL from joern_extract.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for manifest, summary, and shards.")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Local GraphCodeBERT checkpoint path or model id.",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default=None,
        help="Optional tokenizer path or model id. Defaults to model-name-or-path.",
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH, help="Source/sink rules used to preserve API identifiers.")
    parser.add_argument("--batch-size", type=int, default=32, help="Node text batch size for embedding.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum token length per node text.")
    parser.add_argument("--shard-size", type=int, default=512, help="Number of graph samples per output shard.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on graph samples.")
    parser.add_argument(
        "--pooling",
        type=str,
        choices=("mean", "cls"),
        default="mean",
        help="Token pooling strategy for node embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for embedding generation. Use auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        choices=("float32", "float16"),
        default="float32",
        help="Tensor dtype to store in shard files.",
    )
    parser.add_argument(
        "--keep-normalized-text",
        action="store_true",
        help="Persist normalized node texts in shard records for debugging.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate graph records and normalization only. Do not load GraphCodeBERT or write embedding shards.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path for summary JSON. Defaults to <output-dir>/summary.json.",
    )
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


def load_rule_names(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = set(payload.get("sources", {}).keys())
    names.update(payload.get("direct_sinks", {}).keys())
    for family_rules in payload.get("sinks", {}).values():
        names.update(family_rules.keys())
    return names


def build_preserve_identifiers(node_texts: List[str], api_names: set[str]) -> set[str]:
    preserve = set(C_KEYWORDS)
    preserve.update(api_names)
    for text in node_texts:
        # Preserve constants/macros and configured APIs only.
        # Preserving arbitrary call names leaks SARD helper labels such as badSink/goodB2G1Sink.
        preserve.update(token for token in IDENTIFIER_RE.findall(text) if token.isupper())
    return preserve


def normalize_graph_node_texts(nodes: List[Dict[str, object]], api_names: set[str]) -> List[str]:
    node_texts = [str(node.get("text", "") or "") for node in nodes]
    preserve = build_preserve_identifiers(node_texts, api_names)
    identifier_map: Dict[str, str] = {}

    def replace_identifier(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in preserve:
            return token
        if HELPER_NAME_RE.match(token) or SOURCE_SINK_NAME_RE.match(token):
            return "FUNC_HELPER"
        if token not in identifier_map:
            identifier_map[token] = f"VAR_{len(identifier_map)}"
        return identifier_map[token]

    return [IDENTIFIER_RE.sub(replace_identifier, text) for text in node_texts]


def ensure_nodes(record: Dict[str, object]) -> List[Dict[str, object]]:
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


def prepare_graph_records(records: List[Dict[str, object]], api_names: set[str]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    prepared: List[Dict[str, object]] = []
    source_counter = 0
    sink_counter = 0
    direct_sink_counter = 0
    node_type_counter: Counter[str] = Counter()
    total_nodes = 0

    for record in records:
        nodes = ensure_nodes(record)
        normalized_texts = normalize_graph_node_texts(nodes, api_names)
        node_ids = [int(node.get("id", index)) for index, node in enumerate(nodes)]
        node_types = [str(node.get("node_type", "")) for node in nodes]
        node_lines = [int(node.get("line")) if node.get("line") is not None else None for node in nodes]
        node_flags = [
            [
                int(bool(node.get("is_source", False))),
                int(bool(node.get("is_sink", False))),
                int(bool(node.get("is_direct_sink", False))),
            ]
            for node in nodes
        ]

        source_counter += sum(flag[0] for flag in node_flags)
        sink_counter += sum(flag[1] for flag in node_flags)
        direct_sink_counter += sum(flag[2] for flag in node_flags)
        node_type_counter.update(node_types)
        total_nodes += len(nodes)

        prepared.append(
            {
                "idx": str(record.get("idx")),
                "normalized_texts": normalized_texts,
                "node_ids": node_ids,
                "node_types": node_types,
                "node_lines": node_lines,
                "node_flags": node_flags,
            }
        )

    stats = {
        "record_count": len(prepared),
        "total_nodes": total_nodes,
        "source_nodes": source_counter,
        "sink_nodes": sink_counter,
        "direct_sink_nodes": direct_sink_counter,
        "node_type_histogram": dict(node_type_counter.most_common()),
    }
    return prepared, stats


def resolve_device(arg_value: str):
    import torch

    if arg_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg_value)


def pool_hidden_states(last_hidden_state, attention_mask, pooling: str):
    import torch

    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def generate_embeddings(
    prepared_records: List[Dict[str, object]],
    model_name_or_path: str,
    tokenizer_name_or_path: str | None,
    batch_size: int,
    max_length: int,
    pooling: str,
    device,
) -> Dict[str, object]:
    import torch

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for build_graph_embeddings.py. Install it in the target environment first."
        ) from exc

    tokenizer_path = tokenizer_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)

    flat_texts: List[str] = []
    flat_slices: List[Tuple[int, int]] = []
    for record in prepared_records:
        start = len(flat_texts)
        flat_texts.extend(record["normalized_texts"])
        flat_slices.append((start, len(flat_texts)))

    outputs: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(flat_texts), batch_size):
            batch_texts = flat_texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            pooled = pool_hidden_states(hidden, encoded["attention_mask"], pooling)
            outputs.append(pooled.cpu())

    flat_embeddings = torch.cat(outputs, dim=0) if outputs else torch.zeros((0, model.config.hidden_size), dtype=torch.float32)
    return {
        "tokenizer_name_or_path": tokenizer_path,
        "model_name_or_path": model_name_or_path,
        "embedding_dim": int(model.config.hidden_size),
        "flat_embeddings": flat_embeddings,
        "flat_slices": flat_slices,
    }


def convert_save_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    return torch.float32


def build_shard_payload(
    prepared_records: List[Dict[str, object]],
    embedding_bundle: Dict[str, object],
    keep_normalized_text: bool,
    save_dtype_name: str,
) -> List[Dict[str, object]]:
    import torch

    save_dtype = convert_save_dtype(save_dtype_name)
    flat_embeddings = embedding_bundle["flat_embeddings"]
    flat_slices = embedding_bundle["flat_slices"]

    shard_records: List[Dict[str, object]] = []
    for record, (start, end) in zip(prepared_records, flat_slices):
        embedding_tensor = flat_embeddings[start:end].to(dtype=save_dtype)
        item = {
            "idx": record["idx"],
            "node_ids": record["node_ids"],
            "node_types": record["node_types"],
            "node_lines": record["node_lines"],
            "node_flags": torch.tensor(record["node_flags"], dtype=torch.uint8),
            "embeddings": embedding_tensor,
        }
        if keep_normalized_text:
            item["normalized_texts"] = record["normalized_texts"]
        shard_records.append(item)
    return shard_records


def write_embedding_shards(
    output_dir: Path,
    prepared_records: List[Dict[str, object]],
    embedding_bundle: Dict[str, object],
    shard_size: int,
    keep_normalized_text: bool,
    save_dtype_name: str,
) -> List[Dict[str, object]]:
    import torch

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_payload = build_shard_payload(prepared_records, embedding_bundle, keep_normalized_text, save_dtype_name)
    manifest: List[Dict[str, object]] = []

    for shard_index in range(0, len(shard_payload), shard_size):
        chunk = shard_payload[shard_index:shard_index + shard_size]
        file_index = shard_index // shard_size
        shard_rel_path = Path("shards") / f"graph_embeddings_{file_index:05d}.pt"
        shard_abs_path = output_dir / shard_rel_path
        torch.save(
            {
                "metadata": {
                    "embedding_dim": int(embedding_bundle["embedding_dim"]),
                    "model_name_or_path": embedding_bundle["model_name_or_path"],
                    "tokenizer_name_or_path": embedding_bundle["tokenizer_name_or_path"],
                    "save_dtype": save_dtype_name,
                },
                "records": chunk,
            },
            shard_abs_path,
        )
        for record_index, item in enumerate(chunk):
            flags = item["node_flags"]
            manifest.append(
                {
                    "idx": item["idx"],
                    "shard_path": shard_rel_path.as_posix(),
                    "record_index": record_index,
                    "node_count": int(item["embeddings"].shape[0]),
                    "embedding_dim": int(item["embeddings"].shape[1]) if item["embeddings"].ndim == 2 else 0,
                    "source_nodes": int(flags[:, 0].sum().item()) if flags.numel() else 0,
                    "sink_nodes": int(flags[:, 1].sum().item()) if flags.numel() else 0,
                    "direct_sink_nodes": int(flags[:, 2].sum().item()) if flags.numel() else 0,
                }
            )

    return manifest


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else output_dir / "summary.json"
    api_names = load_rule_names(args.rules.resolve())
    records = read_jsonl(input_path, max_samples=args.max_samples)
    prepared_records, stats = prepare_graph_records(records, api_names)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_only:
        preview_records = []
        for record in prepared_records[: min(10, len(prepared_records))]:
            preview_records.append(
                {
                    "idx": record["idx"],
                    "node_count": len(record["node_ids"]),
                    "normalized_text_preview": record["normalized_texts"][: min(3, len(record["normalized_texts"]))],
                }
            )
        preview_path = output_dir / "prepare_preview.jsonl"
        write_jsonl(preview_path, preview_records)
        summary = {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "prepare_only": True,
            "model_name_or_path": args.model_name_or_path,
            "record_count": stats["record_count"],
            "total_nodes": stats["total_nodes"],
            "source_nodes": stats["source_nodes"],
            "sink_nodes": stats["sink_nodes"],
            "direct_sink_nodes": stats["direct_sink_nodes"],
            "node_type_histogram": stats["node_type_histogram"],
            "preview_path": str(preview_path),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"prepared_records={stats['record_count']}")
        print(f"total_nodes={stats['total_nodes']}")
        print(f"preview={preview_path}")
        print(f"summary={summary_path}")
        return

    device = resolve_device(args.device)
    embedding_bundle = generate_embeddings(
        prepared_records=prepared_records,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        device=device,
    )
    manifest_records = write_embedding_shards(
        output_dir=output_dir,
        prepared_records=prepared_records,
        embedding_bundle=embedding_bundle,
        shard_size=args.shard_size,
        keep_normalized_text=args.keep_normalized_text,
        save_dtype_name=args.save_dtype,
    )
    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_records)

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "prepare_only": False,
        "model_name_or_path": embedding_bundle["model_name_or_path"],
        "tokenizer_name_or_path": embedding_bundle["tokenizer_name_or_path"],
        "device": str(device),
        "pooling": args.pooling,
        "save_dtype": args.save_dtype,
        "record_count": stats["record_count"],
        "total_nodes": stats["total_nodes"],
        "source_nodes": stats["source_nodes"],
        "sink_nodes": stats["sink_nodes"],
        "direct_sink_nodes": stats["direct_sink_nodes"],
        "node_type_histogram": stats["node_type_histogram"],
        "embedding_dim": embedding_bundle["embedding_dim"],
        "shard_count": (len(manifest_records) + max(args.shard_size, 1) - 1) // max(args.shard_size, 1),
        "manifest_path": str(manifest_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"records={stats['record_count']}")
    print(f"total_nodes={stats['total_nodes']}")
    print(f"embedding_dim={embedding_bundle['embedding_dim']}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
