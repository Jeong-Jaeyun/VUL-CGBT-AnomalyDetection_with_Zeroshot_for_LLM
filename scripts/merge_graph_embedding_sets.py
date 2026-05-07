import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-input",
        action="append",
        required=True,
        type=Path,
        help="Graph JSONL input. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--embedding-input",
        action="append",
        required=True,
        type=Path,
        help="Embedding directory corresponding to --graph-input. Repeat in the same order.",
    )
    parser.add_argument("--graph-output", type=Path, required=True, help="Merged graph JSONL output.")
    parser.add_argument("--embedding-output", type=Path, required=True, help="Merged embedding directory output.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional summary JSON output.")
    parser.add_argument("--shard-size", type=int, default=512, help="Records per merged embedding shard.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_embedding_manifest(embedding_dir: Path) -> Dict[str, Dict[str, object]]:
    manifest_path = embedding_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")
    return {str(record["idx"]): record for record in read_jsonl(manifest_path)}


def load_shard_records(embedding_dir: Path, shard_rel_path: str) -> List[Dict[str, object]]:
    import torch

    shard_path = embedding_dir / shard_rel_path
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    return list(payload.get("records", []))


def main() -> None:
    args = parse_args()
    if len(args.graph_input) != len(args.embedding_input):
        raise ValueError("--graph-input and --embedding-input counts must match.")

    graph_output = args.graph_output.resolve()
    embedding_output = args.embedding_output.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else embedding_output / "merge_summary.json"

    if graph_output.exists() and not args.overwrite:
        raise FileExistsError(f"Graph output exists: {graph_output}")
    if embedding_output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Embedding output exists: {embedding_output}")
        shutil.rmtree(embedding_output)

    graph_records: List[Dict[str, object]] = []
    merged_embedding_records: List[Dict[str, object]] = []
    seen_ids: set[str] = set()
    label_counter: Counter[int] = Counter()
    dataset_counter: Counter[str] = Counter()
    origin_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    source_summaries: List[Dict[str, object]] = []

    for graph_path, embedding_dir in zip(args.graph_input, args.embedding_input):
        graph_path = graph_path.resolve()
        embedding_dir = embedding_dir.resolve()
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph input not found: {graph_path}")
        if not embedding_dir.exists():
            raise FileNotFoundError(f"Embedding input not found: {embedding_dir}")

        manifest = load_embedding_manifest(embedding_dir)
        shard_cache: Dict[str, List[Dict[str, object]]] = {}
        source_summaries.append(
            {
                "graph": str(graph_path),
                "embedding": str(embedding_dir),
                "summary": read_json(embedding_dir / "summary.json") if (embedding_dir / "summary.json").exists() else {},
            }
        )

        for record in read_jsonl(graph_path):
            sample_id = str(record["idx"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id while merging: {sample_id}")
            if sample_id not in manifest:
                raise KeyError(f"Embedding missing for sample id: {sample_id}")
            seen_ids.add(sample_id)
            graph_records.append(record)
            label_counter[int(record.get("target", 0))] += 1
            dataset_counter[str(record.get("dataset", ""))] += 1
            origin_counter[str(record.get("graph_origin", ""))] += 1
            family_counter.update(str(value) for value in record.get("path_families", []) or [])

            manifest_record = manifest[sample_id]
            shard_rel_path = str(manifest_record["shard_path"])
            if shard_rel_path not in shard_cache:
                shard_cache[shard_rel_path] = load_shard_records(embedding_dir, shard_rel_path)
            merged_embedding_records.append(shard_cache[shard_rel_path][int(manifest_record["record_index"])])

    write_jsonl(graph_output, graph_records)

    import torch

    shards_dir = embedding_output / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: List[Dict[str, object]] = []
    embedding_dim = 0
    total_nodes = 0
    source_nodes = 0
    sink_nodes = 0
    direct_sink_nodes = 0

    for shard_start in range(0, len(merged_embedding_records), max(args.shard_size, 1)):
        shard_index = shard_start // max(args.shard_size, 1)
        chunk = merged_embedding_records[shard_start: shard_start + max(args.shard_size, 1)]
        shard_rel_path = Path("shards") / f"graph_embeddings_{shard_index:05d}.pt"
        shard_path = embedding_output / shard_rel_path
        torch.save(
            {
                "metadata": {
                    "merged": True,
                    "source_count": len(args.embedding_input),
                },
                "records": chunk,
            },
            shard_path,
        )
        for record_index, item in enumerate(chunk):
            flags = item["node_flags"]
            embeddings = item["embeddings"]
            node_count = int(embeddings.shape[0])
            item_embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
            embedding_dim = max(embedding_dim, item_embedding_dim)
            total_nodes += node_count
            source_count = int(flags[:, 0].sum().item()) if flags.numel() else 0
            sink_count = int(flags[:, 1].sum().item()) if flags.numel() else 0
            direct_sink_count = int(flags[:, 2].sum().item()) if flags.numel() else 0
            source_nodes += source_count
            sink_nodes += sink_count
            direct_sink_nodes += direct_sink_count
            manifest_records.append(
                {
                    "idx": item["idx"],
                    "shard_path": shard_rel_path.as_posix(),
                    "record_index": record_index,
                    "node_count": node_count,
                    "embedding_dim": item_embedding_dim,
                    "source_nodes": source_count,
                    "sink_nodes": sink_count,
                    "direct_sink_nodes": direct_sink_count,
                }
            )

    manifest_path = embedding_output / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_records)
    summary = {
        "graph_output": str(graph_output),
        "embedding_output": str(embedding_output),
        "record_count": len(graph_records),
        "label_counts": {str(key): value for key, value in sorted(label_counter.items())},
        "dataset_counts": dict(dataset_counter.most_common()),
        "origin_counts": dict(origin_counter.most_common()),
        "path_family_counts": dict(family_counter.most_common()),
        "total_nodes": total_nodes,
        "source_nodes": source_nodes,
        "sink_nodes": sink_nodes,
        "direct_sink_nodes": direct_sink_nodes,
        "embedding_dim": embedding_dim,
        "shard_count": (len(merged_embedding_records) + max(args.shard_size, 1) - 1) // max(args.shard_size, 1),
        "manifest_path": str(manifest_path),
        "sources": source_summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"records={len(graph_records)}")
    print(f"graph={graph_output}")
    print(f"embedding={embedding_output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
