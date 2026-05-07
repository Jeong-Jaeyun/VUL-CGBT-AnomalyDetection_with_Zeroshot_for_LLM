import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from graph_extractor import build_graph_record


DEFAULT_SEED = 42
DEFAULT_SPLIT_RATIOS = {"train": 0.8, "vale": 0.1, "test": 0.1}
ALLOWED_BIGVUL_LANGS = {"c", "c++", "cpp"}
SARD_SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx"}
CONTROL_FLOW_KEYWORDS = {"if", "for", "while", "switch", "else", "do"}
SARD_FUNCTION_RE = re.compile(
    r"(?m)^[ \t]*(?:static\s+)?(?:[A-Za-z_][\w\s\*\(\)]*?\s+)?(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["devign", "bigvul", "sard"],
        default=["devign", "bigvul", "sard"],
        help="Datasets to prepare.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for splitting.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root directory.",
    )
    parser.add_argument(
        "--devign-source",
        type=str,
        default="DetectVul/devign",
        help="Hugging Face dataset id for Devign.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path for a JSON summary file. Defaults to dataset/preparation_summary.json.",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Regenerate *_with_cfg_dfg.jsonl files from existing split JSONL files without rebuilding splits.",
    )
    parser.add_argument("--cfg-max-nodes", type=int, default=128, help="Maximum CFG nodes per sample.")
    parser.add_argument("--cfg-max-edges", type=int, default=512, help="Maximum CFG edges per sample.")
    parser.add_argument("--dfg-max-nodes", type=int, default=128, help="Maximum DFG nodes per sample.")
    parser.add_argument("--dfg-max-edges", type=int, default=512, help="Maximum DFG edges per sample.")
    return parser.parse_args()


def normalize_record(
    *,
    idx: str,
    func: str,
    target: int,
    dataset_name: str,
    project: str = "",
    commit_id: str = "",
    extra_fields: Dict[str, object] | None = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "idx": str(idx),
        "func": func.strip(),
        "target": int(target),
        "project": project,
        "commit_id": commit_id,
        "dataset": dataset_name,
    }
    if extra_fields:
        record.update(extra_fields)
    return record


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_graph_jsonl(
    path: Path,
    records: Iterable[Dict[str, object]],
    *,
    cfg_max_nodes: int,
    cfg_max_edges: int,
    dfg_max_nodes: int,
    dfg_max_edges: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            graph_record = build_graph_record(
                record,
                cfg_max_nodes=cfg_max_nodes,
                cfg_max_edges=cfg_max_edges,
                dfg_max_nodes=dfg_max_nodes,
                dfg_max_edges=dfg_max_edges,
            )
            handle.write(json.dumps(graph_record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_split_outputs(
    output_dir: Path,
    split_records: Dict[str, List[Dict[str, object]]],
    *,
    cfg_max_nodes: int,
    cfg_max_edges: int,
    dfg_max_nodes: int,
    dfg_max_edges: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split_name in ("train", "vale", "test"):
        records = split_records.get(split_name, [])
        counts[split_name] = write_jsonl(output_dir / f"{split_name}.jsonl", records)
        counts[f"{split_name}_with_cfg_dfg"] = write_graph_jsonl(
            output_dir / f"{split_name}_with_cfg_dfg.jsonl",
            records,
            cfg_max_nodes=cfg_max_nodes,
            cfg_max_edges=cfg_max_edges,
            dfg_max_nodes=dfg_max_nodes,
            dfg_max_edges=dfg_max_edges,
        )
    return counts


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_dataset_output_dir(root: Path, dataset_name: str) -> Path:
    if dataset_name == "devign":
        return root / "dataset" / "devign"
    if dataset_name == "bigvul":
        return root / "dataset" / "bigvul"
    if dataset_name == "sard":
        return root / "dataset" / "SARD"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def regenerate_graph_outputs(
    output_dir: Path,
    *,
    cfg_max_nodes: int,
    cfg_max_edges: int,
    dfg_max_nodes: int,
    dfg_max_edges: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split_name in ("train", "vale", "test"):
        input_path = output_dir / f"{split_name}.jsonl"
        output_path = output_dir / f"{split_name}_with_cfg_dfg.jsonl"
        if not input_path.exists():
            counts[split_name] = 0
            counts[f"{split_name}_with_cfg_dfg"] = 0
            continue

        with input_path.open("r", encoding="utf-8") as handle:
            records = (json.loads(line) for line in handle if line.strip())
            counts[f"{split_name}_with_cfg_dfg"] = write_graph_jsonl(
                output_path,
                records,
                cfg_max_nodes=cfg_max_nodes,
                cfg_max_edges=cfg_max_edges,
                dfg_max_nodes=dfg_max_nodes,
                dfg_max_edges=dfg_max_edges,
            )
        counts[split_name] = count_jsonl_lines(input_path)
    return counts


def split_grouped_records(
    records: List[Dict[str, object]],
    group_key_fn: Callable[[Dict[str, object]], str],
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[group_key_fn(record)].append(record)

    group_items: List[Tuple[str, List[Dict[str, object]]]] = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    total_records = sum(len(items) for _, items in group_items)
    target_counts = {
        split_name: total_records * ratio
        for split_name, ratio in DEFAULT_SPLIT_RATIOS.items()
    }
    split_records: Dict[str, List[Dict[str, object]]] = {"train": [], "vale": [], "test": []}
    current_counts = {split_name: 0 for split_name in split_records}

    for _, group_records in group_items:
        best_split = max(
            split_records,
            key=lambda split_name: target_counts[split_name] - current_counts[split_name],
        )
        if all(current_counts[name] >= target_counts[name] for name in split_records):
            best_split = min(split_records, key=lambda split_name: current_counts[split_name])
        split_records[best_split].extend(group_records)
        current_counts[best_split] += len(group_records)

    return split_records


def prepare_devign(
    root: Path,
    dataset_id: str,
    *,
    cfg_max_nodes: int,
    cfg_max_edges: int,
    dfg_max_nodes: int,
    dfg_max_edges: int,
) -> Tuple[Path, Dict[str, int]]:
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    output_dir = root / "dataset" / "devign"
    split_records: Dict[str, List[Dict[str, object]]] = {"train": [], "vale": [], "test": []}
    split_map = {"train": "train", "validation": "vale", "test": "test"}
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
    parquet_files = [path for path in repo_files if path.startswith("data/") and path.endswith(".parquet")]

    for hf_split, output_split in split_map.items():
        split_candidates = sorted(path for path in parquet_files if Path(path).name.startswith(f"{hf_split}-"))
        if not split_candidates:
            raise FileNotFoundError(f"Could not find parquet shard for split '{hf_split}' in dataset '{dataset_id}'.")

        parquet_path = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=split_candidates[0],
        )
        frame = pd.read_parquet(parquet_path)
        for row in frame.to_dict(orient="records"):
            func = (row.get("func") or row.get("func_clean") or "").strip()
            if not func:
                continue
            split_records[output_split].append(
                normalize_record(
                    idx=str(row.get("id", len(split_records[output_split]))),
                    func=func,
                    target=int(row["target"]),
                    dataset_name="devign",
                    project=str(row.get("project", "")),
                    commit_id=str(row.get("commit_id", "")),
                    extra_fields={"source": dataset_id},
                )
            )

    counts = write_split_outputs(
        output_dir,
        split_records,
        cfg_max_nodes=cfg_max_nodes,
        cfg_max_edges=cfg_max_edges,
        dfg_max_nodes=dfg_max_nodes,
        dfg_max_edges=dfg_max_edges,
    )
    return output_dir, counts


def prepare_bigvul(
    root: Path,
    seed: int,
    *,
    cfg_max_nodes: int,
    cfg_max_edges: int,
    dfg_max_nodes: int,
    dfg_max_edges: int,
) -> Tuple[Path, Dict[str, int]]:
    csv.field_size_limit(10**8)
    csv_path = root / "dataset" / "bigvul" / "MSR_data_cleaned.csv"
    output_dir = root / "dataset" / "bigvul"
    records: List[Dict[str, object]] = []

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            func = (row.get("func_before") or "").strip()
            label = row.get("vul")
            lang = (row.get("lang") or "").strip().lower()
            if not func or label not in {"0", "1"}:
                continue
            if lang and lang not in ALLOWED_BIGVUL_LANGS:
                continue
            records.append(
                normalize_record(
                    idx=f"bigvul-{row_index}",
                    func=func,
                    target=int(label),
                    dataset_name="bigvul",
                    project=str(row.get("project", "")),
                    commit_id=str(row.get("commit_id", "")),
                    extra_fields={
                        "lang": row.get("lang", ""),
                        "file_name": row.get("file_name", ""),
                        "source_csv": str(csv_path.name),
                    },
                )
            )

    split_records = split_grouped_records(
        records,
        group_key_fn=lambda record: str(record.get("commit_id") or record["idx"]),
        seed=seed,
    )
    counts = write_split_outputs(
        output_dir,
        split_records,
        cfg_max_nodes=cfg_max_nodes,
        cfg_max_edges=cfg_max_edges,
        dfg_max_nodes=dfg_max_nodes,
        dfg_max_edges=dfg_max_edges,
    )
    return output_dir, counts


def find_matching_brace(source: str, open_brace_index: int) -> int:
    depth = 0
    for index in range(open_brace_index, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def sard_label_for_name(function_name: str) -> int | None:
    lowered = function_name.lower()
    if lowered in CONTROL_FLOW_KEYWORDS or lowered == "main":
        return None
    if lowered.startswith("good") or "_good" in lowered:
        return 0
    if lowered == "bad" or lowered.endswith("bad") or "_bad" in lowered:
        return 1
    return None


def extract_sard_records(source_path: Path, dataset_root: Path) -> List[Dict[str, object]]:
    source = source_path.read_text(encoding="utf-8", errors="ignore")
    relative_path = source_path.relative_to(dataset_root)
    cwe_name = relative_path.parts[0] if relative_path.parts else ""
    records: List[Dict[str, object]] = []

    for match in SARD_FUNCTION_RE.finditer(source):
        function_name = match.group("name")
        label = sard_label_for_name(function_name)
        if label is None:
            continue

        open_brace_index = source.find("{", match.start())
        close_brace_index = find_matching_brace(source, open_brace_index)
        if open_brace_index == -1 or close_brace_index == -1:
            continue

        snippet = source[match.start() : close_brace_index + 1].strip()
        if not snippet:
            continue

        records.append(
            normalize_record(
                idx=f"{relative_path.as_posix()}::{function_name}",
                func=snippet,
                target=label,
                dataset_name="sard",
                project=cwe_name,
                commit_id="",
                extra_fields={
                    "source_file": relative_path.as_posix(),
                    "function_name": function_name,
                },
            )
        )

    return records


def prepare_sard(
    root: Path,
    seed: int,
    *,
    cfg_max_nodes: int,
    cfg_max_edges: int,
    dfg_max_nodes: int,
    dfg_max_edges: int,
) -> Tuple[Path, Dict[str, int]]:
    dataset_root = root / "dataset" / "SARD"
    output_dir = dataset_root
    records: List[Dict[str, object]] = []

    source_files = sorted(
        path for path in dataset_root.rglob("*") if path.suffix.lower() in SARD_SOURCE_EXTENSIONS
    )
    for source_path in source_files:
        records.extend(extract_sard_records(source_path, dataset_root))

    split_records = split_grouped_records(
        records,
        group_key_fn=lambda record: str(record.get("source_file") or record["idx"]),
        seed=seed,
    )
    counts = write_split_outputs(
        output_dir,
        split_records,
        cfg_max_nodes=cfg_max_nodes,
        cfg_max_edges=cfg_max_edges,
        dfg_max_nodes=dfg_max_nodes,
        dfg_max_edges=dfg_max_edges,
    )
    return output_dir, counts


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summary_path = args.summary_path or (root / "dataset" / "preparation_summary.json")
    summary: Dict[str, Dict[str, object]] = {}

    for dataset_name in args.datasets:
        if args.graph_only:
            output_dir = resolve_dataset_output_dir(root, dataset_name)
            counts = regenerate_graph_outputs(
                output_dir,
                cfg_max_nodes=args.cfg_max_nodes,
                cfg_max_edges=args.cfg_max_edges,
                dfg_max_nodes=args.dfg_max_nodes,
                dfg_max_edges=args.dfg_max_edges,
            )
        elif dataset_name == "devign":
            output_dir, counts = prepare_devign(
                root,
                args.devign_source,
                cfg_max_nodes=args.cfg_max_nodes,
                cfg_max_edges=args.cfg_max_edges,
                dfg_max_nodes=args.dfg_max_nodes,
                dfg_max_edges=args.dfg_max_edges,
            )
        elif dataset_name == "bigvul":
            output_dir, counts = prepare_bigvul(
                root,
                args.seed,
                cfg_max_nodes=args.cfg_max_nodes,
                cfg_max_edges=args.cfg_max_edges,
                dfg_max_nodes=args.dfg_max_nodes,
                dfg_max_edges=args.dfg_max_edges,
            )
        elif dataset_name == "sard":
            output_dir, counts = prepare_sard(
                root,
                args.seed,
                cfg_max_nodes=args.cfg_max_nodes,
                cfg_max_edges=args.cfg_max_edges,
                dfg_max_nodes=args.dfg_max_nodes,
                dfg_max_edges=args.dfg_max_edges,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        summary[dataset_name] = {
            "output_dir": str(output_dir),
            "counts": counts,
            "graph_limits": {
                "cfg_max_nodes": args.cfg_max_nodes,
                "cfg_max_edges": args.cfg_max_edges,
                "dfg_max_nodes": args.dfg_max_nodes,
                "dfg_max_edges": args.dfg_max_edges,
            },
        }
        print(f"[{dataset_name}] output={output_dir}")
        for key, value in counts.items():
            print(f"  {key}: {value}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
