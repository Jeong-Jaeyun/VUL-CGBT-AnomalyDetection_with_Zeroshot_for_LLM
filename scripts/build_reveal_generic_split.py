import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


GENERIC_SINK_PATTERN = re.compile(
    r"\b("
    r"strcpy|strcat|sprintf|vsprintf|snprintf|strncpy|strncat|"
    r"memcpy|memmove|gets|scanf|fscanf|sscanf|recv|recvfrom|read|fread|"
    r"system|popen|execve|execvp|malloc|calloc|realloc|free"
    r")\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root.")
    parser.add_argument("--input-files", nargs="+", type=Path, default=None, help="Reveal JSONL files to merge.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for Reveal generic binary split.",
    )
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negatives sampled per positive.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio on positive groups.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio on positive groups.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional summary path.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalized_record(record: Dict[str, object], source_name: str) -> Dict[str, object]:
    normalized = dict(record)
    original_idx = str(record.get("idx", "") or "")
    normalized["original_idx"] = original_idx
    normalized["idx"] = f"reveal::{source_name}::{original_idx}"
    normalized["dataset"] = "reveal"
    normalized["source_dataset"] = "reveal"
    normalized["subset_name"] = "reveal_generic_binary_split"
    normalized["original_source_file"] = source_name
    return normalized


def negative_priority(record: Dict[str, object]) -> Tuple[int, int, int, str]:
    project_match = 1 if record["negative_project_match"] else 0
    sink_match = 1 if record["negative_sink_match"] else 0
    if project_match and sink_match:
        reason_rank = 3
    elif sink_match:
        reason_rank = 2
    elif project_match:
        reason_rank = 1
    else:
        reason_rank = 0
    return (reason_rank, project_match, sink_match, str(record["idx"]))


def assign_group_splits(
    grouped_records: Dict[str, List[Dict[str, object]]],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, List[Dict[str, object]]]:
    groups = list(grouped_records.items())
    rng.shuffle(groups)
    total_records = sum(len(records) for _, records in groups)
    target_test = int(round(total_records * test_ratio))
    target_val = int(round(total_records * val_ratio))

    split_groups: Dict[str, List[Tuple[str, List[Dict[str, object]]]]] = {
        "train": [],
        "vale": [],
        "test": [],
    }
    split_counts = {"train": 0, "vale": 0, "test": 0}

    for group_key, records in groups:
        count = len(records)
        if split_counts["test"] < target_test:
            split_name = "test"
        elif split_counts["vale"] < target_val:
            split_name = "vale"
        else:
            split_name = "train"
        split_groups[split_name].append((group_key, records))
        split_counts[split_name] += count

    assigned: Dict[str, List[Dict[str, object]]] = {name: [] for name in split_groups}
    for split_name, grouped in split_groups.items():
        for _, records in grouped:
            assigned[split_name].extend(records)
    return assigned


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_files = args.input_files or [
        root / "dataset" / "reveal" / "vale.jsonl",
        root / "dataset" / "reveal" / "test.jsonl",
    ]
    input_files = [path.resolve() for path in input_files]
    output_dir = (args.output_dir or (root / "dataset" / "research_v1" / "reveal_generic_binary_split")).resolve()
    summary_path = args.summary_path or (output_dir / "summary.json")
    rng = random.Random(args.seed)

    merged_records: List[Dict[str, object]] = []
    seen_ids: set[str] = set()

    for input_path in input_files:
        if not input_path.exists():
            raise FileNotFoundError(f"Reveal input file not found: {input_path}")
        for record in read_jsonl(input_path):
            idx = f"reveal::{input_path.name}::{str(record.get('idx', '') or '')}"
            if not idx or idx in seen_ids:
                continue
            seen_ids.add(idx)
            merged_records.append(normalized_record(record, input_path.name))

    positive_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    positive_projects = {
        str(record.get("project", "") or "")
        for record in merged_records
        if int(record.get("target", 0)) == 1 and str(record.get("project", "") or "")
    }
    negative_pool: List[Dict[str, object]] = []

    for normalized in merged_records:
        idx = str(normalized.get("idx", ""))
        target = int(normalized.get("target", 0))
        project = str(normalized.get("project", "") or "")
        func = str(normalized.get("func", "") or "")

        if target == 1:
            group_key = project.strip() or idx
            positive_groups[group_key].append(normalized)
            continue

        sink_match = bool(GENERIC_SINK_PATTERN.search(func))
        project_match = project in positive_projects
        match_reason = "any"
        if project_match and sink_match:
            match_reason = "project+sink"
        elif sink_match:
            match_reason = "sink"
        elif project_match:
            match_reason = "project"
        normalized["subset_role"] = "negative"
        normalized["negative_match_reason"] = match_reason
        normalized["negative_project_match"] = project_match
        normalized["negative_sink_match"] = sink_match
        negative_pool.append(normalized)

    split_positives = assign_group_splits(positive_groups, args.val_ratio, args.test_ratio, rng)
    split_records: Dict[str, List[Dict[str, object]]] = {name: [] for name in ("train", "vale", "test")}
    negative_reason_counters: Dict[str, Counter[str]] = {name: Counter() for name in split_records}

    for split_name, records in split_positives.items():
        for record in records:
            updated = dict(record)
            updated["subset_role"] = "positive"
            updated["research_split"] = split_name
            split_records[split_name].append(updated)

    rng.shuffle(negative_pool)
    negative_pool.sort(key=negative_priority, reverse=True)

    used_negative_ids: set[str] = set()
    for split_name in ("train", "vale", "test"):
        pos_count = sum(1 for record in split_records[split_name] if int(record["target"]) == 1)
        sample_size = min(len(negative_pool), int(round(pos_count * args.negative_ratio)))
        selected: List[Dict[str, object]] = []
        for candidate in negative_pool:
            candidate_id = str(candidate["idx"])
            if candidate_id in used_negative_ids:
                continue
            selected.append(candidate)
            used_negative_ids.add(candidate_id)
            if len(selected) >= sample_size:
                break
        for record in selected:
            updated = dict(record)
            updated["research_split"] = split_name
            split_records[split_name].append(updated)
            negative_reason_counters[split_name][str(record["negative_match_reason"])] += 1

    summary: Dict[str, object] = {
        "input_files": [str(path) for path in input_files],
        "output_dir": str(output_dir),
        "negative_ratio": args.negative_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "merged_record_count": len(merged_records),
        "positive_group_count": len(positive_groups),
        "splits": {},
    }

    for split_name in ("train", "vale", "test"):
        records = sorted(split_records[split_name], key=lambda record: str(record["idx"]))
        write_jsonl(output_dir / f"{split_name}.jsonl", records)
        pos_count = sum(1 for record in records if int(record["target"]) == 1)
        neg_count = len(records) - pos_count
        project_counter = Counter(str(record.get("project", "")) for record in records if int(record["target"]) == 1)
        summary["splits"][split_name] = {
            "combined": len(records),
            "positives": pos_count,
            "negatives": neg_count,
            "positive_project_count": len(project_counter),
            "negative_match_reasons": dict(negative_reason_counters[split_name]),
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"output={output_dir}")
    for split_name in ("train", "vale", "test"):
        split_summary = summary["splits"][split_name]
        print(
            f"[{split_name}] combined={split_summary['combined']} "
            f"positives={split_summary['positives']} negatives={split_summary['negatives']}"
        )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
