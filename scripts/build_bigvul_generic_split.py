import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ALLOWED_BIGVUL_LANGS = {"c", "c++", "cpp"}
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
    parser.add_argument("--input-csv", type=Path, default=None, help="Path to BigVul MSR_data_cleaned.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for BigVul generic binary split.",
    )
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negatives sampled per positive.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio on positive groups.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio on positive groups.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional summary path.")
    return parser.parse_args()


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def base_record(row: Dict[str, str], row_index: int) -> Dict[str, object]:
    return {
        "idx": f"bigvul-{row_index}",
        "func": (row.get("func_before") or "").strip(),
        "target": int(row["vul"]),
        "project": str(row.get("project", "")),
        "commit_id": str(row.get("commit_id", "")),
        "dataset": "bigvul",
        "source_dataset": "bigvul",
        "lang": str(row.get("lang", "")),
        "file_name": str(row.get("file_name", "")),
        "original_cwe_id": str(row.get("CWE ID", "")),
        "source_csv": "MSR_data_cleaned.csv",
    }


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
    input_csv = (args.input_csv or (root / "dataset" / "bigvul" / "MSR_data_cleaned.csv")).resolve()
    output_dir = (args.output_dir or (root / "dataset" / "research_v1" / "bigvul_generic_binary_split")).resolve()
    summary_path = args.summary_path or (output_dir / "summary.json")
    rng = random.Random(args.seed)

    csv.field_size_limit(10**8)
    positive_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    negative_rows: List[Tuple[int, Dict[str, str]]] = []
    positive_projects: set[str] = set()

    with input_csv.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            func = (row.get("func_before") or "").strip()
            label = (row.get("vul") or "").strip()
            lang = (row.get("lang") or "").strip().lower()
            if not func or label not in {"0", "1"}:
                continue
            if lang and lang not in ALLOWED_BIGVUL_LANGS:
                continue

            if label == "1":
                record = base_record(row, row_index)
                record.update(
                    {
                        "subset_name": "bigvul_generic_binary_split",
                        "subset_role": "positive",
                        "cwe_id": str(row.get("CWE ID", "")),
                    }
                )
                group_key = str(record["commit_id"]).strip() or str(record["idx"])
                positive_groups[group_key].append(record)
                if record["project"]:
                    positive_projects.add(str(record["project"]))
            else:
                negative_rows.append((row_index, row))

    split_positives = assign_group_splits(positive_groups, args.val_ratio, args.test_ratio, rng)
    split_records: Dict[str, List[Dict[str, object]]] = {name: [] for name in ("train", "vale", "test")}
    negative_reason_counters: Dict[str, Counter[str]] = {name: Counter() for name in split_records}

    for split_name, records in split_positives.items():
        for record in records:
            updated = dict(record)
            updated["research_split"] = split_name
            split_records[split_name].append(updated)

    candidate_negatives: List[Dict[str, object]] = []
    for row_index, row in negative_rows:
        record = base_record(row, row_index)
        sink_match = bool(GENERIC_SINK_PATTERN.search(str(record["func"])))
        project_match = str(record["project"]) in positive_projects
        match_reason = "any"
        if project_match and sink_match:
            match_reason = "project+sink"
        elif sink_match:
            match_reason = "sink"
        elif project_match:
            match_reason = "project"
        record.update(
            {
                "subset_name": "bigvul_generic_binary_split",
                "subset_role": "negative",
                "cwe_id": str(row.get("CWE ID", "")),
                "negative_match_reason": match_reason,
                "negative_project_match": project_match,
                "negative_sink_match": sink_match,
            }
        )
        candidate_negatives.append(record)

    rng.shuffle(candidate_negatives)
    candidate_negatives.sort(key=negative_priority, reverse=True)

    used_negative_ids: set[str] = set()
    for split_name in ("train", "vale", "test"):
        pos_count = sum(1 for record in split_records[split_name] if int(record["target"]) == 1)
        sample_size = min(len(candidate_negatives), int(round(pos_count * args.negative_ratio)))
        selected: List[Dict[str, object]] = []
        for candidate in candidate_negatives:
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
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "negative_ratio": args.negative_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "positive_group_count": len(positive_groups),
        "splits": {},
    }

    for split_name in ("train", "vale", "test"):
        records = sorted(split_records[split_name], key=lambda record: str(record["idx"]))
        write_jsonl(output_dir / f"{split_name}.jsonl", records)
        pos_count = sum(1 for record in records if int(record["target"]) == 1)
        neg_count = len(records) - pos_count
        cwe_counter = Counter(str(record.get("cwe_id", "")) for record in records if int(record["target"]) == 1)
        summary["splits"][split_name] = {
            "combined": len(records),
            "positives": pos_count,
            "negatives": neg_count,
            "positive_cwe_distribution": dict(cwe_counter),
            "negative_match_reasons": dict(negative_reason_counters[split_name]),
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"input={input_csv}")
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
