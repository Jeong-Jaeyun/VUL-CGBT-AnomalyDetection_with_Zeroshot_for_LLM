import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ALLOWED_BIGVUL_LANGS = {"c", "c++", "cpp"}
FAMILY_SINK_PATTERNS = {
    "CWE-119": re.compile(r"\b(strcpy|strcat|sprintf|vsprintf|memcpy|memmove|gets)\b"),
    "CWE-78": re.compile(r"\b(system|popen|execve|execvp)\b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Path to BigVul MSR_data_cleaned.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the research subset.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["CWE-119", "CWE-78"],
        help="Target CWE families to include.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Number of negatives to sample per positive.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
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


def negative_priority(record: Dict[str, object]) -> Tuple[int, int, str]:
    same_project = 1 if record["negative_project_match"] else 0
    sink_match = 1 if record["negative_sink_match"] else 0
    reason_rank = 0
    if same_project and sink_match:
        reason_rank = 3
    elif sink_match:
        reason_rank = 2
    elif same_project:
        reason_rank = 1
    return (reason_rank, same_project, str(record["idx"]))


def merged_record(existing: Dict[str, object], incoming: Dict[str, object]) -> Dict[str, object]:
    merged = dict(existing)
    existing_families = set(existing.get("subset_families", []))
    incoming_families = set(incoming.get("subset_families", []))
    merged_families = sorted(existing_families | incoming_families)
    merged["subset_families"] = merged_families
    merged["subset_family"] = merged_families[0] if len(merged_families) == 1 else "MULTI"
    merged["cwe_id"] = merged["subset_family"]
    if incoming.get("subset_role") == "positive":
        merged["subset_role"] = "positive"
    if incoming.get("negative_match_reason"):
        reasons = set(existing.get("negative_match_reasons", []))
        reasons.add(str(incoming["negative_match_reason"]))
        if existing.get("negative_match_reason"):
            reasons.add(str(existing["negative_match_reason"]))
        merged["negative_match_reasons"] = sorted(reasons)
    return merged


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_csv = (args.input_csv or (root / "dataset" / "bigvul" / "MSR_data_cleaned.csv")).resolve()
    output_dir = (
        args.output_dir or (root / "dataset" / "research_v1" / "bigvul_cwe119_78_binary_subset")
    ).resolve()
    summary_path = args.summary_path or (output_dir / "summary.json")
    rng = random.Random(args.seed)

    csv.field_size_limit(10**8)
    family_positives: Dict[str, List[Dict[str, object]]] = {family: [] for family in args.families}
    family_positive_projects: Dict[str, set[str]] = {family: set() for family in args.families}
    negative_rows: List[Tuple[int, Dict[str, str]]] = []

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

            cwe_id = (row.get("CWE ID") or "").strip()
            if label == "1" and cwe_id in family_positives:
                record = base_record(row, row_index)
                record.update(
                    {
                        "cwe_id": cwe_id,
                        "subset_family": cwe_id,
                        "subset_families": [cwe_id],
                        "subset_name": "bigvul_cwe119_78_binary_subset",
                        "research_split": "test",
                        "subset_role": "positive",
                    }
                )
                family_positives[cwe_id].append(record)
                if record["project"]:
                    family_positive_projects[cwe_id].add(str(record["project"]))
            elif label == "0":
                negative_rows.append((row_index, row))

    family_negatives: Dict[str, List[Dict[str, object]]] = {family: [] for family in args.families}
    family_match_reasons: Dict[str, Counter[str]] = {family: Counter() for family in args.families}

    for family in args.families:
        pattern = FAMILY_SINK_PATTERNS[family]
        candidates: List[Dict[str, object]] = []
        for row_index, row in negative_rows:
            record = base_record(row, row_index)
            sink_match = bool(pattern.search(str(record["func"])))
            project_match = str(record["project"]) in family_positive_projects[family]
            if not sink_match and not project_match:
                continue

            if project_match and sink_match:
                match_reason = "project+sink"
            elif sink_match:
                match_reason = "sink"
            else:
                match_reason = "project"

            record.update(
                {
                    "cwe_id": family,
                    "subset_family": family,
                    "subset_families": [family],
                    "subset_name": "bigvul_cwe119_78_binary_subset",
                    "research_split": "test",
                    "subset_role": "negative",
                    "negative_match_reason": match_reason,
                    "negative_project_match": project_match,
                    "negative_sink_match": sink_match,
                }
            )
            candidates.append(record)

        rng.shuffle(candidates)
        candidates.sort(key=negative_priority, reverse=True)

        sample_size = min(len(candidates), int(round(len(family_positives[family]) * args.negative_ratio)))
        selected = candidates[:sample_size]
        family_negatives[family] = selected
        family_match_reasons[family].update(
            str(record["negative_match_reason"]) for record in selected
        )

    combined_by_idx: Dict[str, Dict[str, object]] = {}
    summary: Dict[str, object] = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "negative_ratio": args.negative_ratio,
        "families": {},
    }

    for family in args.families:
        family_records = family_positives[family] + family_negatives[family]
        family_key = family.lower().replace("-", "")
        write_jsonl(output_dir / f"test_{family_key}.jsonl", family_records)

        for record in family_records:
            record_id = str(record["idx"])
            if record_id in combined_by_idx:
                combined_by_idx[record_id] = merged_record(combined_by_idx[record_id], record)
            else:
                combined_by_idx[record_id] = dict(record)

        summary["families"][family] = {
            "positives": len(family_positives[family]),
            "negatives": len(family_negatives[family]),
            "unique_positive_projects": len(family_positive_projects[family]),
            "negative_match_reasons": dict(family_match_reasons[family]),
        }

    combined_records = sorted(combined_by_idx.values(), key=lambda record: str(record["idx"]))
    write_jsonl(output_dir / "test.jsonl", combined_records)
    summary["combined_test_count"] = len(combined_records)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"input={input_csv}")
    print(f"output={output_dir}")
    for family in args.families:
        family_summary = summary["families"][family]
        print(
            f"[{family}] positives={family_summary['positives']} negatives={family_summary['negatives']} "
            f"projects={family_summary['unique_positive_projects']}"
        )
        for reason, count in family_summary["negative_match_reasons"].items():
            print(f"  negative_{reason}: {count}")
    print(f"combined_test={summary['combined_test_count']}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
