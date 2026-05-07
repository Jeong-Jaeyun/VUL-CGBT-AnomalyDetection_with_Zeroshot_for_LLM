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
    "CWE-119": re.compile(r"\b(strcpy|strcat|sprintf|vsprintf|snprintf|strncpy|strncat|memcpy|memmove|gets)\b"),
    "CWE-78": re.compile(r"\b(system|popen|execve|execvp)\b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Path to BigVul MSR_data_cleaned.csv.")
    parser.add_argument(
        "--existing-test-jsonl",
        type=Path,
        default=None,
        help="Optional external test JSONL to exclude from the fine-tune pool.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for fine-tune train/vale JSONL.")
    parser.add_argument("--families", nargs="+", default=["CWE-119", "CWE-78"], help="Target CWE families.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negatives to sample per positive.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio applied to positives.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio applied to positives.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional summary JSON path.")
    return parser.parse_args()


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


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
    return merged


def assign_splits(
    records: List[Dict[str, object]],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    if not shuffled:
        return [], [], []

    total = len(shuffled)
    test_count = int(round(total * test_ratio))
    val_count = int(round(total * val_ratio))

    if total >= 3:
        if test_count <= 0:
            test_count = 1
        if val_count <= 0:
            val_count = 1
        while test_count + val_count >= total:
            if test_count >= val_count and test_count > 1:
                test_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                break
    else:
        test_count = 0
        val_count = 0

    test_records = shuffled[:test_count]
    val_records = shuffled[test_count : test_count + val_count]
    train_records = shuffled[test_count + val_count :]
    return train_records, val_records, test_records


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_csv = (args.input_csv or (root / "dataset" / "bigvul" / "MSR_data_cleaned.csv")).resolve()
    existing_test_jsonl = args.existing_test_jsonl.resolve() if args.existing_test_jsonl else None
    output_dir = (
        args.output_dir or (root / "dataset" / "research_v1" / "bigvul_cwe119_78_finetune_subset")
    ).resolve()
    summary_path = args.summary_path or (output_dir / "summary.json")
    rng = random.Random(args.seed)

    excluded_ids = (
        {str(record.get("idx", "")) for record in read_jsonl(existing_test_jsonl)}
        if existing_test_jsonl is not None and existing_test_jsonl.exists()
        else set()
    )

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

            record_id = f"bigvul-{row_index}"
            if record_id in excluded_ids:
                continue

            cwe_id = (row.get("CWE ID") or "").strip()
            if label == "1" and cwe_id in family_positives:
                record = base_record(row, row_index)
                record.update(
                    {
                        "cwe_id": cwe_id,
                        "subset_family": cwe_id,
                        "subset_families": [cwe_id],
                        "subset_name": "bigvul_cwe119_78_finetune_subset",
                        "subset_role": "positive",
                    }
                )
                family_positives[cwe_id].append(record)
                if record["project"]:
                    family_positive_projects[cwe_id].add(str(record["project"]))
            elif label == "0":
                negative_rows.append((row_index, row))

    split_records: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        split_name: {family: [] for family in args.families}
        for split_name in ("train", "vale", "test")
    }

    negative_reason_counters: Dict[str, Dict[str, Counter[str]]] = {
        split_name: {family: Counter() for family in args.families}
        for split_name in ("train", "vale", "test")
    }

    for family in args.families:
        train_pos, val_pos, test_pos = assign_splits(family_positives[family], args.val_ratio, args.test_ratio, rng)
        for split_name, records in (("train", train_pos), ("vale", val_pos), ("test", test_pos)):
            for record in records:
                updated = dict(record)
                updated["research_split"] = split_name
                split_records[split_name][family].append(updated)

        pattern = FAMILY_SINK_PATTERNS[family]
        candidate_negatives: List[Dict[str, object]] = []
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
                    "subset_name": "bigvul_cwe119_78_finetune_subset",
                    "subset_role": "negative",
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
            pos_count = len(split_records[split_name][family])
            sample_size = min(
                len(candidate_negatives),
                int(round(pos_count * args.negative_ratio)),
            )
            selected: List[Dict[str, object]] = []
            for candidate in candidate_negatives:
                if str(candidate["idx"]) in used_negative_ids:
                    continue
                selected.append(candidate)
                used_negative_ids.add(str(candidate["idx"]))
                if len(selected) >= sample_size:
                    break

            for record in selected:
                updated = dict(record)
                updated["research_split"] = split_name
                split_records[split_name][family].append(updated)
                negative_reason_counters[split_name][family][str(record["negative_match_reason"])] += 1

    summary: Dict[str, object] = {
        "input_csv": str(input_csv),
        "excluded_test_jsonl": str(existing_test_jsonl) if existing_test_jsonl is not None else None,
        "excluded_test_count": len(excluded_ids),
        "output_dir": str(output_dir),
        "negative_ratio": args.negative_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "families": {},
        "splits": {},
    }

    for split_name in ("train", "vale", "test"):
        combined_by_idx: Dict[str, Dict[str, object]] = {}
        split_summary: Dict[str, object] = {"families": {}}
        for family in args.families:
            family_records = split_records[split_name][family]
            family_key = family.lower().replace("-", "")
            count = write_jsonl(output_dir / f"{split_name}_{family_key}.jsonl", family_records)
            split_summary["families"][family] = count
            for record in family_records:
                record_id = str(record["idx"])
                if record_id in combined_by_idx:
                    combined_by_idx[record_id] = merged_record(combined_by_idx[record_id], record)
                else:
                    combined_by_idx[record_id] = dict(record)
        combined_records = sorted(combined_by_idx.values(), key=lambda record: str(record["idx"]))
        split_summary["combined"] = write_jsonl(output_dir / f"{split_name}.jsonl", combined_records)
        summary["splits"][split_name] = split_summary

    for family in args.families:
        summary["families"][family] = {
            "total_positive_pool": len(family_positives[family]),
            "train_count": len(split_records["train"][family]),
            "vale_count": len(split_records["vale"][family]),
            "test_count": len(split_records["test"][family]),
            "train_negative_match_reasons": dict(negative_reason_counters["train"][family]),
            "vale_negative_match_reasons": dict(negative_reason_counters["vale"][family]),
            "test_negative_match_reasons": dict(negative_reason_counters["test"][family]),
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"input={input_csv}")
    print(f"excluded_test_count={len(excluded_ids)}")
    print(f"output={output_dir}")
    for split_name in ("train", "vale", "test"):
        print(f"[{split_name}] combined={summary['splits'][split_name]['combined']}")
        for family in args.families:
            count = summary["splits"][split_name]["families"][family]
            print(f"  {family}: {count}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
