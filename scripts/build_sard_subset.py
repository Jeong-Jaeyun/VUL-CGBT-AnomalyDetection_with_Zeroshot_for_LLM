import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


BUFFER_ERROR_CWE_CODES = {
    119,
    120,
    121,
    122,
    124,
    125,
    126,
    127,
    129,
    131,
    134,
    170,
    680,
    785,
    786,
    787,
    788,
    805,
}
TARGET_FAMILY_TO_CODES = {
    "CWE-119": BUFFER_ERROR_CWE_CODES,
    "CWE-78": {78},
}
SPLIT_NAMES = ("train", "vale", "test")
CWE_PREFIX_RE = re.compile(r"^CWE0*(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing prepared SARD split JSONL files.",
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
        help="Target CWE families to keep.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    return parser.parse_args()


def canonicalize_family(project_name: str, families: Iterable[str]) -> str | None:
    match = CWE_PREFIX_RE.match(project_name)
    if not match:
        return None

    numeric_cwe = int(match.group(1))
    for family in families:
        if numeric_cwe in TARGET_FAMILY_TO_CODES.get(family, set()):
            return family
    return None


def read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_record(record: Dict[str, object], family: str, split_name: str) -> Dict[str, object]:
    updated = dict(record)
    updated.update(
        {
            "source_dataset": "sard",
            "subset_name": "sard_cwe119_78",
            "research_split": split_name,
            "cwe_id": family,
            "cwe_variant": str(record.get("project", "")),
        }
    )
    return updated


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_dir = (args.input_dir or (root / "dataset" / "SARD")).resolve()
    output_dir = (args.output_dir or (root / "dataset" / "research_v1" / "sard_cwe119_78")).resolve()
    summary_path = args.summary_path or (output_dir / "summary.json")

    family_records_by_split: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        split_name: defaultdict(list) for split_name in SPLIT_NAMES
    }
    combined_by_split: Dict[str, List[Dict[str, object]]] = {split_name: [] for split_name in SPLIT_NAMES}
    original_variant_counts: Dict[str, Counter[str]] = {family: Counter() for family in args.families}

    for split_name in SPLIT_NAMES:
        input_path = input_dir / f"{split_name}.jsonl"
        for record in read_jsonl(input_path):
            family = canonicalize_family(str(record.get("project", "")), args.families)
            if family is None:
                continue

            updated = build_record(record, family, split_name)
            family_records_by_split[split_name][family].append(updated)
            combined_by_split[split_name].append(updated)
            original_variant_counts[family][str(record.get("project", ""))] += 1

    summary: Dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "families": args.families,
        "splits": {},
        "family_variants": {
            family: dict(counter.most_common())
            for family, counter in original_variant_counts.items()
        },
        "notes": [],
    }

    for split_name in SPLIT_NAMES:
        split_summary: Dict[str, object] = {
            "combined": write_jsonl(output_dir / f"{split_name}.jsonl", combined_by_split[split_name]),
            "families": {},
        }
        for family in args.families:
            family_key = family.lower().replace("-", "")
            count = write_jsonl(
                output_dir / f"{split_name}_{family_key}.jsonl",
                family_records_by_split[split_name][family],
            )
            split_summary["families"][family] = count
        summary["splits"][split_name] = split_summary

    for family in args.families:
        family_total = sum(
            int(summary["splits"][split_name]["families"][family])  # type: ignore[index]
            for split_name in SPLIT_NAMES
        )
        if family_total == 0:
            summary["notes"].append(f"No samples found for {family} in the local SARD subset.")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"input={input_dir}")
    print(f"output={output_dir}")
    for split_name in SPLIT_NAMES:
        print(f"[{split_name}] combined={summary['splits'][split_name]['combined']}")
        for family in args.families:
            count = summary["splits"][split_name]["families"][family]
            print(f"  {family}: {count}")
    for note in summary["notes"]:
        print(f"note={note}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
