import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PATTERNS: Dict[str, re.Pattern[str]] = {
    "classic_source_api": re.compile(r"\b(recv|recvfrom|read|fread|fgets|scanf|fscanf|sscanf|getenv)\s*\("),
    "numeric_conversion": re.compile(r"\b(atoi|atol|atoll|strtol|strtoll|strtoul|strtoull)\s*\("),
    "classic_sink_api": re.compile(
        r"\b(strcpy|strcat|sprintf|vsprintf|snprintf|strncpy|strncat|memcpy|memmove|system|popen|execve|execvp|gets|malloc|calloc|realloc|free)\s*\("
    ),
    "array_index": re.compile(r"\[[^\]]+\]"),
    "pointer_write": re.compile(r"\*\s*[A-Za-z_][A-Za-z0-9_]*\s*="),
    "field_write": re.compile(r"[A-Za-z_][A-Za-z0-9_.>\-]*\s*="),
    "loop": re.compile(r"\b(for|while)\b"),
    "param_deref": re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\*[^;\n]*\)|\b[A-Za-z_][A-Za-z0-9_]*\s*\*\s*[A-Za-z_][A-Za-z0-9_]*"),
    "struct_ptr": re.compile(r"->"),
    "custom_read_like": re.compile(
        r"\b(get_bits|get_bits1|avio_r[lb]\d+|xenstore_read_[A-Za-z_]+|i_stream_next_line|copy_from_user|memdup_user|skb_put_data)\s*\("
    ),
    "alloc_size_from_param": re.compile(
        r"\b(malloc|calloc|realloc|g_malloc0|kmalloc|kzalloc|vmalloc)\s*\([^;\n]*\b[A-Za-z_][A-Za-z0-9_]*\b"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Dataset graph input in the form name=path/to/file.jsonl. Repeat for multiple datasets.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional Markdown report output path.")
    parser.add_argument("--sample-limit", type=int, default=8, help="Fallback-positive sample count per dataset.")
    parser.add_argument("--sample-code-limit", type=int, default=1200, help="Max code chars per sample.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def parse_named_inputs(items: List[str]) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --input value: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        pairs.append((name.strip(), path))
    return pairs


def init_pattern_counter() -> Dict[str, int]:
    return {name: 0 for name in PATTERNS}


def rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def analyze_dataset(
    *,
    name: str,
    path: Path,
    rng: random.Random,
    sample_limit: int,
    sample_code_limit: int,
) -> Dict[str, object]:
    totals = {
        "records": 0,
        "positive_records": 0,
        "negative_records": 0,
    }
    origin_counts: Dict[str, int] = {}
    label_origin_counts: Dict[str, int] = {}
    safety_hint_counts: Dict[str, int] = {}
    pattern_counts = {
        "positive_fallback": init_pattern_counter(),
        "positive_codeql_path": init_pattern_counter(),
        "positive_heuristic_path": init_pattern_counter(),
    }
    pattern_totals = {
        "positive_fallback": 0,
        "positive_codeql_path": 0,
        "positive_heuristic_path": 0,
    }
    fallback_positive_samples: List[Dict[str, object]] = []

    for record in read_jsonl(path):
        totals["records"] += 1
        label = int(record.get("target", 0))
        label_name = "positive" if label == 1 else "negative"
        totals[f"{label_name}_records"] += 1

        origin = str(record.get("graph_origin", "fallback") or "fallback")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        label_origin_key = f"{label_name}_{origin}"
        label_origin_counts[label_origin_key] = label_origin_counts.get(label_origin_key, 0) + 1

        safety_hint = str(record.get("path_safety_hint", "none") or "none")
        safety_hint_counts[safety_hint] = safety_hint_counts.get(safety_hint, 0) + 1

        if label != 1:
            continue

        bucket = None
        if origin == "fallback":
            bucket = "positive_fallback"
        elif origin == "codeql_path":
            bucket = "positive_codeql_path"
        elif origin == "heuristic_path":
            bucket = "positive_heuristic_path"

        if bucket is None:
            continue

        pattern_totals[bucket] += 1
        code = str(record.get("code", "") or "")
        for pattern_name, pattern in PATTERNS.items():
            if pattern.search(code):
                pattern_counts[bucket][pattern_name] += 1

        if origin == "fallback":
            fallback_positive_samples.append(
                {
                    "idx": str(record.get("idx", "")),
                    "path_count": int(record.get("path_count", 0) or 0),
                    "dataset": str(record.get("dataset", "")),
                    "project": str(record.get("project", "")),
                    "code": code[:sample_code_limit].replace("\n", " "),
                }
            )

    sampled_fallbacks = rng.sample(fallback_positive_samples, min(sample_limit, len(fallback_positive_samples)))
    pattern_rates: Dict[str, List[Dict[str, object]]] = {}
    for bucket_name, counter in pattern_counts.items():
        denominator = pattern_totals[bucket_name]
        ranked = sorted(counter.items(), key=lambda item: item[1], reverse=True)
        pattern_rates[bucket_name] = [
            {
                "pattern": pattern_name,
                "count": count,
                "rate": round(rate(count, denominator), 4),
            }
            for pattern_name, count in ranked
        ]

    return {
        "name": name,
        "path": str(path),
        "totals": totals,
        "origin_counts": origin_counts,
        "label_origin_counts": label_origin_counts,
        "origin_rates": {key: round(rate(value, totals["records"]), 4) for key, value in origin_counts.items()},
        "safety_hint_counts": safety_hint_counts,
        "pattern_totals": pattern_totals,
        "pattern_rates": pattern_rates,
        "fallback_positive_samples": sampled_fallbacks,
    }


def render_markdown(datasets: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# Fallback Audit")
    lines.append("")

    for dataset in datasets:
        lines.append(f"## {dataset['name']}")
        totals = dataset["totals"]
        lines.append("")
        lines.append(
            f"- records: {totals['records']} | positive: {totals['positive_records']} | negative: {totals['negative_records']}"
        )

        origin_counts = dataset["origin_counts"]
        origin_rates = dataset["origin_rates"]
        origin_summary = ", ".join(
            f"{origin}={count} ({origin_rates.get(origin, 0.0):.1%})" for origin, count in sorted(origin_counts.items())
        )
        lines.append(f"- origins: {origin_summary}")

        label_origin_counts = dataset["label_origin_counts"]
        label_summary = ", ".join(f"{key}={value}" for key, value in sorted(label_origin_counts.items()))
        lines.append(f"- label/origin: {label_summary}")

        for bucket_name in ("positive_fallback", "positive_codeql_path", "positive_heuristic_path"):
            bucket_total = dataset["pattern_totals"].get(bucket_name, 0)
            if bucket_total <= 0:
                continue
            lines.append(f"- {bucket_name} top patterns (n={bucket_total}):")
            for item in dataset["pattern_rates"][bucket_name][:6]:
                lines.append(f"  - {item['pattern']}: {item['count']} ({item['rate']:.1%})")

        if dataset["fallback_positive_samples"]:
            lines.append("- sampled fallback positives:")
            for sample in dataset["fallback_positive_samples"]:
                code = sample["code"]
                lines.append(f"  - {sample['idx']} | project={sample['project']} | path_count={sample['path_count']}")
                lines.append(f"    - {code}")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    named_inputs = parse_named_inputs(args.input)
    rng = random.Random(args.seed)

    datasets = [
        analyze_dataset(
            name=name,
            path=path,
            rng=rng,
            sample_limit=args.sample_limit,
            sample_code_limit=args.sample_code_limit,
        )
        for name, path in named_inputs
    ]

    payload = {"datasets": datasets}

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(datasets), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
