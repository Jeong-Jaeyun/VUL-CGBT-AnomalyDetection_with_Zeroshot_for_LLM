import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--vale-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--vale-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stratified_sample(records: list[dict], per_class: int, rng: random.Random) -> list[dict]:
    by_target: dict[int, list[dict]] = {0: [], 1: []}
    for record in records:
        target = int(record.get("target", record.get("label", 0)))
        if target in by_target:
            by_target[target].append(record)

    sampled: list[dict] = []
    for target in (0, 1):
        candidates = list(by_target[target])
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise ValueError(
                f"Not enough samples for class {target}: requested {per_class}, found {len(candidates)}"
            )
        sampled.extend(candidates[:per_class])

    rng.shuffle(sampled)
    return sampled


def summarize(records: list[dict]) -> dict[str, int]:
    counts = {0: 0, 1: 0}
    for record in records:
        counts[int(record.get("target", record.get("label", 0)))] += 1
    return {"total": len(records), "negative": counts[0], "positive": counts[1]}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_records = read_jsonl(args.train_input)
    vale_records = read_jsonl(args.vale_input)

    sampled_train = stratified_sample(train_records, args.train_per_class, rng)
    sampled_vale = stratified_sample(vale_records, args.vale_per_class, rng)

    output_dir = args.output_dir
    train_path = output_dir / "train.jsonl"
    vale_path = output_dir / "vale.jsonl"
    summary_path = output_dir / "summary.json"

    write_jsonl(train_path, sampled_train)
    write_jsonl(vale_path, sampled_vale)

    summary = {
        "seed": args.seed,
        "train_per_class": args.train_per_class,
        "vale_per_class": args.vale_per_class,
        "train": summarize(sampled_train),
        "vale": summarize(sampled_vale),
        "train_input": str(args.train_input),
        "vale_input": str(args.vale_input),
        "train_output": str(train_path),
        "vale_output": str(vale_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"train={train_path}")
    print(f"vale={vale_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
