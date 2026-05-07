import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True, help="CodeQL database directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL for function spans.")
    parser.add_argument(
        "--query",
        type=Path,
        default=Path("qlpacks/vulcgbt-cpp/queries/function_spans.ql"),
        help="Path to the function spans query.",
    )
    parser.add_argument("--codeql", type=Path, required=True, help="Path to CodeQL CLI.")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=str(cwd), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    return result


def write_jsonl(path: Path, records: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    database = args.database.resolve()
    query = (root / args.query).resolve() if not args.query.is_absolute() else args.query.resolve()
    output = args.output.resolve()
    bqrs_path = output.with_suffix(".bqrs")

    run_command(
        [str(args.codeql), "query", "run", str(query), f"--database={database}", f"--output={bqrs_path}"],
        cwd=root,
    )
    decode = run_command(
        [str(args.codeql), "bqrs", "decode", str(bqrs_path), "--format=csv"],
        cwd=root,
    )

    rows = list(csv.reader(decode.stdout.splitlines()))
    records: list[dict[str, object]] = []
    for row in rows[1:]:
        if len(row) < 6:
            continue
        records.append(
            {
                "base_name": row[0],
                "relative_path": row[1],
                "qualified_name": row[2],
                "name": row[3],
                "start_line": int(row[4]),
                "end_line": int(row[5]),
            }
        )

    count = write_jsonl(output, records)
    summary = {
        "database": str(database),
        "query": str(query),
        "output": str(output),
        "bqrs": str(bqrs_path),
        "record_count": count,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"records={count}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
