import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


TARGET_FUNCTION_PLACEHOLDER = """predicate isMetadataTargetFunction(Function f) {\n  none()\n}\n"""
TARGET_PARAMETER_PLACEHOLDER = """predicate isMetadataTargetParameterSource(DataFlow::Node source) {\n  exists(Parameter p |\n    source.asParameter() = p and\n    isMetadataTargetFunction(p.getFunction())\n  )\n}\n"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True, help="CodeQL database directory.")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Artifact directory for logs and SARIF.")
    parser.add_argument(
        "--database-summary",
        type=Path,
        default=None,
        help="Optional summary JSON from build_codeql_database.py. Used to derive metadata-driven query variants.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional metadata JSONL. Overrides metadata path from --database-summary when provided.",
    )
    parser.add_argument(
        "--query",
        type=Path,
        default=Path("qlpacks/vulcgbt-cpp/queries/untrusted_flow.ql"),
        help="Path to the CodeQL query to run.",
    )
    parser.add_argument(
        "--pack-root",
        type=Path,
        default=Path("qlpacks/vulcgbt-cpp"),
        help="Path to the local query pack root.",
    )
    parser.add_argument("--codeql", type=str, default="codeql", help="Path or command name for CodeQL CLI.")
    parser.add_argument("--threads", type=int, default=0, help="CodeQL thread count. 0 keeps CLI default.")
    parser.add_argument("--ram", type=int, default=0, help="Optional RAM limit in MB for CodeQL.")
    parser.add_argument(
        "--metadata-pair-chunk-size",
        type=int,
        default=0,
        help="Optional chunk size for metadata target pairs. When > 0, split large generated queries into smaller chunks and merge SARIF runs.",
    )
    parser.add_argument("--rerun-pack-install", action="store_true", help="Always run `codeql pack install` before analysis.")
    parser.add_argument("--rerun", action="store_true", help="Force query reevaluation instead of using cached results.")
    parser.add_argument(
        "--include-helper-target-params",
        action="store_true",
        help="Also treat target methods ending with Source/Sink as metadata-driven parameter sources.",
    )
    return parser.parse_args()


def resolve_codeql_path(codeql_arg: str) -> str:
    direct_path = Path(codeql_arg)
    if direct_path.exists():
        return str(direct_path.resolve())

    which_result = shutil.which(codeql_arg)
    if which_result:
        return which_result

    candidate = (
        Path.home()
        / "AppData"
        / "Roaming"
        / "Code"
        / "User"
        / "globalStorage"
        / "github.vscode-codeql"
        / "distribution1"
        / "codeql"
        / "codeql.exe"
    )
    if candidate.exists():
        return str(candidate.resolve())

    raise FileNotFoundError(f"Could not locate CodeQL CLI: {codeql_arg}")


def run_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), check=False, capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def normalize_rel_path(path_value: str) -> str:
    return path_value.replace("\\", "/").lstrip("./")


def escape_ql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\"", "\\\"")


def iter_metadata_target_pairs(
    metadata_records: Iterable[dict[str, object]],
    include_helper_target_params: bool,
) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for record in metadata_records:
        staged_rel_path = normalize_rel_path(str(record.get("staged_rel_path", "") or ""))
        target_method = str(record.get("target_method", "") or "")
        if not staged_rel_path or not target_method:
            continue
        if not include_helper_target_params and (target_method.endswith("Sink") or target_method.endswith("Source")):
            continue
        pairs.add((staged_rel_path, target_method))
    return sorted(pairs)


def build_metadata_target_function_predicate_from_pairs(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return TARGET_FUNCTION_PLACEHOLDER

    clauses = [
        f'(f.getLocation().getFile().getRelativePath() = "{escape_ql_string(staged_rel_path)}" and '
        f'f.getName() = "{escape_ql_string(target_method)}")'
        for staged_rel_path, target_method in pairs
    ]

    def build_balanced_or(items: list[str]) -> str:
        if len(items) == 1:
            return items[0]
        midpoint = len(items) // 2
        left = build_balanced_or(items[:midpoint])
        right = build_balanced_or(items[midpoint:])
        return f"({left} or {right})"

    expression = build_balanced_or(clauses)
    return "\n".join(
        [
            "predicate isMetadataTargetFunction(Function f) {",
            f"  {expression}",
            "}",
            "",
        ]
    )


def build_metadata_parameter_source_predicate_from_pairs(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return "predicate isMetadataTargetParameterSource(DataFlow::Node source) {\n  none()\n}\n"

    lines = [
        "predicate isMetadataTargetParameterSource(DataFlow::Node source) {",
        "  exists(Parameter p |",
        "    isMetadataTargetFunction(p.getFunction()) and",
        "    (",
        "      source.asParameter() = p",
        "      or",
        "      source.asParameter(1) = p",
        "    )",
        "  )",
        "}",
        "",
    ]
    return "\n".join(lines)


def chunk_pairs(pairs: list[tuple[str, str]], chunk_size: int) -> list[list[tuple[str, str]]]:
    if chunk_size <= 0 or len(pairs) <= chunk_size:
        return [pairs]
    return [pairs[index:index + chunk_size] for index in range(0, len(pairs), chunk_size)]


def write_generated_query(
    query_text: str,
    pack_root: Path,
    artifact_dir: Path,
    query_stem: str,
    pairs: list[tuple[str, str]],
    chunk_index: int | None = None,
) -> Path:
    target_function_predicate = build_metadata_target_function_predicate_from_pairs(pairs)
    parameter_source_predicate = build_metadata_parameter_source_predicate_from_pairs(pairs)

    generated_query_text = query_text.replace(TARGET_FUNCTION_PLACEHOLDER, target_function_predicate)
    generated_query_text = generated_query_text.replace(TARGET_PARAMETER_PLACEHOLDER, parameter_source_predicate)

    generated_dir = pack_root / "queries" / "_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".chunk{chunk_index:03d}" if chunk_index is not None else ""
    generated_query_path = generated_dir / f"{artifact_dir.name}.{query_stem}{suffix}.generated.ql"
    generated_query_path.write_text(generated_query_text, encoding="utf-8", newline="\n")
    return generated_query_path


def maybe_generate_metadata_queries(
    query_path: Path,
    pack_root: Path,
    artifact_dir: Path,
    metadata_path: Path | None,
    include_helper_target_params: bool,
    metadata_pair_chunk_size: int,
) -> tuple[list[Path], int, int]:
    if metadata_path is None or not metadata_path.exists():
        return [query_path], 0, 1
    if query_path.name != "untrusted_flow.ql":
        return [query_path], 0, 1

    query_text = query_path.read_text(encoding="utf-8")
    if TARGET_FUNCTION_PLACEHOLDER not in query_text or TARGET_PARAMETER_PLACEHOLDER not in query_text:
        return [query_path], 0, 1

    metadata_records = read_jsonl(metadata_path)
    pairs = iter_metadata_target_pairs(metadata_records, include_helper_target_params)
    pair_chunks = chunk_pairs(pairs, metadata_pair_chunk_size)
    generated_query_paths = [
        write_generated_query(
            query_text=query_text,
            pack_root=pack_root,
            artifact_dir=artifact_dir,
            query_stem=query_path.stem,
            pairs=pair_chunk,
            chunk_index=index if len(pair_chunks) > 1 else None,
        )
        for index, pair_chunk in enumerate(pair_chunks)
    ]
    return generated_query_paths, len(pairs), len(pair_chunks)


def read_text_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_sarif_runs(source_paths: list[Path], destination_path: Path) -> None:
    merged_payload: dict[str, object] | None = None
    merged_runs: list[dict[str, object]] = []

    for source_path in source_paths:
        payload = read_text_json(source_path)
        runs = payload.get("runs", []) or []
        if merged_payload is None:
            merged_payload = payload
            merged_payload["runs"] = []
        merged_runs.extend(runs)

    if merged_payload is None:
        merged_payload = {"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "runs": []}
    merged_payload["runs"] = merged_runs
    destination_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")


def analyze_query(
    *,
    codeql_path: str,
    database: Path,
    query_path: Path,
    cwd: Path,
    output_sarif_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    rerun: bool,
    threads: int,
    ram: int,
) -> None:
    command = [
        codeql_path,
        "database",
        "analyze",
        str(database),
        str(query_path),
        "--format=sarif-latest",
        f"--output={output_sarif_path}",
    ]
    if rerun:
        command.append("--rerun")
    if threads > 0:
        command.append(f"--threads={threads}")
    if ram > 0:
        command.append(f"--ram={ram}")
    run_command(command, cwd=cwd, stdout_path=stdout_path, stderr_path=stderr_path)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact_dir = args.artifact_dir.resolve()
    database = args.database.resolve()
    query_path = (root / args.query).resolve() if not args.query.is_absolute() else args.query.resolve()
    pack_root = (root / args.pack_root).resolve() if not args.pack_root.is_absolute() else args.pack_root.resolve()
    codeql_path = resolve_codeql_path(args.codeql)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_path: Path | None = None
    database_summary_path = args.database_summary.resolve() if args.database_summary else None
    if args.metadata_path is not None:
        metadata_path = args.metadata_path.resolve()
    elif database_summary_path is not None and database_summary_path.exists():
        database_summary = read_json(database_summary_path)
        summary_metadata_path = database_summary.get("metadata_path")
        if summary_metadata_path:
            metadata_path = Path(str(summary_metadata_path)).resolve()

    query_paths, metadata_pair_count, query_chunk_count = maybe_generate_metadata_queries(
        query_path=query_path,
        pack_root=pack_root,
        artifact_dir=artifact_dir,
        metadata_path=metadata_path,
        include_helper_target_params=args.include_helper_target_params,
        metadata_pair_chunk_size=args.metadata_pair_chunk_size,
    )

    query_path = query_paths[0]
    sarif_path = artifact_dir / f"{query_path.stem}.sarif"
    pack_install_stdout = artifact_dir / "codeql-pack-install.stdout.log"
    pack_install_stderr = artifact_dir / "codeql-pack-install.stderr.log"
    analyze_stdout = artifact_dir / "codeql-analyze.stdout.log"
    analyze_stderr = artifact_dir / "codeql-analyze.stderr.log"

    lock_file = pack_root / "codeql-pack.lock.yml"
    if args.rerun_pack_install or not lock_file.exists():
        run_command(
            [codeql_path, "pack", "install"],
            cwd=pack_root,
            stdout_path=pack_install_stdout,
            stderr_path=pack_install_stderr,
        )

    if len(query_paths) == 1:
        analyze_query(
            codeql_path=codeql_path,
            database=database,
            query_path=query_path,
            cwd=root,
            output_sarif_path=sarif_path,
            stdout_path=analyze_stdout,
            stderr_path=analyze_stderr,
            rerun=args.rerun,
            threads=args.threads,
            ram=args.ram,
        )
        chunk_summaries: list[dict[str, object]] = []
    else:
        chunk_dir = artifact_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_sarif_paths: list[Path] = []
        chunk_summaries = []
        for chunk_index, chunk_query_path in enumerate(query_paths):
            chunk_name = f"chunk{chunk_index:03d}"
            chunk_sarif_path = chunk_dir / f"{chunk_query_path.stem}.sarif"
            chunk_stdout_path = chunk_dir / f"{chunk_name}.codeql-analyze.stdout.log"
            chunk_stderr_path = chunk_dir / f"{chunk_name}.codeql-analyze.stderr.log"
            analyze_query(
                codeql_path=codeql_path,
                database=database,
                query_path=chunk_query_path,
                cwd=root,
                output_sarif_path=chunk_sarif_path,
                stdout_path=chunk_stdout_path,
                stderr_path=chunk_stderr_path,
                rerun=args.rerun,
                threads=args.threads,
                ram=args.ram,
            )
            chunk_sarif_paths.append(chunk_sarif_path)
            chunk_summaries.append(
                {
                    "chunk_index": chunk_index,
                    "query": str(chunk_query_path),
                    "sarif": str(chunk_sarif_path),
                    "analyze_stdout_log": str(chunk_stdout_path),
                    "analyze_stderr_log": str(chunk_stderr_path),
                }
            )

        merge_sarif_runs(chunk_sarif_paths, sarif_path)
        analyze_stdout.write_text("chunked execution; see artifact_dir/chunks for per-chunk logs\n", encoding="utf-8")
        analyze_stderr.write_text("chunked execution; see artifact_dir/chunks for per-chunk logs\n", encoding="utf-8")

    summary = {
        "database": str(database),
        "query": str(query_path),
        "queries": [str(path) for path in query_paths],
        "database_summary": str(database_summary_path) if database_summary_path else None,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "metadata_pair_count": metadata_pair_count,
        "metadata_pair_chunk_size": args.metadata_pair_chunk_size,
        "query_chunk_count": query_chunk_count,
        "include_helper_target_params": args.include_helper_target_params,
        "pack_root": str(pack_root),
        "codeql_path": codeql_path,
        "sarif": str(sarif_path),
        "analyze_stdout_log": str(analyze_stdout),
        "analyze_stderr_log": str(analyze_stderr),
        "chunk_summaries": chunk_summaries,
        "pack_install_stdout_log": str(pack_install_stdout),
        "pack_install_stderr_log": str(pack_install_stderr),
        "pack_lock_exists": lock_file.exists(),
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"sarif={sarif_path}")
    print(f"summary={artifact_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
