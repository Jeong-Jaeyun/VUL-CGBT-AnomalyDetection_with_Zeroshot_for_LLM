import argparse
import json
import locale
import re
import shutil
import subprocess
import shutil as shutil_mod
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SUPPORTED_SOURCE_SUFFIXES = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL subset file.")
    parser.add_argument("--output-db", type=Path, required=True, help="Output CodeQL database directory.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Artifact directory for staged files, metadata, and logs.",
    )
    parser.add_argument(
        "--raw-source-root",
        type=Path,
        default=None,
        help="Optional root directory for loading raw source files referenced by record['source_file'].",
    )
    parser.add_argument(
        "--codeql",
        type=str,
        default="codeql",
        help="Path or command name for CodeQL CLI.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="cpp",
        help="CodeQL language for database creation.",
    )
    parser.add_argument(
        "--build-mode",
        type=str,
        default="manual",
        help="CodeQL build mode. Typical values: autobuild, manual.",
    )
    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Optional build command for compiled-language database creation.",
    )
    parser.add_argument("--c-compiler", type=str, default="gcc", help="C compiler for generated manual build scripts.")
    parser.add_argument("--cpp-compiler", type=str, default="g++", help="C++ compiler for generated manual build scripts.")
    parser.add_argument("--threads", type=int, default=0, help="CodeQL thread count. 0 keeps CLI default.")
    parser.add_argument("--ram", type=int, default=0, help="Optional RAM limit in MB for CodeQL.")
    parser.add_argument(
        "--source-extension",
        type=str,
        default=".c",
        help="File extension used when staging function-only records.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on staged records.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stage files and metadata only. Do not invoke CodeQL.",
    )
    parser.add_argument(
        "--skip-database-create",
        action="store_true",
        help="Reuse an existing CodeQL database directory and skip CLI invocation.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep staged source files after database creation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing CodeQL database directory.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path for summary JSON. Defaults to <artifact-dir>/summary.json.",
    )
    parser.add_argument(
        "--no-dedupe-raw-sources",
        action="store_true",
        help="Stage each raw source file once per record instead of sharing a single staged copy per source_file.",
    )
    parser.add_argument(
        "--allow-compile-errors",
        action="store_true",
        help="Continue the traced manual build even when individual compiler invocations fail.",
    )
    return parser.parse_args()


def resolve_codeql_path(codeql_arg: str) -> str:
    direct_path = Path(codeql_arg)
    if direct_path.exists():
        return str(direct_path.resolve())

    which_result = shutil_mod.which(codeql_arg)
    if which_result:
        return which_result

    user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
    candidates = [
        user_profile
        / "AppData"
        / "Roaming"
        / "Code"
        / "User"
        / "globalStorage"
        / "github.vscode-codeql"
        / "distribution1"
        / "codeql"
        / "codeql.exe",
        user_profile / "codeql" / "codeql.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    raise FileNotFoundError(f"Could not locate CodeQL CLI: {codeql_arg}")


def read_jsonl(path: Path, max_samples: int | None = None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def safe_file_stem(index: int, record_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", record_id).strip("_")
    normalized = normalized[:80] if normalized else "sample"
    return f"{index:06d}_{normalized}"


def infer_source_extension(record: Dict[str, object], default_extension: str) -> str:
    for field in ("file_name", "source_file"):
        suffix = Path(str(record.get(field, "") or "")).suffix.lower()
        if suffix in SUPPORTED_SOURCE_SUFFIXES:
            return suffix
    return default_extension


def normalize_rel_path(path_value: str) -> str:
    return path_value.replace("\\", "/").lstrip("./")


def copy_sibling_headers(raw_candidate: Path, staged_target: Path) -> None:
    source_dir = raw_candidate.parent
    target_dir = staged_target.parent
    if not source_dir.exists() or not target_dir.exists():
        return

    for header_path in source_dir.iterdir():
        if not header_path.is_file():
            continue
        if header_path.suffix.lower() not in {".h", ".hpp", ".hh", ".hxx"}:
            continue
        destination = target_dir / header_path.name
        if destination.exists():
            continue
        shutil.copyfile(header_path, destination)


def extract_target_method_name(record: Dict[str, object], default_name: str = "") -> str:
    function_name = str(record.get("function_name", "") or "")
    if function_name:
        return function_name

    source_code = str(record.get("func", "") or "")
    if not source_code:
        return default_name

    # Strip common comments that may appear between the signature and opening brace.
    cleaned_source = re.sub(r"/\*.*?\*/", " ", source_code, flags=re.S)
    cleaned_source = re.sub(r"//[^\n]*", " ", cleaned_source)

    match = re.search(
        r"(?ms)^\s*(?:[A-Za-z_][A-Za-z0-9_:\-<>\[\],*\s]+\s+)?([A-Za-z_~][A-Za-z0-9_~]*)\s*\([^;{}]*\)\s*\{",
        cleaned_source,
    )
    if match:
        candidate = match.group(1)
        if candidate not in {"if", "for", "while", "switch", "return"}:
            return candidate

    # Fallback: infer from the first function-like header before the first opening brace.
    header_region = cleaned_source.split("{", 1)[0]
    fallback_matches = re.findall(r"([A-Za-z_~][A-Za-z0-9_~]*)\s*\([^;{}]*\)", header_region, flags=re.S)
    for candidate in reversed(fallback_matches):
        if candidate not in {"if", "for", "while", "switch", "return", "sizeof"}:
            return candidate

    record_id = str(record.get("idx", "") or "")
    if "::" in record_id:
        candidate = record_id.rsplit("::", 1)[-1]
        if re.fullmatch(r"[A-Za-z_~][A-Za-z0-9_~]*", candidate):
            return candidate
    return default_name


def stage_records(
    records: List[Dict[str, object]],
    staged_dir: Path,
    raw_source_root: Path | None,
    default_extension: str,
    dedupe_raw_sources: bool,
) -> List[Dict[str, object]]:
    staged_dir.mkdir(parents=True, exist_ok=True)
    metadata: List[Dict[str, object]] = []
    shared_staged_paths: Dict[str, Path] = {}

    for index, record in enumerate(records):
        record_id = str(record.get("idx", index))
        source_file = str(record.get("source_file", "") or "")
        staged_source_origin = "function_only"
        file_stem = safe_file_stem(index, record_id)

        if raw_source_root is not None and source_file:
            raw_candidate = raw_source_root / source_file
            if raw_candidate.exists():
                staged_source_origin = "raw_source_file"
                normalized_source_file = normalize_rel_path(source_file)
                if dedupe_raw_sources:
                    shared_path = shared_staged_paths.get(normalized_source_file)
                    if shared_path is None:
                        shared_path = staged_dir / "raw_sources" / Path(normalized_source_file)
                        shared_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(raw_candidate, shared_path)
                        copy_sibling_headers(raw_candidate, shared_path)
                        shared_staged_paths[normalized_source_file] = shared_path
                    staged_path = shared_path
                else:
                    source_name = Path(source_file).name or f"{file_stem}{infer_source_extension(record, default_extension)}"
                    staged_path = staged_dir / "raw_records" / f"{index:06d}" / source_name
                    staged_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(raw_candidate, staged_path)
                    copy_sibling_headers(raw_candidate, staged_path)
            else:
                file_suffix = infer_source_extension(record, default_extension)
                staged_path = staged_dir / "functions" / f"{file_stem}{file_suffix}"
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                source_code = str(record.get("func", "") or "")
                staged_path.write_text(source_code.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
        else:
            file_suffix = infer_source_extension(record, default_extension)
            staged_path = staged_dir / "functions" / f"{file_stem}{file_suffix}"
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            source_code = str(record.get("func", "") or "")
            staged_path.write_text(source_code.rstrip("\n") + "\n", encoding="utf-8", newline="\n")

        metadata.append(
            {
                "idx": record_id,
                "target": int(record.get("target", 0)),
                "dataset": str(record.get("dataset", "")),
                "project": str(record.get("project", "")),
                "cwe_id": str(record.get("cwe_id", "")),
                "subset_name": str(record.get("subset_name", "")),
                "source_file": source_file,
                "staged_path": str(staged_path),
                "staged_rel_path": str(staged_path.relative_to(staged_dir)),
                "staged_file_name": staged_path.name,
                "staged_source_origin": staged_source_origin,
                "target_method": extract_target_method_name(record),
            }
        )

    return metadata


def run_command(
    command: List[str],
    cwd: Path | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, check=False, capture_output=True, text=False)
    preferred_encoding = locale.getpreferredencoding(False) or "utf-8"

    def decode_output(payload: bytes) -> str:
        for encoding in (preferred_encoding, "utf-8", "cp949"):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        return payload.decode(preferred_encoding, errors="replace")

    stdout_text = decode_output(result.stdout or b"")
    stderr_text = decode_output(result.stderr or b"")
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(stdout_text, encoding="utf-8")
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(stderr_text, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + stdout_text
            + "\nstderr:\n"
            + stderr_text
        )
    return result


def safe_object_stem(relative_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(relative_path)).strip("_")


def ensure_sard_support_headers(support_dir: Path) -> Dict[str, str]:
    support_dir.mkdir(parents=True, exist_ok=True)

    std_testcase_h = """#ifndef STD_TESTCASE_H
#define STD_TESTCASE_H

#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>

#define GLOBAL_CONST_TRUE 1
#define GLOBAL_CONST_FALSE 0
#define GLOBAL_CONST_FIVE 5
#define ALLOCA alloca

typedef struct _twoIntsStruct {
  int intOne;
  int intTwo;
} twoIntsStruct;

#ifdef __cplusplus
#include <new>
class OneIntClass {
public:
  int intOne;
};

class TwoIntsClass {
public:
  int intOne;
  int intTwo;
};

extern "C" {
#endif

extern int globalTrue;
extern int globalFalse;
extern int globalFive;

int globalReturnsTrue(void);
int globalReturnsFalse(void);
int globalReturnsTrueOrFalse(void);
int RAND32(void);

void printLine(const char *);
void printWLine(const wchar_t *);
void printIntLine(int);
void printShortLine(short);
void printLongLine(long);
void printLongLongLine(long long);
void printFloatLine(float);
void printDoubleLine(double);
void printSizeTLine(size_t);
void printHexCharLine(char);
void printUnsignedLine(unsigned);
void printHexUnsignedCharLine(unsigned char);
void printHexUnsignedShortLine(unsigned short);
void printHexUnsignedIntLine(unsigned int);
void printHexUnsignedLongLine(unsigned long);
void printHexUnsignedLongLongLine(unsigned long long);
void printStructLine(const void *);
void printBytesLine(const unsigned char *);
void printIntPointerLine(const int *);
void printLongLongIntLine(long long);
void printLongLongHexLine(long long);
void printUnsignedCharLine(unsigned char);
void printUnsignedShortLine(unsigned short);
void printUnsignedLongLine(unsigned long);
void printUnsignedLongLongLine(unsigned long long);
void printCharLine(char);
void printBoolLine(int);
void printHexLongLongLine(long long);

#ifdef __cplusplus
}
#endif

#endif
"""

    std_testcase_io_h = """#ifndef STD_TESTCASE_IO_H
#define STD_TESTCASE_IO_H

#include "std_testcase.h"

#endif
"""

    std_testcase_h_path = support_dir / "std_testcase.h"
    std_testcase_io_h_path = support_dir / "std_testcase_io.h"
    std_testcase_h_path.write_text(std_testcase_h, encoding="utf-8", newline="\n")
    std_testcase_io_h_path.write_text(std_testcase_io_h, encoding="utf-8", newline="\n")

    return {
        "support_dir": str(support_dir),
        "std_testcase_h": str(std_testcase_h_path),
        "std_testcase_io_h": str(std_testcase_io_h_path),
    }


def build_windows_manual_command(
    staged_dir: Path,
    artifact_dir: Path,
    c_compiler: str,
    cpp_compiler: str,
    allow_compile_errors: bool,
) -> Tuple[Path, str, Dict[str, str]]:
    build_script = artifact_dir / "build_staged_sources.cmd"
    build_dir = artifact_dir / "objects"
    build_dir.mkdir(parents=True, exist_ok=True)
    support_info = ensure_sard_support_headers(artifact_dir / "support")
    support_dir = Path(support_info["support_dir"])

    compile_lines = [
        "@echo off",
        "setlocal enabledelayedexpansion",
        f'cd /d "{staged_dir}"',
    ]

    source_files = sorted(
        path
        for path in staged_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"}
    )
    if not source_files:
        raise FileNotFoundError(f"No compilable source files found under {staged_dir}")

    compile_lines.append(f'if not exist "{build_dir}" mkdir "{build_dir}"')
    for source_path in source_files:
        relative_path = source_path.relative_to(staged_dir)
        object_name = safe_object_stem(relative_path) + ".obj"
        object_path = build_dir / object_name
        compiler = cpp_compiler if source_path.suffix.lower() in {".cc", ".cpp", ".cxx"} else c_compiler
        compile_lines.append(f'echo Compiling "{relative_path}"')
        compile_lines.append(f'{compiler} -c "{relative_path}" -I "{support_dir}" -o "{object_path}"')
        if allow_compile_errors:
            compile_lines.append(
                f'if errorlevel 1 echo Compiler failed for "{relative_path}", continuing for CodeQL tracing'
            )
        else:
            compile_lines.append("if errorlevel 1 exit /b 1")

    compile_lines.append("exit /b 0")
    build_script.write_text("\n".join(compile_lines) + "\n", encoding="utf-8", newline="\n")
    return build_script, f'cmd.exe /C "{build_script}"', support_info


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_path = args.input.resolve()
    output_db = args.output_db.resolve()
    artifact_dir = args.artifact_dir.resolve()
    raw_source_root = args.raw_source_root.resolve() if args.raw_source_root else None
    summary_path = args.summary_path.resolve() if args.summary_path else artifact_dir / "summary.json"
    codeql_path = resolve_codeql_path(args.codeql)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = artifact_dir / "staged_sources"
    metadata_path = artifact_dir / "metadata.jsonl"
    stdout_path = artifact_dir / "codeql-database-create.stdout.log"
    stderr_path = artifact_dir / "codeql-database-create.stderr.log"
    generated_build_script_path = artifact_dir / "build_staged_sources.cmd"
    support_files: Dict[str, str] = {}

    if staged_dir.exists() and not args.keep_staging:
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path, max_samples=args.max_samples)
    metadata_records = stage_records(
        records=records,
        staged_dir=staged_dir,
        raw_source_root=raw_source_root,
        default_extension=args.source_extension,
        dedupe_raw_sources=not args.no_dedupe_raw_sources,
    )
    write_jsonl(metadata_path, metadata_records)

    effective_build_mode = args.build_mode
    effective_command = args.command
    if not args.prepare_only and not args.skip_database_create and not effective_command and effective_build_mode == "manual":
        generated_build_script_path, effective_command, support_files = build_windows_manual_command(
            staged_dir=staged_dir,
            artifact_dir=artifact_dir,
            c_compiler=args.c_compiler,
            cpp_compiler=args.cpp_compiler,
            allow_compile_errors=args.allow_compile_errors,
        )

    if not args.prepare_only and not args.skip_database_create:
        should_reset_output_db = output_db.exists() and (args.overwrite or not summary_path.exists())
        if should_reset_output_db:
            shutil.rmtree(output_db)

        command = [
            codeql_path,
            "database",
            "create",
            str(output_db),
            f"--language={args.language}",
            f"--source-root={staged_dir}",
        ]
        if args.overwrite:
            command.append("--overwrite")
        if effective_build_mode and not effective_command:
            command.append(f"--build-mode={effective_build_mode}")
        if effective_command:
            command.extend(["--command", effective_command])
        if args.threads > 0:
            command.append(f"--threads={args.threads}")
        if args.ram > 0:
            command.append(f"--ram={args.ram}")

        run_command(command, cwd=root, stdout_path=stdout_path, stderr_path=stderr_path)

    if not args.keep_staging:
        shutil.rmtree(staged_dir, ignore_errors=True)

    summary = {
        "input": str(input_path),
        "output_db": str(output_db),
        "artifact_dir": str(artifact_dir),
        "raw_source_root": str(raw_source_root) if raw_source_root else None,
        "record_count": len(records),
        "metadata_count": len(metadata_records),
        "prepare_only": bool(args.prepare_only),
        "skip_database_create": bool(args.skip_database_create),
        "dedupe_raw_sources": not args.no_dedupe_raw_sources,
        "allow_compile_errors": bool(args.allow_compile_errors),
        "database_exists": output_db.exists(),
        "metadata_path": str(metadata_path),
        "staged_dir": str(staged_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "language": args.language,
        "build_mode": effective_build_mode,
        "command": effective_command,
        "codeql_path": codeql_path,
        "generated_build_script": str(generated_build_script_path) if generated_build_script_path.exists() else None,
        "support_files": support_files,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"records={len(records)}")
    print(f"metadata={metadata_path}")
    print(f"database={output_db}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
