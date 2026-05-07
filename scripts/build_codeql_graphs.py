import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SOURCE_LINE_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"\brecv\s*\(",
        r"\brecvfrom\s*\(",
        r"\bread\s*\(",
        r"\bfgets\s*\(",
        r"\bfread\s*\(",
        r"\bscanf\s*\(",
        r"\bfscanf\s*\(",
        r"\bsscanf\s*\(",
        r"\bgetenv\s*\(",
    )
]
CONVERSION_LINE_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"\batoi\s*\(",
        r"\batol\s*\(",
        r"\batoll\s*\(",
        r"\bstrtol\s*\(",
        r"\bstrtoll\s*\(",
        r"\bstrtoul\s*\(",
        r"\bstrtoull\s*\(",
    )
]
CALL_NAME_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CALL_SKIP_NAMES = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "catch",
}
FUNCTION_HEADER_SKIP_PREFIXES = CALL_SKIP_NAMES | {"do"}
ARRAY_ACCESS_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]")
POINTER_WRITE_PATTERN = re.compile(r"\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
FIELD_WRITE_ASSIGN_PATTERN = re.compile(r"(->|\.)\s*[A-Za-z_][A-Za-z0-9_]*\s*=")
READ_LIKE_CALL_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"\bget_bits1?\s*\(",
        r"\bavio_r[lb]\d+\s*\(",
        r"\bxenstore_read_[A-Za-z_][A-Za-z0-9_]*\s*\(",
        r"\bi_stream_next_line\s*\(",
        r"\bcopy_from_user\s*\(",
        r"\bmemdup_user\s*\(",
        r"\b(parse|decode)_[A-Za-z_][A-Za-z0-9_]*\s*\(",
    )
]
EXEC_SINK_CALL_NAMES = {"system", "popen", "execve", "execvp"}
FORMAT_SINK_CALL_NAMES = {"sprintf", "snprintf", "vsprintf"}
COPY_SINK_CALL_NAMES = {"memcpy", "memmove", "strcpy", "strcat", "strncpy", "strncat"}
HIGH_RISK_SINK_CALL_NAMES = EXEC_SINK_CALL_NAMES | FORMAT_SINK_CALL_NAMES | COPY_SINK_CALL_NAMES
FALLBACK_SINK_LINE_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"\bmemcpy\s*\(",
        r"\bmemmove\s*\(",
        r"\bstrcpy\s*\(",
        r"\bstrcat\s*\(",
        r"\bsprintf\s*\(",
        r"\bvsprintf\s*\(",
        r"\bsnprintf\s*\(",
        r"\bstrncpy\s*\(",
        r"\bstrncat\s*\(",
        r"\bsystem\s*\(",
        r"\bpopen\s*\(",
        r"\bexecve\s*\(",
        r"\bexecvp\s*\(",
        r"\bmalloc\s*\(",
        r"\bcalloc\s*\(",
        r"\brealloc\s*\(",
        r"\bfree\s*\(",
    )
]
DIRECT_SINK_LINE_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"\bgets\s*\(",
    )
]
GUARD_LINE_PATTERN = re.compile(r"\b(if|while|for)\b")
COMPARISON_OPERATOR_PATTERN = re.compile(r"(<=|>=|==|!=|<|>)")
FALLBACK_MAX_NODES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-summary", type=Path, required=True, help="Summary JSON from build_codeql_database.py")
    parser.add_argument("--query-summary", type=Path, required=True, help="Summary JSON from codeql_extract_paths.py")
    parser.add_argument("--function-spans", type=Path, default=None, help="Optional JSONL with CodeQL-extracted function spans.")
    parser.add_argument("--output", type=Path, required=True, help="Output graph JSONL path.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional summary JSON output.")
    parser.add_argument("--max-paths-per-sample", type=int, default=16, help="Maximum SARIF paths to merge per sample.")
    parser.add_argument("--max-steps-per-path", type=int, default=64, help="Maximum thread-flow steps to keep per SARIF path.")
    parser.add_argument("--edge-type", type=str, default="CODEQL_FLOW", help="Edge type label for sequential path steps.")
    parser.add_argument(
        "--family-edge-types",
        action="store_true",
        help="Encode inferred vulnerability family in path edge labels.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def load_function_spans(path: Path | None) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    if path is None or not path.exists():
        return {}
    span_index: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for record in read_jsonl(path):
        relative_path = normalize_rel_path(str(record.get("relative_path", "") or ""))
        name = str(record.get("name", "") or "")
        start_line = int(record.get("start_line", 0) or 0)
        end_line = int(record.get("end_line", 0) or 0)
        if not relative_path or not name or start_line <= 0 or end_line <= 0:
            continue
        span_index.setdefault((Path(relative_path).name, name), []).append((start_line, end_line))
    return span_index


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_rel_path(path_value: str) -> str:
    return path_value.replace("\\", "/").lstrip("./")


def load_source_lines(path: Path | None, cache: Dict[object, List[str]], inline_source_text: str = "") -> List[str]:
    cache_key: object = path if path is not None and path.exists() else ("inline_source", inline_source_text)
    if cache_key not in cache:
        if path is not None and path.exists():
            cache[cache_key] = path.read_text(encoding="utf-8", errors="replace").splitlines()
        else:
            cache[cache_key] = inline_source_text.splitlines()
    return cache[cache_key]


def strip_comments_and_literals(lines: List[str]) -> List[str]:
    stripped: List[str] = []
    in_block_comment = False
    for line in lines:
        current: List[str] = []
        index = 0
        while index < len(line):
            char = line[index]
            next_char = line[index + 1] if index + 1 < len(line) else ""

            if in_block_comment:
                if char == "*" and next_char == "/":
                    in_block_comment = False
                    index += 2
                else:
                    index += 1
                continue

            if char == "/" and next_char == "*":
                in_block_comment = True
                index += 2
                continue
            if char == "/" and next_char == "/":
                break

            if char in {'"', "'"}:
                quote = char
                current.append(" ")
                index += 1
                while index < len(line):
                    if line[index] == "\\":
                        index += 2
                        continue
                    if line[index] == quote:
                        index += 1
                        break
                    index += 1
                continue

            current.append(char)
            index += 1
        stripped.append("".join(current))
    return stripped


def load_stripped_source_lines(
    path: Path | None,
    source_cache: Dict[object, List[str]],
    stripped_cache: Dict[object, List[str]],
    inline_source_text: str = "",
) -> List[str]:
    cache_key: object = path if path is not None and path.exists() else ("inline_source", inline_source_text)
    if cache_key not in stripped_cache:
        stripped_cache[cache_key] = strip_comments_and_literals(
            load_source_lines(path, source_cache, inline_source_text)
        )
    return stripped_cache[cache_key]


def get_line_text(
    path: Path | None,
    line_number: int | None,
    cache: Dict[object, List[str]],
    inline_source_text: str = "",
) -> str:
    if line_number is None or line_number <= 0:
        return ""
    lines = load_source_lines(path, cache, inline_source_text)
    if line_number > len(lines):
        return ""
    return lines[line_number - 1].strip()


def resolve_source_path(
    metadata_record: Dict[str, object],
    staged_root: Path,
    raw_source_root: Path | None,
) -> Path | None:
    staged_rel_path = normalize_rel_path(str(metadata_record.get("staged_rel_path", "") or ""))
    if staged_rel_path:
        staged_path = staged_root / Path(staged_rel_path)
        if staged_path.exists():
            return staged_path
    source_file = str(metadata_record.get("source_file", "") or "")
    if raw_source_root is not None and source_file:
        raw_path = raw_source_root / source_file
        if raw_path.exists():
            return raw_path
    staged_path_value = str(metadata_record.get("staged_path", "") or "")
    if staged_path_value:
        direct_path = Path(staged_path_value)
        if direct_path.exists():
            return direct_path
    return None


def infer_method_span_from_source(
    source_path: Path | None,
    target_method: str,
    source_cache: Dict[object, List[str]],
    stripped_cache: Dict[object, List[str]],
    inline_source_text: str = "",
) -> List[Tuple[int, int]]:
    if (source_path is None and not inline_source_text) or not target_method:
        return []

    raw_lines = load_source_lines(source_path, source_cache, inline_source_text)
    stripped_lines = load_stripped_source_lines(source_path, source_cache, stripped_cache, inline_source_text)
    name_pattern = re.compile(rf"\b{re.escape(target_method)}\s*\(")
    spans: List[Tuple[int, int]] = []

    for start_index, line in enumerate(stripped_lines):
        if not name_pattern.search(line):
            continue

        header_parts: List[str] = []
        brace_line_index: int | None = None
        prototype_only = False

        for probe_index in range(start_index, min(len(stripped_lines), start_index + 20)):
            probe_line = stripped_lines[probe_index]
            header_parts.append(probe_line)
            combined = " ".join(header_parts)
            if "{" in combined:
                brace_line_index = probe_index
                break
            if ";" in combined:
                prototype_only = True
                break

        if prototype_only or brace_line_index is None:
            continue

        header = " ".join(header_parts).split("{", 1)[0]
        if not looks_like_function_header(header, target_method):
            continue

        brace_depth = 0
        seen_open_brace = False
        for end_index in range(brace_line_index, len(stripped_lines)):
            for char in stripped_lines[end_index]:
                if char == "{":
                    brace_depth += 1
                    seen_open_brace = True
                elif char == "}" and seen_open_brace:
                    brace_depth -= 1
                    if brace_depth == 0:
                        spans.append((start_index + 1, end_index + 1))
                        break
            if spans and spans[-1][0] == start_index + 1:
                break

    return spans


def looks_like_function_header(header: str, target_method: str) -> bool:
    stripped = header.strip()
    if not stripped or stripped.startswith("#"):
        return False

    method_match = re.search(rf"\b{re.escape(target_method)}\s*\(", stripped)
    if not method_match:
        return False

    prefix = stripped[: method_match.start()]
    prefix_identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prefix)
    if not prefix_identifiers:
        return False
    if prefix_identifiers[-1] in FUNCTION_HEADER_SKIP_PREFIXES:
        return False

    return True


def extract_function_parameter_names(
    source_path: Path | None,
    target_method: str,
    span: Tuple[int, int],
    source_cache: Dict[object, List[str]],
    stripped_cache: Dict[object, List[str]],
    inline_source_text: str = "",
) -> List[str]:
    if (source_path is None and not inline_source_text) or not target_method:
        return []

    stripped_lines = load_stripped_source_lines(source_path, source_cache, stripped_cache, inline_source_text)
    start_line, _ = span
    header_parts: List[str] = []
    header = ""
    for probe_index in range(start_line - 1, min(len(stripped_lines), start_line + 20)):
        probe_line = stripped_lines[probe_index]
        header_parts.append(probe_line)
        combined = " ".join(header_parts)
        if "{" in combined or ";" in combined:
            header = combined.split("{", 1)[0]
            break
    if not header:
        header = " ".join(header_parts)

    method_match = re.search(rf"\b{re.escape(target_method)}\s*\((.*)\)", header)
    if not method_match:
        return []

    parameter_blob = method_match.group(1).strip()
    if not parameter_blob or parameter_blob == "void":
        return []

    parameter_names: List[str] = []
    for chunk in parameter_blob.split(","):
        candidate = chunk.split("=")[0].strip()
        if not candidate or candidate == "void":
            continue
        identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", candidate)
        if not identifiers:
            continue
        last_identifier = identifiers[-1]
        if last_identifier in {
            "const",
            "volatile",
            "unsigned",
            "signed",
            "int",
            "char",
            "short",
            "long",
            "float",
            "double",
            "bool",
            "void",
            "struct",
            "class",
        }:
            continue
        parameter_names.append(last_identifier)
    return parameter_names


def has_upper_bound_guard(lines: List[str], sink_line: int, variable_name: str, window_start: int) -> bool:
    guard_pattern = re.compile(rf"\b{re.escape(variable_name)}\b\s*(<|<=)")
    reverse_guard_pattern = re.compile(rf"(<|<=)\s*\(*\s*\b{re.escape(variable_name)}\b")
    start_line = max(window_start, sink_line - 6)
    for line_number in range(start_line, sink_line):
        line = lines[line_number - 1]
        if "if" not in line:
            continue
        if guard_pattern.search(line) or reverse_guard_pattern.search(line):
            return True
    return False


def has_lower_bound_guard(lines: List[str], sink_line: int, variable_name: str, window_start: int) -> bool:
    direct_pattern = re.compile(rf"\b{re.escape(variable_name)}\b\s*(>=|>)\s*0\b")
    reverse_pattern = re.compile(rf"\b0\b\s*(<=|<)\s*\(*\s*\b{re.escape(variable_name)}\b")
    start_line = max(window_start, sink_line - 6)
    for line_number in range(start_line, sink_line):
        line = lines[line_number - 1]
        if "if" not in line:
            continue
        if direct_pattern.search(line) or reverse_pattern.search(line):
            return True
    return False


def has_complete_bounds_guard(lines: List[str], sink_line: int, variable_name: str, window_start: int) -> bool:
    return has_lower_bound_guard(lines, sink_line, variable_name, window_start) and has_upper_bound_guard(
        lines, sink_line, variable_name, window_start
    )


def line_has_tainted_allocation(line: str, variable_name: str | None) -> bool:
    if not variable_name:
        return False
    array_alloc_pattern = re.compile(rf"\bnew\b[^\[]*\[\s*{re.escape(variable_name)}\s*\]")
    call_alloc_pattern = re.compile(rf"\b(malloc|calloc|realloc)\s*\([^)]*\b{re.escape(variable_name)}\b")
    return bool(array_alloc_pattern.search(line) or call_alloc_pattern.search(line))


def infer_local_taint_variable(lines: List[str], span: Tuple[int, int]) -> str | None:
    start_line, end_line = span
    candidate_indexes: List[str] = []
    seen_candidates = set()
    for line_number in range(start_line, end_line + 1):
        line = lines[line_number - 1]
        array_match = ARRAY_ACCESS_PATTERN.search(line)
        if array_match:
            index_name = array_match.group(2)
            if index_name not in seen_candidates:
                seen_candidates.add(index_name)
                candidate_indexes.append(index_name)

    for candidate in candidate_indexes:
        assignment_pattern = re.compile(rf"\b{re.escape(candidate)}\b\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\b")
        for line_number in range(start_line, end_line + 1):
            line = lines[line_number - 1]
            if any(operator in line for operator in ("==", "!=", ">=", "<=")):
                continue
            assignment_match = assignment_pattern.search(line)
            if not assignment_match:
                continue
            rhs_name = assignment_match.group(1)
            if rhs_name != candidate:
                return candidate

    if len(candidate_indexes) == 1:
        return candidate_indexes[0]
    return None


def infer_unsafe_sink_lines(lines: List[str], span: Tuple[int, int], variable_name: str | None = None) -> List[int]:
    sink_lines: List[int] = []
    start_line, end_line = span
    for line_number in range(start_line, end_line + 1):
        line = lines[line_number - 1]
        array_match = ARRAY_ACCESS_PATTERN.search(line)
        if array_match:
            index_name = array_match.group(2)
            if variable_name and index_name != variable_name:
                continue
            if not has_complete_bounds_guard(lines, line_number, index_name, start_line):
                sink_lines.append(line_number)
                continue
        if line_has_tainted_allocation(line, variable_name):
            if not has_complete_bounds_guard(lines, line_number, str(variable_name), start_line):
                sink_lines.append(line_number)
    return sink_lines


def infer_guarded_sink_lines(lines: List[str], span: Tuple[int, int], variable_name: str | None = None) -> List[int]:
    guarded_lines: List[int] = []
    start_line, end_line = span
    for line_number in range(start_line, end_line + 1):
        line = lines[line_number - 1]
        array_match = ARRAY_ACCESS_PATTERN.search(line)
        if array_match:
            index_name = array_match.group(2)
            if variable_name and index_name != variable_name:
                continue
            if has_complete_bounds_guard(lines, line_number, index_name, start_line):
                guarded_lines.append(line_number)
                continue
        if line_has_tainted_allocation(line, variable_name):
            if has_complete_bounds_guard(lines, line_number, str(variable_name), start_line):
                guarded_lines.append(line_number)
    return guarded_lines


def extract_simple_assignment_target(line: str) -> str | None:
    if "=" not in line or any(operator in line for operator in ("==", "!=", ">=", "<=")):
        return None
    left = line.split("=", 1)[0]
    if "[" in left or "->" in left or "." in left:
        return None
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", left)
    if not identifiers:
        return None
    candidate = identifiers[-1]
    if candidate in {
        "const",
        "volatile",
        "unsigned",
        "signed",
        "int",
        "char",
        "short",
        "long",
        "float",
        "double",
        "bool",
        "void",
        "struct",
        "class",
        "return",
    }:
        return None
    return candidate


def split_assignment_expression(line: str) -> Tuple[str, str] | None:
    if "=" not in line or any(operator in line for operator in ("==", "!=", ">=", "<=")):
        return None
    left, right = line.split("=", 1)
    left = left.strip()
    right = right.strip().rstrip(";")
    if not left or not right:
        return None
    return left, right


def line_references_tainted_name(line: str, tainted_names: Dict[str, int]) -> bool:
    return any(re.search(rf"\b{re.escape(name)}\b", line) for name in tainted_names)


def extract_call_names(line: str) -> List[str]:
    return [call_name for call_name in CALL_NAME_PATTERN.findall(line) if call_name not in CALL_SKIP_NAMES]


def line_has_read_like_signal(line: str) -> bool:
    if any(pattern.search(line) for pattern in SOURCE_LINE_PATTERNS):
        return True
    if any(pattern.search(line) for pattern in CONVERSION_LINE_PATTERNS):
        return True
    if any(pattern.search(line) for pattern in READ_LIKE_CALL_PATTERNS):
        return True
    return False


def line_has_command_sink(line: str) -> bool:
    return any(re.search(rf"\b{re.escape(name)}\s*\(", line) for name in EXEC_SINK_CALL_NAMES)


def line_has_buffer_api_sink(line: str) -> bool:
    return any(re.search(rf"\b{re.escape(name)}\s*\(", line) for name in COPY_SINK_CALL_NAMES | FORMAT_SINK_CALL_NAMES)


def line_has_exact_pointer_write(line: str) -> bool:
    assignment = split_assignment_expression(line)
    if assignment is None:
        return False
    left, _ = assignment
    return extract_pointer_write_target_name(left) is not None


def line_has_array_signal(line: str) -> bool:
    return bool(ARRAY_ACCESS_PATTERN.search(line))


def line_has_field_write_signal(line: str) -> bool:
    assignment = split_assignment_expression(line)
    if assignment is None:
        return False
    left, right = assignment
    return extract_field_write_base(left) is not None and line_has_read_like_signal(right)


def extract_pointer_write_target_name(left: str) -> str | None:
    match = re.match(r"^\s*(?:\(\s*)?\*\s*([A-Za-z_][A-Za-z0-9_]*)\b", left)
    if not match:
        return None
    return match.group(1)


def extract_array_write_target(left: str) -> Tuple[str, str] | None:
    match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^\]]+)\s*\]\s*$", left)
    if not match:
        return None
    return match.group(1), match.group(2)


def extract_field_write_base(left: str) -> str | None:
    match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(->|\.)\s*[A-Za-z_][A-Za-z0-9_]*\s*$", left)
    if not match:
        return None
    return match.group(1)


def classify_path_family(
    path: List[Dict[str, object]],
    source_path: Path | None,
    source_cache: Dict[object, List[str]],
    inline_source_text: str = "",
) -> str:
    source_lines = load_source_lines(source_path, source_cache, inline_source_text)
    path_lines: List[str] = []
    for step in path:
        line_number = int(step.get("line") or 0)
        if 1 <= line_number <= len(source_lines):
            path_lines.append(source_lines[line_number - 1].strip())
        else:
            path_lines.append(str(step.get("message", "") or ""))

    combined = "\n".join(path_lines)
    if any(line_has_command_sink(line) for line in path_lines):
        return "COMMAND_INJECTION"
    if any(line_has_buffer_api_sink(line) for line in path_lines):
        return "BUFFER_API"
    if any(line_has_exact_pointer_write(line) for line in path_lines):
        return "POINTER_WRITE"
    if any(line_has_array_signal(line) for line in path_lines):
        if re.search(r"\b(for|while)\b", combined) or any(COMPARISON_OPERATOR_PATTERN.search(line) for line in path_lines):
            return "INDEX_FLOW"
        return "ARRAY_ACCESS"
    if any(line_has_field_write_signal(line) for line in path_lines):
        return "PARSER_STATE"
    if any(line_has_read_like_signal(line) for line in path_lines):
        return "TAINT_FLOW"
    return "STATE_MUTATION"


def line_has_parameter_sensitive_sink(
    line: str,
    tainted_names: Dict[str, int],
    *,
    allow_format_without_read_signal: bool = False,
) -> bool:
    if not tainted_names or not line_references_tainted_name(line, tainted_names):
        return False

    assignment = split_assignment_expression(line)
    if assignment is not None:
        left, right = assignment
        pointer_target = extract_pointer_write_target_name(left)
        if pointer_target and pointer_target in tainted_names:
            return True

        array_target = extract_array_write_target(left)
        if array_target is not None:
            array_base, array_index = array_target
            if array_base in tainted_names:
                return True
            if any(re.search(rf"\b{re.escape(name)}\b", array_index) for name in tainted_names):
                return True

        field_base = extract_field_write_base(left)
        if field_base and field_base in tainted_names and line_has_read_like_signal(right):
            return True

    call_names = extract_call_names(line)
    if not call_names:
        return False

    primary_call_name = call_names[0]
    referenced_count = sum(1 for name in tainted_names if re.search(rf"\b{re.escape(name)}\b", line))

    if primary_call_name in EXEC_SINK_CALL_NAMES:
        return True
    if primary_call_name in COPY_SINK_CALL_NAMES:
        return referenced_count >= 2 or line_has_read_like_signal(line)
    if primary_call_name in FORMAT_SINK_CALL_NAMES:
        return allow_format_without_read_signal or referenced_count >= 2 or line_has_read_like_signal(line)

    return False


def is_structurally_safe_path(
    path: List[Dict[str, object]],
    source_path: Path | None,
    source_cache: Dict[object, List[str]],
    stripped_cache: Dict[object, List[str]],
    inline_source_text: str = "",
) -> bool:
    if (source_path is None and not inline_source_text) or not path:
        return False

    source_lines = load_source_lines(source_path, source_cache, inline_source_text)
    last_step = path[-1]
    line_number = int(last_step.get("line") or 0)
    if line_number <= 0 or line_number > len(source_lines):
        return False

    line_text = source_lines[line_number - 1]
    array_match = ARRAY_ACCESS_PATTERN.search(line_text)
    if array_match:
        return has_complete_bounds_guard(source_lines, line_number, array_match.group(2), 1)

    for match in CALL_NAME_PATTERN.finditer(line_text):
        call_name = match.group(1)
        if call_name in CALL_SKIP_NAMES:
            continue
        helper_spans = infer_method_span_from_source(
            source_path=source_path,
            target_method=call_name,
            source_cache=source_cache,
            stripped_cache=stripped_cache,
            inline_source_text=inline_source_text,
        )
        for helper_span in helper_spans:
            helper_parameters = extract_function_parameter_names(
                source_path=source_path,
                target_method=call_name,
                span=helper_span,
                source_cache=source_cache,
                stripped_cache=stripped_cache,
                inline_source_text=inline_source_text,
            )
            taint_var_name = helper_parameters[0] if helper_parameters else infer_local_taint_variable(source_lines, helper_span)
            guarded_sink_lines = infer_guarded_sink_lines(source_lines, helper_span, taint_var_name)
            unsafe_sink_lines = infer_unsafe_sink_lines(source_lines, helper_span, taint_var_name)
            if guarded_sink_lines and not unsafe_sink_lines:
                return True

    return False


def infer_parameter_paths(
    metadata_record: Dict[str, object],
    source_path: Path | None,
    staged_rel_path: str,
    source_cache: Dict[object, List[str]],
    stripped_cache: Dict[object, List[str]],
    inline_source_text: str = "",
) -> List[List[Dict[str, object]]]:
    target_method = str(metadata_record.get("target_method", "") or "")
    if (source_path is None and not inline_source_text) or not target_method or not staged_rel_path:
        return []

    target_spans = infer_method_span_from_source(
        source_path=source_path,
        target_method=target_method,
        source_cache=source_cache,
        stripped_cache=stripped_cache,
        inline_source_text=inline_source_text,
    )
    if not target_spans:
        return []

    source_lines = load_source_lines(source_path, source_cache, inline_source_text)
    stripped_lines = load_stripped_source_lines(source_path, source_cache, stripped_cache, inline_source_text)
    parameter_paths: List[List[Dict[str, object]]] = []

    for target_span in target_spans:
        start_line, end_line = target_span
        parameter_names = extract_function_parameter_names(
            source_path=source_path,
            target_method=target_method,
            span=target_span,
            source_cache=source_cache,
            stripped_cache=stripped_cache,
            inline_source_text=inline_source_text,
        )
        if not parameter_names:
            continue

        tainted_names: Dict[str, int] = {name: start_line for name in parameter_names}
        alias_lines: Dict[str, int] = {}
        read_signal_hits: List[int] = []
        candidate_sinks: List[Tuple[int, List[str], int | None]] = []

        for line_number in range(start_line, end_line + 1):
            stripped_line = stripped_lines[line_number - 1].strip()
            if not stripped_line:
                continue

            if line_has_read_like_signal(stripped_line):
                read_signal_hits.append(line_number)

            assignment_target = extract_simple_assignment_target(stripped_line)
            if assignment_target and assignment_target not in tainted_names:
                rhs = stripped_line.split("=", 1)[1]
                if any(re.search(rf"\b{re.escape(name)}\b", rhs) for name in tainted_names):
                    tainted_names[assignment_target] = line_number
                    alias_lines[assignment_target] = line_number

            recent_read_signal = next((hit for hit in reversed(read_signal_hits) if hit < line_number), None)
            if not line_has_parameter_sensitive_sink(
                stripped_line,
                tainted_names,
                allow_format_without_read_signal=recent_read_signal is not None,
            ):
                continue

            referenced = [
                name for name in tainted_names
                if re.search(rf"\b{re.escape(name)}\b", stripped_line)
            ]
            if referenced:
                candidate_sinks.append((line_number, referenced, recent_read_signal))

        for sink_line, referenced_names, recent_read_signal in candidate_sinks:
            ordered_lines: List[int] = []
            source_line = min(tainted_names[name] for name in referenced_names)
            ordered_lines.append(source_line)

            for alias_name, alias_line in sorted(alias_lines.items(), key=lambda item: item[1]):
                if alias_line <= source_line or alias_line >= sink_line:
                    continue
                if alias_name in referenced_names:
                    ordered_lines.append(alias_line)

            if recent_read_signal is not None and source_line < recent_read_signal < sink_line:
                ordered_lines.append(recent_read_signal)

            for guard_line in range(max(start_line, sink_line - 6), sink_line):
                guard_text = stripped_lines[guard_line - 1].strip()
                if not guard_text:
                    continue
                if not GUARD_LINE_PATTERN.search(guard_text):
                    continue
                if not any(re.search(rf"\b{re.escape(name)}\b", guard_text) for name in referenced_names):
                    continue
                ordered_lines.append(guard_line)
                break

            ordered_lines.append(sink_line)

            seen_lines = set()
            path_locations: List[Dict[str, object]] = []
            for line_number in ordered_lines:
                if line_number in seen_lines:
                    continue
                seen_lines.add(line_number)
                path_locations.append(
                    {
                        "uri": staged_rel_path,
                        "line": line_number,
                        "column": 1,
                        "message": "heuristic_parameter_flow",
                    }
                )

            if len(path_locations) >= 2:
                parameter_paths.append(path_locations)
                break

        if parameter_paths:
            return parameter_paths

    return parameter_paths


def infer_helper_paths(
    metadata_record: Dict[str, object],
    source_path: Path | None,
    staged_rel_path: str,
    source_cache: Dict[object, List[str]],
    stripped_cache: Dict[object, List[str]],
    inline_source_text: str = "",
) -> List[List[Dict[str, object]]]:
    target_method = str(metadata_record.get("target_method", "") or "")
    if (source_path is None and not inline_source_text) or not target_method or not staged_rel_path:
        return []

    target_spans = infer_method_span_from_source(
        source_path=source_path,
        target_method=target_method,
        source_cache=source_cache,
        stripped_cache=stripped_cache,
        inline_source_text=inline_source_text,
    )
    if not target_spans:
        return []

    source_lines = load_source_lines(source_path, source_cache, inline_source_text)
    helper_paths: List[List[Dict[str, object]]] = []

    for target_span in target_spans:
        start_line, end_line = target_span
        source_hits: List[int] = []
        conversion_hits: List[int] = []
        helper_calls: List[Tuple[int, str]] = []

        for line_number in range(start_line, end_line + 1):
            line = source_lines[line_number - 1]
            if any(pattern.search(line) for pattern in SOURCE_LINE_PATTERNS):
                source_hits.append(line_number)
            if any(pattern.search(line) for pattern in CONVERSION_LINE_PATTERNS):
                conversion_hits.append(line_number)
            for match in CALL_NAME_PATTERN.finditer(line):
                call_name = match.group(1)
                if call_name in CALL_SKIP_NAMES or call_name == target_method:
                    continue
                helper_calls.append((line_number, call_name))

        if not helper_calls:
            continue
        if not source_hits and not conversion_hits:
            continue

        for call_line, helper_name in helper_calls:
            helper_spans = infer_method_span_from_source(
                source_path=source_path,
                target_method=helper_name,
                source_cache=source_cache,
                stripped_cache=stripped_cache,
                inline_source_text=inline_source_text,
            )
            for helper_span in helper_spans:
                helper_parameters = extract_function_parameter_names(
                    source_path=source_path,
                    target_method=helper_name,
                    span=helper_span,
                    source_cache=source_cache,
                    stripped_cache=stripped_cache,
                    inline_source_text=inline_source_text,
                )
                taint_var_name = helper_parameters[0] if helper_parameters else infer_local_taint_variable(source_lines, helper_span)
                sink_lines = infer_unsafe_sink_lines(source_lines, helper_span, taint_var_name)
                if not sink_lines:
                    continue

                ordered_lines: List[int] = []
                ordered_lines.extend(source_hits[-2:])
                ordered_lines.extend(conversion_hits[-1:])
                ordered_lines.append(call_line)
                ordered_lines.append(helper_span[0])
                ordered_lines.append(sink_lines[0])

                seen_lines = set()
                path_locations: List[Dict[str, object]] = []
                for line_number in ordered_lines:
                    if line_number <= 0 or line_number in seen_lines:
                        continue
                    seen_lines.add(line_number)
                    path_locations.append(
                        {
                            "uri": staged_rel_path,
                            "line": line_number,
                            "column": 1,
                            "message": "heuristic_helper_flow",
                        }
                    )

                if len(path_locations) >= 3:
                    helper_paths.append(path_locations)
                    return helper_paths

    return helper_paths


def extract_thread_flow_locations(result: Dict[str, object], max_steps: int) -> List[List[Dict[str, object]]]:
    paths: List[List[Dict[str, object]]] = []
    for code_flow in result.get("codeFlows", []) or []:
        for thread_flow in code_flow.get("threadFlows", []) or []:
            locations = thread_flow.get("locations", []) or []
            flattened: List[Dict[str, object]] = []
            for item in locations[:max_steps]:
                location = item.get("location", {}) or {}
                physical = location.get("physicalLocation", {}) or {}
                artifact = physical.get("artifactLocation", {}) or {}
                region = physical.get("region", {}) or {}
                uri = str(artifact.get("uri", "") or "")
                if not uri:
                    continue
                flattened.append(
                    {
                        "uri": normalize_rel_path(uri),
                        "line": int(region.get("startLine")) if region.get("startLine") is not None else None,
                        "column": int(region.get("startColumn")) if region.get("startColumn") is not None else None,
                        "message": str((location.get("message", {}) or {}).get("text", "") or ""),
                    }
                )
            if flattened:
                paths.append(flattened)
    return paths


def index_paths_by_file(sarif_payload: Dict[str, object], max_steps: int) -> Dict[str, List[List[Dict[str, object]]]]:
    indexed: Dict[str, List[List[Dict[str, object]]]] = {}
    for run in sarif_payload.get("runs", []) or []:
        for result in run.get("results", []) or []:
            for path in extract_thread_flow_locations(result, max_steps):
                first_uri = path[0]["uri"]
                indexed.setdefault(first_uri, []).append(path)
    return indexed


def path_intersects_spans(path: List[Dict[str, object]], spans: List[Tuple[int, int]]) -> bool:
    for step in path:
        line = step.get("line")
        if line is None:
            continue
        for start_line, end_line in spans:
            if start_line <= line <= end_line:
                return True
    return False


def build_graph_from_paths(
    metadata_record: Dict[str, object],
    input_record: Dict[str, object],
    staged_root: Path,
    raw_source_root: Path | None,
    paths: List[List[Dict[str, object]]],
    path_families: List[str],
    max_paths: int,
    edge_type: str,
    family_edge_types: bool,
    source_cache: Dict[object, List[str]],
    stripped_cache: Dict[object, List[str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    node_index_by_key: Dict[Tuple[str, int | None, int | None], int] = {}

    for path_index, path in enumerate(paths[:max_paths]):
        local_node_ids: List[int] = []
        path_family = path_families[path_index] if path_index < len(path_families) else "TAINT_FLOW"
        path_edge_type = f"{edge_type}:{path_family}" if family_edge_types else edge_type
        for step_index, location in enumerate(path):
            key = (location["uri"], location["line"], location["column"])
            if key not in node_index_by_key:
                source_path = staged_root / Path(location["uri"])
                if not source_path.exists() and raw_source_root and metadata_record.get("source_file"):
                    source_path = raw_source_root / str(metadata_record.get("source_file"))
                line_text = get_line_text(
                    source_path,
                    location["line"],
                    source_cache,
                    str(input_record.get("func", "") or ""),
                )
                node_id = len(nodes)
                node_index_by_key[key] = node_id
                nodes.append(
                    {
                        "id": node_id,
                        "text": line_text or str(location.get("message", "") or ""),
                        "node_type": f"PATH_{path_family}" if family_edge_types else "PATH_NODE",
                        "line": location["line"],
                        "is_source": step_index == 0,
                        "is_sink": step_index == len(path) - 1,
                        "is_direct_sink": False,
                        "path_family": path_family,
                    }
                )
            node_id = node_index_by_key[key]
            if step_index == 0:
                nodes[node_id]["is_source"] = True
            if step_index == len(path) - 1:
                nodes[node_id]["is_sink"] = True
            local_node_ids.append(node_id)

        for src, dst in zip(local_node_ids, local_node_ids[1:]):
            edges.append({"src": src, "dst": dst, "edge_type": path_edge_type, "path_family": path_family})

    if nodes:
        return nodes, edges

    source_path = resolve_source_path(metadata_record, staged_root, raw_source_root)
    inline_source_text = str(input_record.get("func", "") or "")
    source_lines = load_source_lines(source_path, source_cache, inline_source_text)
    stripped_lines = load_stripped_source_lines(source_path, source_cache, stripped_cache, inline_source_text)
    target_method = str(metadata_record.get("target_method", "") or "")
    spans = infer_method_span_from_source(
        source_path=source_path,
        target_method=target_method,
        source_cache=source_cache,
        stripped_cache=stripped_cache,
        inline_source_text=inline_source_text,
    )

    if spans:
        start_line = min(start for start, _ in spans)
        end_line = max(end for _, end in spans)
    else:
        start_line = 1
        end_line = len(source_lines)

    selected_line_numbers: List[int] = []
    selected_line_set = set()
    line_roles: Dict[int, str] = {}

    def mark_line(line_number: int, role: str) -> None:
        if line_number < start_line or line_number > end_line:
            return
        if line_number not in selected_line_set:
            selected_line_set.add(line_number)
            selected_line_numbers.append(line_number)
        current = line_roles.get(line_number)
        priority = {
            "DIRECT_SINK_STMT": 6,
            "SINK_STMT": 5,
            "SOURCE_STMT": 4,
            "CONVERT_STMT": 3,
            "GUARD_STMT": 2,
            "CALL_STMT": 1,
            "STATEMENT": 0,
        }
        if current is None or priority[role] > priority[current]:
            line_roles[line_number] = role

    for line_number in range(start_line, end_line + 1):
        stripped_line = stripped_lines[line_number - 1].strip()
        raw_line = source_lines[line_number - 1].strip()
        if not stripped_line:
            continue

        if any(pattern.search(stripped_line) for pattern in SOURCE_LINE_PATTERNS):
            mark_line(line_number, "SOURCE_STMT")
        if any(pattern.search(stripped_line) for pattern in CONVERSION_LINE_PATTERNS):
            mark_line(line_number, "CONVERT_STMT")
        if any(pattern.search(stripped_line) for pattern in DIRECT_SINK_LINE_PATTERNS):
            mark_line(line_number, "DIRECT_SINK_STMT")
        if (
            any(pattern.search(stripped_line) for pattern in FALLBACK_SINK_LINE_PATTERNS)
            or ARRAY_ACCESS_PATTERN.search(stripped_line)
            or POINTER_WRITE_PATTERN.search(stripped_line)
        ):
            mark_line(line_number, "SINK_STMT")
        if GUARD_LINE_PATTERN.search(stripped_line) and COMPARISON_OPERATOR_PATTERN.search(stripped_line):
            mark_line(line_number, "GUARD_STMT")
        if any(
            call_name not in CALL_SKIP_NAMES
            for call_name in CALL_NAME_PATTERN.findall(stripped_line)
        ):
            mark_line(line_number, "CALL_STMT")
        elif raw_line and raw_line not in {"{", "}"}:
            if (
                any(keyword in stripped_line for keyword in ("=", "return", "[", "]"))
                or len(stripped_line) < 160
            ):
                mark_line(line_number, "STATEMENT")

    if selected_line_numbers:
        expanded_line_numbers = set(selected_line_numbers)
        for line_number in list(selected_line_numbers):
            for neighbor in (line_number - 1, line_number + 1):
                if neighbor < start_line or neighbor > end_line:
                    continue
                neighbor_line = source_lines[neighbor - 1].strip()
                if neighbor_line and neighbor_line not in {"{", "}"}:
                    expanded_line_numbers.add(neighbor)
                    line_roles.setdefault(neighbor, "STATEMENT")
        selected_line_numbers = sorted(expanded_line_numbers)[:FALLBACK_MAX_NODES]

    if selected_line_numbers:
        fallback_nodes: List[Dict[str, object]] = []
        fallback_edges: List[Dict[str, object]] = []
        node_id_by_line: Dict[int, int] = {}

        for line_number in selected_line_numbers:
            role = line_roles.get(line_number, "STATEMENT")
            text = source_lines[line_number - 1].strip() or stripped_lines[line_number - 1].strip()
            node_id = len(fallback_nodes)
            node_id_by_line[line_number] = node_id
            fallback_nodes.append(
                {
                    "id": node_id,
                    "text": text,
                    "node_type": role,
                    "line": line_number,
                    "is_source": role == "SOURCE_STMT",
                    "is_sink": role in {"SINK_STMT", "DIRECT_SINK_STMT"},
                    "is_direct_sink": role == "DIRECT_SINK_STMT",
                }
            )

        ordered_lines = sorted(selected_line_numbers)
        for src_line, dst_line in zip(ordered_lines, ordered_lines[1:]):
            fallback_edges.append(
                {
                    "src": node_id_by_line[src_line],
                    "dst": node_id_by_line[dst_line],
                    "edge_type": "FALLBACK_SEQ",
                }
            )

        taint_source_lines = [line for line in ordered_lines if line_roles.get(line) in {"SOURCE_STMT", "CONVERT_STMT"}]
        sink_like_lines = [line for line in ordered_lines if line_roles.get(line) in {"SINK_STMT", "DIRECT_SINK_STMT", "CALL_STMT"}]
        guard_lines = [line for line in ordered_lines if line_roles.get(line) == "GUARD_STMT"]

        for src_line in taint_source_lines:
            for dst_line in sink_like_lines:
                if src_line < dst_line and (dst_line - src_line) <= 20:
                    fallback_edges.append(
                        {
                            "src": node_id_by_line[src_line],
                            "dst": node_id_by_line[dst_line],
                            "edge_type": "FALLBACK_TAINT",
                        }
                    )

        for guard_line in guard_lines:
            for sink_line in sink_like_lines:
                if guard_line < sink_line and (sink_line - guard_line) <= 8:
                    fallback_edges.append(
                        {
                            "src": node_id_by_line[guard_line],
                            "dst": node_id_by_line[sink_line],
                            "edge_type": "FALLBACK_GUARD",
                        }
                    )
                    break

        if fallback_nodes:
            return fallback_nodes, fallback_edges

    fallback_text = str(input_record.get("func") or input_record.get("source_file") or "")
    return (
        [
            {
                "id": 0,
                "text": fallback_text,
                "node_type": "METHOD",
                "line": 1,
                "is_source": False,
                "is_sink": False,
                "is_direct_sink": False,
            }
        ],
        [],
    )


def main() -> None:
    args = parse_args()
    database_summary = read_json(args.database_summary.resolve())
    query_summary = read_json(args.query_summary.resolve())

    input_records = read_jsonl(Path(str(database_summary["input"])))
    input_by_id = {str(record["idx"]): record for record in input_records}
    metadata_records = read_jsonl(Path(str(database_summary["metadata_path"])))
    sarif_payload = read_json(Path(str(query_summary["sarif"])))

    staged_root = Path(str(database_summary["artifact_dir"])) / "staged_sources"
    raw_source_root = Path(str(database_summary["raw_source_root"])) if database_summary.get("raw_source_root") else None
    paths_by_file = index_paths_by_file(sarif_payload, max_steps=args.max_steps_per_path)
    function_spans = load_function_spans(args.function_spans.resolve() if args.function_spans else None)
    source_cache: Dict[object, List[str]] = {}
    stripped_cache: Dict[object, List[str]] = {}

    output_records: List[Dict[str, object]] = []
    graph_count_with_paths = 0
    graph_count_fallback = 0
    duplicate_mapping_count = 0
    source_span_hits = 0
    codeql_span_hits = 0
    heuristic_path_hits = 0
    safe_negative_path_hits = 0
    family_counter: Counter[str] = Counter()
    family_origin_counter: Counter[str] = Counter()

    for metadata_record in metadata_records:
        sample_id = str(metadata_record["idx"])
        input_record = input_by_id.get(sample_id, {})
        staged_rel_path = normalize_rel_path(str(metadata_record.get("staged_rel_path", "") or ""))
        matched_paths = list(paths_by_file.get(staged_rel_path, []))
        graph_origin = "fallback"
        path_safety_hint = "none"
        target_method = str(metadata_record.get("target_method", "") or "")
        staged_file_name = str(metadata_record.get("staged_file_name", "") or "")
        spans = function_spans.get((staged_file_name, target_method), [])
        spans = [span for span in spans if span[0] > 0 and span[1] >= span[0]]
        source_path = resolve_source_path(metadata_record, staged_root, raw_source_root)
        inline_source_text = str(input_record.get("func", "") or "")
        if not spans or all(start_line == end_line for start_line, end_line in spans):
            inferred_spans = infer_method_span_from_source(
                source_path=source_path,
                target_method=target_method,
                source_cache=source_cache,
                stripped_cache=stripped_cache,
                inline_source_text=inline_source_text,
            )
            if inferred_spans:
                spans = inferred_spans
                source_span_hits += 1
        elif spans:
            codeql_span_hits += 1
        if spans:
            filtered_paths = [path for path in matched_paths if path_intersects_spans(path, spans)]
            matched_paths = filtered_paths
        if matched_paths:
            safe_paths: List[List[Dict[str, object]]] = []
            unsafe_paths: List[List[Dict[str, object]]] = []
            for path in matched_paths:
                if is_structurally_safe_path(
                    path=path,
                    source_path=source_path,
                    source_cache=source_cache,
                    stripped_cache=stripped_cache,
                    inline_source_text=inline_source_text,
                ):
                    safe_paths.append(path)
                else:
                    unsafe_paths.append(path)

            target_value = int(metadata_record.get("target", input_record.get("target", 0)))
            if unsafe_paths:
                matched_paths = unsafe_paths
                path_safety_hint = "unsafe"
            elif target_value == 0 and safe_paths:
                matched_paths = safe_paths
                path_safety_hint = "safe"
                safe_negative_path_hits += 1
            else:
                matched_paths = []

        if not matched_paths:
            parameter_paths = infer_parameter_paths(
                metadata_record=metadata_record,
                source_path=source_path,
                staged_rel_path=staged_rel_path,
                source_cache=source_cache,
                stripped_cache=stripped_cache,
                inline_source_text=inline_source_text,
            )
            if parameter_paths:
                matched_paths = parameter_paths
                heuristic_path_hits += 1
                graph_origin = "heuristic_path"
                path_safety_hint = "unsafe"

        if not matched_paths:
            heuristic_paths = infer_helper_paths(
                metadata_record=metadata_record,
                source_path=source_path,
                staged_rel_path=staged_rel_path,
                source_cache=source_cache,
                stripped_cache=stripped_cache,
                inline_source_text=inline_source_text,
            )
            if heuristic_paths:
                matched_paths = heuristic_paths
                heuristic_path_hits += 1
                graph_origin = "heuristic_path"
                path_safety_hint = "unsafe"

        if len(matched_paths) > args.max_paths_per_sample:
            matched_paths = matched_paths[: args.max_paths_per_sample]

        if matched_paths:
            graph_count_with_paths += 1
            if graph_origin != "heuristic_path":
                graph_origin = "codeql_path"
        else:
            graph_count_fallback += 1

        path_families = [
            classify_path_family(
                path=path,
                source_path=source_path,
                source_cache=source_cache,
                inline_source_text=inline_source_text,
            )
            for path in matched_paths
        ]
        for path_family in set(path_families):
            family_counter[path_family] += 1
            family_origin_counter[f"{graph_origin}:{path_family}"] += 1

        nodes, edges = build_graph_from_paths(
            metadata_record=metadata_record,
            input_record=input_record,
            staged_root=staged_root,
            raw_source_root=raw_source_root,
            paths=matched_paths,
            path_families=path_families,
            max_paths=args.max_paths_per_sample,
            edge_type=args.edge_type,
            family_edge_types=args.family_edge_types,
            source_cache=source_cache,
            stripped_cache=stripped_cache,
        )

        output_records.append(
            {
                "idx": sample_id,
                "target": int(metadata_record.get("target", input_record.get("target", 0))),
                "dataset": str(metadata_record.get("dataset", input_record.get("dataset", ""))),
                "project": str(metadata_record.get("project", input_record.get("project", ""))),
                "cwe_id": str(metadata_record.get("cwe_id", input_record.get("cwe_id", ""))),
                "source_file": str(metadata_record.get("source_file", input_record.get("source_file", ""))),
                "code": str(input_record.get("func", "") or ""),
                "func": str(input_record.get("func", "") or ""),
                "nodes": nodes,
                "edges": edges,
                "graph_origin": graph_origin,
                "path_safety_hint": path_safety_hint,
                "path_count": len(matched_paths),
                "path_families": path_families,
                "staged_rel_path": staged_rel_path,
            }
        )

    output_count = write_jsonl(args.output.resolve(), output_records)
    summary_payload = {
        "database_summary": str(args.database_summary.resolve()),
        "query_summary": str(args.query_summary.resolve()),
        "output": str(args.output.resolve()),
        "records": output_count,
        "graphs_with_paths": graph_count_with_paths,
        "graphs_with_fallback": graph_count_fallback,
        "duplicate_mapping_count": duplicate_mapping_count,
        "codeql_span_hits": codeql_span_hits,
        "source_span_hits": source_span_hits,
        "heuristic_path_hits": heuristic_path_hits,
        "safe_negative_path_hits": safe_negative_path_hits,
        "edge_type": args.edge_type,
        "family_edge_types": bool(args.family_edge_types),
        "path_family_counts": dict(family_counter.most_common()),
        "path_family_origin_counts": dict(family_origin_counter.most_common()),
        "max_paths_per_sample": args.max_paths_per_sample,
        "max_steps_per_path": args.max_steps_per_path,
    }
    summary_path = args.summary_path.resolve() if args.summary_path else args.output.resolve().with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"records={output_count}")
    print(f"graphs_with_paths={graph_count_with_paths}")
    print(f"graphs_with_fallback={graph_count_fallback}")
    print(f"output={args.output.resolve()}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
