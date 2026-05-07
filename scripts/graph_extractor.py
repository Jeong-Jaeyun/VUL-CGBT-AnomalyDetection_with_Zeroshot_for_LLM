import re
from typing import Dict, List, Sequence, Tuple


CONTROL_FLOW_PREFIXES = ("if", "else if", "else", "for", "while", "do", "switch")
TERMINATOR_KINDS = {"return", "break", "continue", "goto"}
TYPE_KEYWORDS = {
    "_bool",
    "auto",
    "bool",
    "char",
    "class",
    "double",
    "enum",
    "float",
    "int",
    "long",
    "short",
    "signed",
    "size_t",
    "ssize_t",
    "struct",
    "typedef",
    "union",
    "unsigned",
    "void",
}
QUALIFIER_KEYWORDS = {
    "const",
    "constexpr",
    "extern",
    "inline",
    "mutable",
    "register",
    "restrict",
    "static",
    "thread_local",
    "volatile",
}
RESERVED_KEYWORDS = TYPE_KEYWORDS | QUALIFIER_KEYWORDS | {
    "alignas",
    "alignof",
    "asm",
    "break",
    "case",
    "catch",
    "continue",
    "default",
    "delete",
    "do",
    "else",
    "false",
    "for",
    "friend",
    "goto",
    "if",
    "namespace",
    "new",
    "nullptr",
    "operator",
    "private",
    "protected",
    "public",
    "return",
    "sizeof",
    "switch",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typename",
    "using",
    "while",
}
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT_RE = re.compile(r"//.*?$", re.M)
STRING_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\])*"')
CHAR_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])*'")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
COMPOUND_ASSIGNMENT_RE = re.compile(r"<<=|>>=|\+=|-=|\*=|/=|%=|&=|\|=|\^=")
SIMPLE_ASSIGNMENT_RE = re.compile(r"(?<![=!<>+\-*/%&|^])=(?!=)")
INCREMENT_RE = re.compile(r"(?:\+\+|--)\s*([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*(?:\+\+|--)")
LABEL_RE = re.compile(r"^[A-Za-z_]\w*\s*:$")
GOTO_RE = re.compile(r"\bgoto\s+([A-Za-z_]\w*)\b")


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_source(source: str) -> str:
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    source = STRING_LITERAL_RE.sub('"STR"', source)
    source = CHAR_LITERAL_RE.sub("'CHR'", source)
    source = BLOCK_COMMENT_RE.sub(" ", source)
    source = LINE_COMMENT_RE.sub(" ", source)
    return source


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def split_top_level(text: str, delimiter: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    for char in text:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)

        if char == delimiter and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)

    if current:
        parts.append("".join(current))
    return parts


def classify_cfg_node(text: str) -> str:
    stripped = compact_whitespace(text)
    lowered = stripped.lower()
    if lowered.startswith("else if"):
        return "else_if"
    if lowered.startswith("if"):
        return "if"
    if lowered.startswith("else"):
        return "else"
    if lowered.startswith("for") or lowered.startswith("while") or lowered.startswith("do"):
        return "loop"
    if lowered.startswith("switch"):
        return "switch"
    if lowered.startswith("case ") or lowered.startswith("default"):
        return "case"
    if lowered.startswith("return"):
        return "return"
    if lowered.startswith("break"):
        return "break"
    if lowered.startswith("continue"):
        return "continue"
    if lowered.startswith("goto"):
        return "goto"
    if LABEL_RE.match(stripped):
        return "label"
    if stripped.endswith("{") and "(" in stripped and not lowered.startswith(CONTROL_FLOW_PREFIXES):
        return "signature"
    return "stmt"


def extract_cfg_node_metadata(source: str, max_nodes: int | None = None) -> List[Dict[str, object]]:
    sanitized = sanitize_source(source)
    nodes: List[Dict[str, object]] = []
    current: List[str] = []
    brace_depth = 0
    paren_depth = 0

    def emit(raw_text: str, depth: int) -> None:
        text = compact_whitespace(raw_text)
        if not text:
            return
        nodes.append(
            {
                "text": text,
                "depth": depth,
                "kind": classify_cfg_node(text),
            }
        )

    for char in sanitized:
        if char == "(":
            current.append(char)
            paren_depth += 1
            continue
        if char == ")":
            current.append(char)
            paren_depth = max(0, paren_depth - 1)
            continue

        if char == "{" and paren_depth == 0:
            raw_text = "".join(current).strip()
            if raw_text:
                emit(f"{raw_text} {{", brace_depth)
            current = []
            brace_depth += 1
            continue

        if char == "}" and paren_depth == 0:
            raw_text = "".join(current).strip()
            if raw_text:
                emit(raw_text, brace_depth)
            current = []
            brace_depth = max(0, brace_depth - 1)
            continue

        if char == ";" and paren_depth == 0:
            current.append(char)
            emit("".join(current), brace_depth)
            current = []
            continue

        if char == ":" and paren_depth == 0:
            current.append(char)
            candidate = compact_whitespace("".join(current))
            if candidate.startswith("case ") or candidate.startswith("default") or LABEL_RE.match(candidate):
                emit(candidate, brace_depth)
                current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        emit(tail, brace_depth)

    if max_nodes is not None and max_nodes > 0:
        return nodes[:max_nodes]
    return nodes


def find_block_span(nodes: Sequence[Dict[str, object]], header_index: int) -> Tuple[int | None, int | None]:
    next_index = header_index + 1
    if next_index >= len(nodes):
        return None, None

    header_depth = int(nodes[header_index]["depth"])
    next_depth = int(nodes[next_index]["depth"])

    if next_depth > header_depth:
        end_index = next_index
        while end_index < len(nodes) and int(nodes[end_index]["depth"]) > header_depth:
            end_index += 1
        after_index = end_index if end_index < len(nodes) else None
        last_body_index = end_index - 1
        return after_index, last_body_index

    after_index = next_index + 1 if next_index + 1 < len(nodes) else None
    return after_index, next_index


def build_cfg_graph(source: str, max_nodes: int = 128, max_edges: int = 512) -> Tuple[List[str], List[List[int]]]:
    nodes = extract_cfg_node_metadata(source, max_nodes=max_nodes)
    if not nodes:
        return [], []

    control_ranges: Dict[int, Tuple[int | None, int | None]] = {}
    label_indices: Dict[str, int] = {}
    for index, node in enumerate(nodes):
        if node["kind"] in {"if", "else_if", "else", "loop", "switch"}:
            control_ranges[index] = find_block_span(nodes, index)
        if node["kind"] == "label":
            label_name = str(node["text"]).rstrip(":").strip()
            if label_name:
                label_indices[label_name] = index

    edges: List[List[int]] = []
    seen_edges = set()

    def add_edge(source_index: int | None, target_index: int | None) -> None:
        if source_index is None or target_index is None:
            return
        if source_index == target_index:
            return
        if source_index < 0 or target_index < 0:
            return
        if source_index >= len(nodes) or target_index >= len(nodes):
            return
        edge = (source_index, target_index)
        if edge in seen_edges:
            return
        if len(edges) >= max_edges:
            return
        seen_edges.add(edge)
        edges.append([source_index, target_index])

    for index, node in enumerate(nodes[:-1]):
        if node["kind"] not in TERMINATOR_KINDS:
            add_edge(index, index + 1)

    def find_enclosing_header(node_index: int, candidate_kinds: Sequence[str]) -> int | None:
        for header_index in range(node_index - 1, -1, -1):
            if nodes[header_index]["kind"] not in candidate_kinds:
                continue
            after_index, last_body_index = control_ranges.get(header_index, (None, None))
            if last_body_index is None:
                continue
            if header_index < node_index <= last_body_index:
                return header_index
        return None

    for index, node in enumerate(nodes):
        kind = str(node["kind"])
        after_index, last_body_index = control_ranges.get(index, (None, None))

        if kind in {"if", "else_if"}:
            add_edge(index, after_index)
        elif kind == "loop":
            add_edge(index, after_index)
            add_edge(last_body_index, index)
        elif kind == "switch":
            add_edge(index, after_index)
            if last_body_index is not None:
                for case_index in range(index + 1, last_body_index + 1):
                    if nodes[case_index]["kind"] == "case":
                        add_edge(index, case_index)
        elif kind == "continue":
            add_edge(index, find_enclosing_header(index, ("loop",)))
        elif kind == "break":
            header_index = find_enclosing_header(index, ("loop", "switch"))
            if header_index is not None:
                add_edge(index, control_ranges.get(header_index, (None, None))[0])
        elif kind == "goto":
            match = GOTO_RE.search(str(node["text"]))
            if match:
                add_edge(index, label_indices.get(match.group(1)))

    return [str(node["text"]) for node in nodes], edges


def extract_call_names(text: str) -> set[str]:
    call_names = set()
    for match in CALL_RE.finditer(text):
        name = match.group(1)
        if name not in RESERVED_KEYWORDS:
            call_names.add(name)
    return call_names


def extract_identifiers(text: str, *, exclude_calls: bool = True) -> List[str]:
    call_names = extract_call_names(text) if exclude_calls else set()
    identifiers: List[str] = []
    seen = set()
    for match in IDENTIFIER_RE.finditer(text):
        name = match.group()
        if name in RESERVED_KEYWORDS:
            continue
        start = match.start()
        if start > 0 and text[start - 1] == ".":
            continue
        if start > 1 and text[start - 2 : start] == "->":
            continue
        if exclude_calls and name in call_names:
            continue
        if name not in seen:
            identifiers.append(name)
            seen.add(name)
    return identifiers


def extract_parameter_names(signature_text: str) -> List[str]:
    open_paren = signature_text.find("(")
    close_paren = signature_text.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren <= open_paren:
        return []

    parameters = signature_text[open_paren + 1 : close_paren]
    names: List[str] = []
    for parameter in split_top_level(parameters, ","):
        parameter = parameter.strip()
        if not parameter or parameter == "void" or "..." in parameter:
            continue
        candidate = parameter.split("=", 1)[0]
        candidate = candidate.replace("*", " ").replace("&", " ")
        ids = extract_identifiers(candidate, exclude_calls=False)
        filtered = [name for name in ids if name not in TYPE_KEYWORDS and name not in QUALIFIER_KEYWORDS]
        if filtered:
            names.append(filtered[-1])
    return dedupe_preserve_order(names)


def looks_like_declaration(statement: str) -> bool:
    stripped = statement.strip().rstrip(";")
    lowered = stripped.lower()
    if lowered.startswith(("if", "else", "for", "while", "switch", "return", "goto", "break", "continue")):
        return False
    if lowered.startswith(("struct ", "union ", "enum ", "class ")):
        return True

    leading_tokens = IDENTIFIER_RE.findall(stripped)
    if not leading_tokens:
        return False

    base_token = None
    for token in leading_tokens:
        token_lower = token.lower()
        if token_lower in QUALIFIER_KEYWORDS:
            continue
        base_token = token
        break

    if base_token is None:
        return False

    base_lower = base_token.lower()
    return (
        base_lower in TYPE_KEYWORDS
        or base_token.endswith("_t")
        or base_token[:1].isupper()
    )


def extract_declared_names(statement: str) -> List[str]:
    if not looks_like_declaration(statement):
        return []

    body = statement.strip().rstrip(";")
    names: List[str] = []
    for segment in split_top_level(body, ","):
        segment = segment.strip()
        if not segment:
            continue
        left = segment.split("=", 1)[0].strip()
        if not left:
            continue
        if "->" in left or "." in left:
            continue
        if "[" in left:
            left = left.split("[", 1)[0]
        left = left.replace("*", " ").replace("&", " ")
        ids = extract_identifiers(left, exclude_calls=False)
        filtered = [name for name in ids if name not in TYPE_KEYWORDS and name not in QUALIFIER_KEYWORDS]
        if filtered:
            names.append(filtered[-1])
    return dedupe_preserve_order(names)


def find_assignment_operator(statement: str) -> Tuple[int | None, str | None]:
    matches: List[Tuple[int, str]] = []
    for match in COMPOUND_ASSIGNMENT_RE.finditer(statement):
        matches.append((match.start(), match.group()))
    for match in SIMPLE_ASSIGNMENT_RE.finditer(statement):
        matches.append((match.start(), match.group()))
    if not matches:
        return None, None
    return min(matches, key=lambda item: item[0])


def extract_assignment_targets(lhs: str) -> List[str]:
    targets: List[str] = []
    for segment in split_top_level(lhs, ","):
        segment = segment.strip()
        if not segment:
            continue
        if segment.startswith("*") or segment.startswith("(*"):
            continue
        if "->" in segment or "." in segment:
            continue
        if "[" in segment:
            segment = segment.split("[", 1)[0]
        segment = segment.replace("*", " ").replace("&", " ")
        ids = extract_identifiers(segment, exclude_calls=False)
        filtered = [name for name in ids if name not in TYPE_KEYWORDS and name not in QUALIFIER_KEYWORDS]
        if filtered:
            targets.append(filtered[-1])
    return dedupe_preserve_order(targets)


def extract_increment_targets(statement: str) -> List[str]:
    targets: List[str] = []
    for match in INCREMENT_RE.finditer(statement):
        target = match.group(1) or match.group(2)
        if target:
            targets.append(target)
    return dedupe_preserve_order(targets)


def analyze_statement_for_data_flow(statement: str, kind: str, node_index: int) -> Tuple[List[str], List[str]]:
    text = compact_whitespace(statement.rstrip("{"))
    if not text:
        return [], []

    if node_index == 0 and kind == "signature":
        return extract_parameter_names(text), []

    if kind in {"goto", "label"}:
        return [], []

    definitions: List[str] = []
    uses: List[str] = []
    declared_names = extract_declared_names(text)
    assignment_position, assignment_operator = find_assignment_operator(text)

    if assignment_position is not None and assignment_operator is not None:
        lhs = text[:assignment_position]
        rhs = text[assignment_position + len(assignment_operator) :]
        assignment_targets = extract_assignment_targets(lhs)
        definitions.extend(assignment_targets)
        uses.extend(extract_identifiers(rhs))

        lhs_uses = [name for name in extract_identifiers(lhs) if name not in assignment_targets]
        uses.extend(lhs_uses)
        if assignment_operator != "=":
            uses.extend(assignment_targets)
    else:
        if kind in {"if", "else_if", "loop", "switch", "case", "return", "stmt", "else", "signature"}:
            uses.extend(extract_identifiers(text))

    definitions.extend(declared_names)

    increment_targets = extract_increment_targets(text)
    definitions.extend(increment_targets)
    uses.extend(increment_targets)

    if assignment_position is None and declared_names:
        uses = [name for name in uses if name not in declared_names]

    return dedupe_preserve_order(definitions), dedupe_preserve_order(uses)


def build_dfg_graph(source: str, max_nodes: int = 128, max_edges: int = 512) -> Tuple[List[str], List[List[int]]]:
    cfg_nodes = extract_cfg_node_metadata(source)
    if not cfg_nodes:
        return [], []

    dfg_nodes: List[str] = []
    dfg_edges: List[List[int]] = []
    seen_edges = set()
    last_definition: Dict[str, int] = {}

    def add_edge(source_index: int, target_index: int) -> None:
        edge = (source_index, target_index)
        if source_index == target_index:
            return
        if edge in seen_edges:
            return
        if len(dfg_edges) >= max_edges:
            return
        seen_edges.add(edge)
        dfg_edges.append([source_index, target_index])

    for node_index, node in enumerate(cfg_nodes):
        definitions, uses = analyze_statement_for_data_flow(str(node["text"]), str(node["kind"]), node_index)
        if not definitions and not uses:
            continue
        if len(dfg_nodes) >= max_nodes:
            break

        dfg_index = len(dfg_nodes)
        dfg_nodes.append(str(node["text"]))

        for variable_name in uses:
            previous_definition = last_definition.get(variable_name)
            if previous_definition is not None:
                add_edge(previous_definition, dfg_index)

        for variable_name in definitions:
            previous_definition = last_definition.get(variable_name)
            if previous_definition is not None:
                add_edge(previous_definition, dfg_index)
            last_definition[variable_name] = dfg_index

    return dfg_nodes, dfg_edges


def build_graph_record(
    record: Dict[str, object],
    *,
    cfg_max_nodes: int = 128,
    cfg_max_edges: int = 512,
    dfg_max_nodes: int = 128,
    dfg_max_edges: int = 512,
) -> Dict[str, object]:
    graph_record = dict(record)
    source_code = str(record.get("func", "") or "")
    cfg_nodes, cfg_edges = build_cfg_graph(
        source_code,
        max_nodes=cfg_max_nodes,
        max_edges=cfg_max_edges,
    )
    dfg_nodes, dfg_edges = build_dfg_graph(
        source_code,
        max_nodes=dfg_max_nodes,
        max_edges=dfg_max_edges,
    )
    graph_record.update(
        {
            "cfg_nodes": cfg_nodes,
            "cfg_edges": cfg_edges,
            "dfg_nodes": dfg_nodes,
            "dfg_edges": dfg_edges,
        }
    )
    return graph_record
