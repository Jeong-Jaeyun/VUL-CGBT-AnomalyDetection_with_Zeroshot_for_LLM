# CodeQL Windows Migration

## Goal

Replace the Joern/WSL extraction backbone with a Windows-native CodeQL pipeline while preserving the current research-level model quality:

- `GraphCodeBERT` offline node embeddings stay.
- `RelationAwareVulGAT` stays.
- `metrics.py` stays.
- `SARD-only training/selection, BigVul, Devign, and Reveal external test-only` protocol stays.
- Only the graph extraction and orchestration layer changes.

## Design Decision

The migration target is not "remove graph analysis". The target is:

`Joern CPG slicing on WSL` -> `CodeQL data-flow path extraction on Windows`

This keeps the research claim structural and path-based, but removes the current WSL/Joern bottleneck.

## Why Switch

Current Joern path:

- depends on WSL/Linux tooling
- is hard to debug from the current environment
- has shown `joern-slice` query explosion on `debug 100`
- blocks reproducible iteration speed

CodeQL path is preferable here because:

- Windows CLI execution is practical
- source/sink rules can be expressed as semantic queries instead of only regex filters
- path queries are easier to justify in a paper
- local debugging is easier in the current environment

## Keep vs Replace

### Keep

- `scripts/build_sard_subset.py`
- `scripts/build_bigvul_subset.py`
- `scripts/build_graph_embeddings.py`
- `code/gat_conv.py`
- `code/gat_model.py`
- `code/run_with_gat.py`
- `metrics.py`
- `configs/source_sink_rules_v1.json`

### Replace

- `scripts/joern_extract.py`
- `scripts/run_joern_extract_wsl.sh`
- `scripts/joern_ensure_ossdataflow.sc`
- `scripts/joern_probe_calls.sc`
- `scripts/run_build_graph_embeddings_wsl.sh`
- `scripts/run_train_gat_wsl.sh`

## Target Pipeline

## Current Experimental Protocol

The current main protocol is no longer the original `SARD-only` setup.

The main protocol is now:

- Main source and target domains: `Reveal generic` and `BigVul generic`
- External hard-case test only: `Devign`
- Main reported directions:
  - `Reveal -> BigVul`
  - `BigVul -> Reveal`
- Selection metric: `balanced_accuracy`
- Default graph variant: `targeted_v4`

Interpretation:

- `Reveal` and `BigVul` are the main transferable domains.
- `Devign` is retained as a harder external stress test.
- `Devign`-source training exists only for ablation and debugging.

The legacy `SARD-only` protocol is still useful as an early baseline or smoke path, but it is not the main research protocol anymore.

### Stage 1. Dataset Subsets

Current main usage:

- Build `Reveal generic` train/val/test.
- Build `BigVul generic` train/val/test.
- Use `Devign` test as external hard-case evaluation data.
- Keep `SARD` only for legacy baseline or debugging paths.

### Stage 2. CodeQL Database Creation

New Windows-native step.

Input:

- JSONL subset records
- raw source root

Output:

- per-split CodeQL database

Recommended new script:

- `scripts/build_codeql_database.py`
- wrapper: `scripts/run_codeql_extract_windows.ps1`

Responsibilities:

- stage source files under a Windows artifact directory
- optionally stage raw source files instead of only `func`
- create a CodeQL database from staged C/C++ files
- use a C/C++ build-capable CodeQL mode such as `autobuild` or `manual`

### Stage 3. CodeQL Path Extraction

New backbone.

Recommended new files:

- `qlpacks/vulcgbt-cpp/qlpack.yml`
- `qlpacks/vulcgbt-cpp/queries/SourceSinkConfig.qll`
- `qlpacks/vulcgbt-cpp/queries/CodeQLSourceSinkPaths.ql`
- `scripts/codeql_extract_paths.py`

Responsibilities:

- define source predicates
- define sink predicates
- run path query
- export results in JSON/CSV/SARIF-compatible form

### Stage 4. Graph JSON Build

This is the Joern replacement output stage.

Recommended new file:

- `scripts/build_codeql_graphs.py`

Responsibilities:

- convert CodeQL path results into graph JSONL
- produce the same graph schema expected by `build_graph_embeddings.py` and `run_with_gat.py`
- attach:
  - `nodes`
  - `edges`
  - `source_tags`
  - `sink_tags`
  - `slice_strategy: codeql_dataflow_path`

### Stage 5. GraphCodeBERT Embedding Cache

Unchanged interface.

Existing script stays:

- `scripts/build_graph_embeddings.py`

### Stage 6. GAT Training and Evaluation

Model code stays.

Only Windows wrapper changes.

Recommended new wrapper:

- `scripts/run_train_gat_windows.ps1`

## Research Positioning

The paper-level statement becomes:

> We use CodeQL-based source-sink path extraction as a deterministic static-analysis front-end, and train a relation-aware, structural-role-aware GAT over the extracted vulnerability-relevant path graph.

This is cleaner than the current Joern regex-heavy setup.

## CodeQL Graph Schema

The downstream schema should remain compatible with the current embedding and GAT code.

```json
{
  "idx": "...",
  "dataset": "sard",
  "cwe_id": "CWE-119",
  "target": 1,
  "code": "...",
  "nodes": [
    {
      "id": 0,
      "text": "fgets(inputBuffer, CHAR_ARRAY_SIZE, stdin)",
      "node_type": "CALL",
      "line": 42,
      "file": "CWE122_..._21.cpp",
      "is_source": true,
      "is_sink": false,
      "is_direct_sink": false
    }
  ],
  "edges": [
    {
      "src": 0,
      "dst": 1,
      "edge_type": "DATA_FLOW"
    }
  ],
  "slice_strategy": "codeql_dataflow_path",
  "source_tags": ["fgets"],
  "sink_tags": ["array_write"]
}
```

## Source/Sink Strategy Under CodeQL

### Sources

Preserve the current taxonomy:

- `argv`
- `getenv`
- `fgets`
- `recv`
- `recvfrom`
- `read`
- `fread`
- `scanf`
- `fscanf`
- `sscanf`

### Sinks

Split by semantics, not by regex only.

#### CWE-78

- `system`
- `popen`
- `execve`
- `execvp`

#### CWE-119

- library sinks:
  - `memcpy`
  - `memmove`
  - `strcpy`
  - `strcat`
  - `sprintf`
  - `vsprintf`
- structural sinks:
  - array write
  - pointer write
  - unsafe indexed memory access

#### Direct Sink

- `gets`

## Critical Architectural Change

Under CodeQL, do not rely on a single regex-heavy sink filter to drive the whole extraction.

Instead:

1. Define semantic source predicates.
2. Define semantic sink predicates.
3. Ask CodeQL for source-to-sink paths.
4. Convert path outputs into graph JSON.

This is the main reason the CodeQL path is likely to be more stable than the Joern path.

## Windows Artifact Layout

Recommended root:

- repo: `C:\Users\UCS\Vul-CGBT-main`
- artifacts: `C:\Users\UCS\artifacts\vul-cgbt`
- models: `C:\Users\UCS\models\graphcodebert-base`

Recommended artifact directories:

- `C:\Users\UCS\artifacts\vul-cgbt\codeql-db`
- `C:\Users\UCS\artifacts\vul-cgbt\codeql-results`
- `C:\Users\UCS\artifacts\vul-cgbt\graphs`
- `C:\Users\UCS\artifacts\vul-cgbt\embeddings`
- `C:\Users\UCS\artifacts\vul-cgbt\checkpoints`
- `C:\Users\UCS\artifacts\vul-cgbt\logs`

## New Script Map

### Extraction

Replace:

- `scripts/joern_extract.py`
- `scripts/run_joern_extract_wsl.sh`

With:

- `scripts/build_codeql_database.py`
- `scripts/codeql_extract_paths.py`
- `scripts/build_codeql_graphs.py`
- `scripts/run_codeql_extract_windows.ps1`

### Embeddings

Replace wrapper only:

- `scripts/run_build_graph_embeddings_wsl.sh`

With:

- `scripts/run_build_graph_embeddings_windows.ps1`

### Training

Replace wrapper only:

- `scripts/run_train_gat_wsl.sh`

With:

- `scripts/run_train_gat_windows.ps1`

## Migration Constraints

### Must Keep

- output schema compatibility with `run_with_gat.py`
- deterministic dataset splits
- local artifact reproducibility
- source/sink tags in graph JSON

### Can Change

- static-analysis backend
- extraction wrappers
- artifact directory layout
- intermediate result format

## Implementation Order

1. add Windows path convention doc
2. add CodeQL migration spec
3. add PowerShell wrappers
4. add CodeQL database builder
5. add CodeQL path query pack
6. add CodeQL path-to-graph converter
7. switch embedding/training wrappers to Windows
8. validate on `debug 100`
9. run full preprocessing

## Validation Checklist

Before full preprocessing, require all of the following on `debug 100`:

- database creation succeeds
- CodeQL path query returns non-zero results
- graph JSONL is generated
- fallback ratio is near zero
- `build_graph_embeddings.py --prepare-only` succeeds
- `run_with_gat.py` can load the graph JSON and embedding cache

## Immediate Next Step

The next implementation step should be:

`run_codeql_extract_windows.ps1` + `build_codeql_database.py`

That is the smallest Windows-native replacement that removes the current WSL/Joern dependency from the critical path.
