# Windows Path Convention

## Goal

Standardize a Windows-native artifact layout for the CodeQL migration path.

## Repository

- repo root: `C:\Users\UCS\Vul-CGBT-main`

## Model Cache

- GraphCodeBERT checkpoint: `C:\Users\UCS\models\graphcodebert-base`

## Artifact Root

- artifact root: `C:\Users\UCS\artifacts\vul-cgbt`

## Artifact Directories

- CodeQL databases: `C:\Users\UCS\artifacts\vul-cgbt\codeql-db`
- CodeQL query results: `C:\Users\UCS\artifacts\vul-cgbt\codeql-results`
- graph JSONL: `C:\Users\UCS\artifacts\vul-cgbt\graphs`
- embedding cache: `C:\Users\UCS\artifacts\vul-cgbt\embeddings`
- checkpoints: `C:\Users\UCS\artifacts\vul-cgbt\checkpoints`
- logs: `C:\Users\UCS\artifacts\vul-cgbt\logs`

## Recommended PowerShell Environment

```powershell
$env:VULCGBT_REPO = "C:\Users\UCS\Vul-CGBT-main"
$env:VULCGBT_MODELS = "C:\Users\UCS\models"
$env:VULCGBT_ARTIFACTS = "C:\Users\UCS\artifacts\vul-cgbt"
$env:VULCGBT_CODEQL = "C:\Users\UCS\AppData\Roaming\Code\User\globalStorage\github.vscode-codeql\distribution1\codeql\codeql.exe"
$env:VULCGBT_CODEQL_DB = "C:\Users\UCS\artifacts\vul-cgbt\codeql-db"
$env:VULCGBT_CODEQL_RESULTS = "C:\Users\UCS\artifacts\vul-cgbt\codeql-results"
$env:VULCGBT_GRAPHS = "C:\Users\UCS\artifacts\vul-cgbt\graphs"
$env:VULCGBT_EMBEDDINGS = "C:\Users\UCS\artifacts\vul-cgbt\embeddings"
$env:VULCGBT_CHECKPOINTS = "C:\Users\UCS\artifacts\vul-cgbt\checkpoints"
$env:GRAPHCODEBERT_DIR = "C:\Users\UCS\models\graphcodebert-base"
```

## Initial Directory Setup

```powershell
New-Item -ItemType Directory -Force -Path "C:\Users\UCS\models" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\Users\UCS\artifacts\vul-cgbt\codeql-db" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\Users\UCS\artifacts\vul-cgbt\codeql-results" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\Users\UCS\artifacts\vul-cgbt\graphs" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\Users\UCS\artifacts\vul-cgbt\embeddings" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\Users\UCS\artifacts\vul-cgbt\checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\Users\UCS\artifacts\vul-cgbt\logs" | Out-Null
```

## Script Policy

- PowerShell wrappers should resolve relative input paths against `$env:VULCGBT_REPO`.
- Heavy outputs must go under `$env:VULCGBT_ARTIFACTS`.
- Training outputs must go under `$env:VULCGBT_CHECKPOINTS\<experiment_name>`.
- No new WSL-specific path assumptions should be introduced in the CodeQL path.

## Wrapper Targets

Planned wrappers:

- `scripts/run_codeql_extract_windows.ps1`
- `scripts/run_build_graph_embeddings_windows.ps1`
- `scripts/run_train_gat_windows.ps1`
- `scripts/run_main_protocol_windows.ps1`
