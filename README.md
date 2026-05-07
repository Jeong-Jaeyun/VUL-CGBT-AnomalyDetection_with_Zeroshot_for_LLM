# Vul-CGBT

Windows-native `CodeQL -> GraphCodeBERT -> GAT` pipeline for vulnerability detection research.

## GitHub Upload Notes

This repository is intended to track source code, configs, docs, and handwritten CodeQL queries.

The following local assets should stay out of version control:

- large datasets and derived JSONL files under `dataset/`
- generated CodeQL queries under `qlpacks/vulcgbt-cpp/queries/_generated/`
- local Python cache directories such as `__pycache__/`
- the optional `apex/` checkout used only for `--fp16` training

The optional `apex` dependency is not required for normal runs. If you need mixed-precision training, install NVIDIA Apex separately instead of committing a nested third-party repository.

For a first GitHub upload from this folder:

```powershell
git init -b main
git add .
git status
```

Large local data can remain on disk; the repository ignore rules are set up so those files stay untracked.

## Current Main Protocol

The main protocol is now fixed as:

- Main source and target domains: `Reveal generic` and `BigVul generic`
- External hard-case test only: `Devign`
- Default graph family: `targeted_v4`
- Default model selection and threshold selection metric: `balanced_accuracy`

This means:

- `Reveal -> BigVul` is a main transfer experiment.
- `BigVul -> Reveal` is a main transfer experiment.
- `Devign` is evaluated only as an external hard-case domain.
- `Devign` source training remains available only for ablation, not for the main claim.

## Environment

Typical Windows setup:

```powershell
conda activate gcn
cd C:\Users\UCS\Vul-CGBT-main
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Optional:

- install NVIDIA Apex separately only if you plan to use `--fp16`

Expected CUDA example:

- `2.10.0+cu126`
- `True`

## Artifact Layout

Default artifact root:

- `C:\Users\UCS\artifacts\vul-cgbt`

Important subdirectories:

- CodeQL databases: `C:\Users\UCS\artifacts\vul-cgbt\codeql-db`
- CodeQL query results: `C:\Users\UCS\artifacts\vul-cgbt\codeql-results`
- graph JSONL: `C:\Users\UCS\artifacts\vul-cgbt\graphs`
- embedding cache: `C:\Users\UCS\artifacts\vul-cgbt\embeddings`
- checkpoints: `C:\Users\UCS\artifacts\vul-cgbt\checkpoints`

## Main Runs

Run both main protocol directions:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\UCS\Vul-CGBT-main\scripts\run_main_protocol_windows.ps1 `
  -Mode all `
  -PythonExe C:\Users\UCS\anaconda3\envs\gcn\python.exe `
  -GraphTag targeted_v4 `
  -Epochs 10 `
  -BatchSize 64 `
  -EvalBatchSize 64 `
  -NumWorkers 0 `
  -ShardCacheSize 64
```

Run only `Reveal -> BigVul`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\UCS\Vul-CGBT-main\scripts\run_main_protocol_windows.ps1 `
  -Mode reveal-to-bigvul `
  -PythonExe C:\Users\UCS\anaconda3\envs\gcn\python.exe
```

Run only `BigVul -> Reveal`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\UCS\Vul-CGBT-main\scripts\run_main_protocol_windows.ps1 `
  -Mode bigvul-to-reveal `
  -PythonExe C:\Users\UCS\anaconda3\envs\gcn\python.exe
```

What the wrapper does:

- builds missing CodeQL graph artifacts
- builds missing GraphCodeBERT embeddings
- trains the main source-domain model
- evaluates the paired main target domain
- evaluates `Devign` as hard-case external test automatically

## Lower-Level Script

If you need a single source-domain run, use:

- [run_cross_dataset_transfer_windows.ps1](C:/Users/UCS/Vul-CGBT-main/scripts/run_cross_dataset_transfer_windows.ps1)

Examples:

- `-Source reveal-generic`
- `-Source bigvul-generic`

The `-Source devign` path is retained only for ablation and debugging.

## Current Practical Defaults

These defaults are stable on the current Windows setup:

- `GraphTag=targeted_v4`
- `FamilyEdgeTypes`
- `BatchSize=64`
- `EvalBatchSize=64`
- `NumWorkers=0`
- `ShardCacheSize=64` for merged or large runs
- `CheckpointMetric=balanced_accuracy`
- `ThresholdMetric=balanced_accuracy`

## Current Research Position

The working interpretation is:

- `Reveal` and `BigVul` share enough structural signal for meaningful transfer.
- `Devign` is a much harder external domain.
- Adding `Devign` source supervision currently hurts generalization more than it helps.
- Therefore the main claim should center on `Reveal <-> BigVul`, with `Devign` reported separately as a hard-case failure domain.

## Related Files

- [run_main_protocol_windows.ps1](C:/Users/UCS/Vul-CGBT-main/scripts/run_main_protocol_windows.ps1)
- [run_cross_dataset_transfer_windows.ps1](C:/Users/UCS/Vul-CGBT-main/scripts/run_cross_dataset_transfer_windows.ps1)
- [run_leave_one_dataset_out_windows.ps1](C:/Users/UCS/Vul-CGBT-main/scripts/run_leave_one_dataset_out_windows.ps1)
- [run_research_codeql_pipeline_windows.ps1](C:/Users/UCS/Vul-CGBT-main/scripts/run_research_codeql_pipeline_windows.ps1)
- [run_with_gat.py](C:/Users/UCS/Vul-CGBT-main/code/run_with_gat.py)
- [codeql_windows_migration.md](C:/Users/UCS/Vul-CGBT-main/docs/codeql_windows_migration.md)
- [windows_path_convention.md](C:/Users/UCS/Vul-CGBT-main/docs/windows_path_convention.md)
