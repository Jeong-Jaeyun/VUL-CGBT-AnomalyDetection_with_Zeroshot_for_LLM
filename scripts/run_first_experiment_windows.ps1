param(
    [switch]$RerunQuery,
    [switch]$RerunEmbeddings,
    [switch]$SkipReveal,
    [switch]$ResumeFromBestCheckpoint,
    [int]$Epochs = 20,
    [int]$BatchSize = 64,
    [int]$EvalBatchSize = 64,
    [int]$NumWorkers = 0,
    [int]$ShardCacheSize = 8,
    [string]$CheckpointMetric = "balanced_accuracy",
    [string]$ThresholdMetric = "balanced_accuracy",
    [string]$ExperimentPrefix = "first_experiment",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

if (-not $env:VULCGBT_REPO) {
    $env:VULCGBT_REPO = "C:\Users\UCS\Vul-CGBT-main"
}
if (-not $env:VULCGBT_ARTIFACTS) {
    $env:VULCGBT_ARTIFACTS = "C:\Users\UCS\artifacts\vul-cgbt"
}
if (-not $env:VULCGBT_GRAPHS) {
    $env:VULCGBT_GRAPHS = Join-Path $env:VULCGBT_ARTIFACTS "graphs"
}
if (-not $env:VULCGBT_EMBEDDINGS) {
    $env:VULCGBT_EMBEDDINGS = Join-Path $env:VULCGBT_ARTIFACTS "embeddings"
}
if (-not $env:VULCGBT_CHECKPOINTS) {
    $env:VULCGBT_CHECKPOINTS = Join-Path $env:VULCGBT_ARTIFACTS "checkpoints"
}
if (-not $env:GRAPHCODEBERT_DIR) {
    $graphCodeBertDir = "C:\Users\UCS\models\graphcodebert-base"
    if (Test-Path $graphCodeBertDir) {
        $env:GRAPHCODEBERT_DIR = $graphCodeBertDir
    }
}

$pythonExe = if ($PythonExe) {
    $PythonExe
} elseif ($env:CONDA_PREFIX) {
    Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    "python"
}
$pipelineWrapper = Join-Path $env:VULCGBT_REPO "scripts\run_research_codeql_pipeline_windows.ps1"

function Invoke-PipelineProfile {
    param([string]$ProfileName)

    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $pipelineWrapper,
        $ProfileName
    )
    if ($RerunQuery) {
        $args += "-RerunQuery"
    }
    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline profile failed: $ProfileName"
    }
}

function Invoke-EmbeddingBuild {
    param(
        [string]$GraphFile,
        [string]$EmbeddingDirName
    )

    $embeddingDir = Join-Path $env:VULCGBT_EMBEDDINGS $EmbeddingDirName
    $manifestPath = Join-Path $embeddingDir "manifest.jsonl"
    if ((-not $RerunEmbeddings) -and (Test-Path $manifestPath)) {
        return
    }

    $graphPath = if ([System.IO.Path]::IsPathRooted($GraphFile)) { $GraphFile } else { Join-Path $env:VULCGBT_GRAPHS $GraphFile }
    $args = @(
        "scripts/build_graph_embeddings.py",
        "--input", $graphPath,
        "--output-dir", $embeddingDir
    )
    & $pythonExe @args
    if ($LASTEXITCODE -ne 0) {
        throw "Embedding build failed: $EmbeddingDirName"
    }
}

function Invoke-GatRun {
    param(
        [string]$TrainGraphs,
        [string]$TrainEmbeddings,
        [string]$ValidGraphs,
        [string]$ValidEmbeddings,
        [string]$TestGraphs,
        [string]$TestEmbeddings,
        [string]$ExperimentName,
        [string[]]$ExtraArgs
    )

    $trainGraphsPath = if ([System.IO.Path]::IsPathRooted($TrainGraphs)) { $TrainGraphs } else { Join-Path $env:VULCGBT_GRAPHS $TrainGraphs }
    $validGraphsPath = if ([System.IO.Path]::IsPathRooted($ValidGraphs)) { $ValidGraphs } else { Join-Path $env:VULCGBT_GRAPHS $ValidGraphs }
    $testGraphsPath = if ([System.IO.Path]::IsPathRooted($TestGraphs)) { $TestGraphs } else { Join-Path $env:VULCGBT_GRAPHS $TestGraphs }
    $trainEmbeddingsPath = if ([System.IO.Path]::IsPathRooted($TrainEmbeddings)) { $TrainEmbeddings } else { Join-Path $env:VULCGBT_EMBEDDINGS $TrainEmbeddings }
    $validEmbeddingsPath = if ([System.IO.Path]::IsPathRooted($ValidEmbeddings)) { $ValidEmbeddings } else { Join-Path $env:VULCGBT_EMBEDDINGS $ValidEmbeddings }
    $testEmbeddingsPath = if ([System.IO.Path]::IsPathRooted($TestEmbeddings)) { $TestEmbeddings } else { Join-Path $env:VULCGBT_EMBEDDINGS $TestEmbeddings }
    $outputDir = Join-Path $env:VULCGBT_CHECKPOINTS $ExperimentName
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    $args = @(
        "code/run_with_gat.py",
        "--train-graphs", $trainGraphsPath,
        "--train-embeddings", $trainEmbeddingsPath,
        "--valid-graphs", $validGraphsPath,
        "--valid-embeddings", $validEmbeddingsPath,
        "--test-graphs", $testGraphsPath,
        "--test-embeddings", $testEmbeddingsPath,
        "--output-dir", $outputDir
    )
    if ($ExtraArgs) {
        $args += $ExtraArgs
    }
    & $pythonExe @args
    if ($LASTEXITCODE -ne 0) {
        throw "GAT run failed: $ExperimentName"
    }
}

Set-Location $env:VULCGBT_REPO

$profiles = @(
    "sard-train-cwe119",
    "sard-val-cwe119",
    "sard-test-cwe119",
    "bigvul-test-cwe119",
    "bigvul-test-cwe78",
    "devign-test"
)
if (-not $SkipReveal) {
    $profiles += "reveal-test"
}

foreach ($profile in $profiles) {
    Invoke-PipelineProfile -ProfileName $profile
}

$embeddingTargets = @(
    @{ Graph = "sard_cwe119_train_graphs_targeted_v1.jsonl"; Embedding = "sard_cwe119_train_graphcodebert" },
    @{ Graph = "sard_cwe119_val_graphs_targeted_v1.jsonl"; Embedding = "sard_cwe119_val_graphcodebert" },
    @{ Graph = "sard_cwe119_test_graphs_targeted_v1.jsonl"; Embedding = "sard_cwe119_test_graphcodebert" },
    @{ Graph = "bigvul_cwe119_test_graphs_targeted_v1.jsonl"; Embedding = "bigvul_cwe119_test_graphcodebert" },
    @{ Graph = "bigvul_cwe78_test_graphs_targeted_v1.jsonl"; Embedding = "bigvul_cwe78_test_graphcodebert" },
    @{ Graph = "devign_test_graphs_targeted_v1.jsonl"; Embedding = "devign_test_graphcodebert" }
)
if (-not $SkipReveal) {
    $embeddingTargets += @{ Graph = "reveal_test_graphs_targeted_v1.jsonl"; Embedding = "reveal_test_graphcodebert" }
}

foreach ($target in $embeddingTargets) {
    Invoke-EmbeddingBuild -GraphFile $target.Graph -EmbeddingDirName $target.Embedding
}

$mainExperimentName = "${ExperimentPrefix}_sard_cwe119_main"
$mainCheckpointDir = Join-Path $env:VULCGBT_CHECKPOINTS $mainExperimentName
$bestCheckpointPath = Join-Path $mainCheckpointDir "best_model.pt"
$selectedThresholdPath = Join-Path $mainCheckpointDir "selected_threshold.json"

if ($ResumeFromBestCheckpoint) {
    if (-not (Test-Path $bestCheckpointPath)) {
        throw "Best checkpoint not found for resume: $bestCheckpointPath"
    }

    Invoke-GatRun `
        -TrainGraphs "sard_cwe119_train_graphs_targeted_v1.jsonl" `
        -TrainEmbeddings "sard_cwe119_train_graphcodebert" `
        -ValidGraphs "sard_cwe119_val_graphs_targeted_v1.jsonl" `
        -ValidEmbeddings "sard_cwe119_val_graphcodebert" `
        -TestGraphs "sard_cwe119_test_graphs_targeted_v1.jsonl" `
        -TestEmbeddings "sard_cwe119_test_graphcodebert" `
        -ExperimentName $mainExperimentName `
        -ExtraArgs @(
            "--epochs", "0",
            "--batch-size", "$BatchSize",
            "--eval-batch-size", "$EvalBatchSize",
            "--num-workers", "$NumWorkers",
            "--shard-cache-size", "$ShardCacheSize",
            "--checkpoint-metric", $CheckpointMetric,
            "--threshold-metric", $ThresholdMetric,
            "--checkpoint-path", $bestCheckpointPath,
            "--tune-threshold-on-valid",
            "--save-explanations"
        )
} else {
    Invoke-GatRun `
        -TrainGraphs "sard_cwe119_train_graphs_targeted_v1.jsonl" `
        -TrainEmbeddings "sard_cwe119_train_graphcodebert" `
        -ValidGraphs "sard_cwe119_val_graphs_targeted_v1.jsonl" `
        -ValidEmbeddings "sard_cwe119_val_graphcodebert" `
        -TestGraphs "sard_cwe119_test_graphs_targeted_v1.jsonl" `
        -TestEmbeddings "sard_cwe119_test_graphcodebert" `
        -ExperimentName $mainExperimentName `
        -ExtraArgs @(
            "--epochs", "$Epochs",
            "--batch-size", "$BatchSize",
            "--eval-batch-size", "$EvalBatchSize",
            "--num-workers", "$NumWorkers",
            "--shard-cache-size", "$ShardCacheSize",
            "--checkpoint-metric", $CheckpointMetric,
            "--threshold-metric", $ThresholdMetric,
            "--tune-threshold-on-valid",
            "--save-explanations"
        )
}

if (-not (Test-Path $bestCheckpointPath)) {
    throw "Best checkpoint not found: $bestCheckpointPath"
}
if (-not (Test-Path $selectedThresholdPath)) {
    throw "Selected threshold not found: $selectedThresholdPath"
}

$thresholdPayload = Get-Content $selectedThresholdPath -Raw | ConvertFrom-Json
$selectedThreshold = [string]$thresholdPayload.selected_threshold

$evaluationTargets = @(
    @{ Name = "${ExperimentPrefix}_eval_bigvul_cwe119"; Graph = "bigvul_cwe119_test_graphs_targeted_v1.jsonl"; Embedding = "bigvul_cwe119_test_graphcodebert" },
    @{ Name = "${ExperimentPrefix}_eval_bigvul_cwe78"; Graph = "bigvul_cwe78_test_graphs_targeted_v1.jsonl"; Embedding = "bigvul_cwe78_test_graphcodebert" },
    @{ Name = "${ExperimentPrefix}_eval_devign"; Graph = "devign_test_graphs_targeted_v1.jsonl"; Embedding = "devign_test_graphcodebert" }
)
if (-not $SkipReveal) {
    $evaluationTargets += @{ Name = "${ExperimentPrefix}_eval_reveal"; Graph = "reveal_test_graphs_targeted_v1.jsonl"; Embedding = "reveal_test_graphcodebert" }
}

foreach ($target in $evaluationTargets) {
    Invoke-GatRun `
        -TrainGraphs "sard_cwe119_train_graphs_targeted_v1.jsonl" `
        -TrainEmbeddings "sard_cwe119_train_graphcodebert" `
        -ValidGraphs "sard_cwe119_val_graphs_targeted_v1.jsonl" `
        -ValidEmbeddings "sard_cwe119_val_graphcodebert" `
        -TestGraphs $target.Graph `
        -TestEmbeddings $target.Embedding `
        -ExperimentName $target.Name `
        -ExtraArgs @(
            "--epochs", "0",
            "--batch-size", "$BatchSize",
            "--eval-batch-size", "$EvalBatchSize",
            "--num-workers", "$NumWorkers",
            "--shard-cache-size", "$ShardCacheSize",
            "--checkpoint-metric", $CheckpointMetric,
            "--threshold-metric", $ThresholdMetric,
            "--checkpoint-path", $bestCheckpointPath,
            "--prediction-threshold", $selectedThreshold,
            "--save-explanations"
        )
}
