param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("bigvul", "devign")]
    [string]$Dataset,

    [switch]$RerunQuery,
    [switch]$RerunEmbeddings,
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$EvalBatchSize = 64,
    [int]$NumWorkers = 0,
    [int]$ShardCacheSize = 8,
    [string]$ExperimentPrefix = "target_finetune",
    [string]$PythonExe = "",
    [string]$BaseCheckpoint = ""
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

$pythonExe = if ($PythonExe) {
    $PythonExe
} elseif ($env:CONDA_PREFIX) {
    Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    "python"
}

$pipelineWrapper = Join-Path $env:VULCGBT_REPO "scripts\run_research_codeql_pipeline_windows.ps1"

$defaultBaseCheckpoint = Join-Path $env:VULCGBT_CHECKPOINTS "phase1_codeql_core_masked_sard_cwe119_main\best_model.pt"
$baseCheckpointPath = if ($BaseCheckpoint) { $BaseCheckpoint } else { $defaultBaseCheckpoint }
if (-not (Test-Path $baseCheckpointPath)) {
    throw "Base checkpoint not found: $baseCheckpointPath"
}
$baseCheckpointDir = Split-Path -Parent $baseCheckpointPath
$baseRelationVocabPath = Join-Path $baseCheckpointDir "relation_vocab.json"
$baseNodeTypeVocabPath = Join-Path $baseCheckpointDir "node_type_vocab.json"
if (-not (Test-Path $baseRelationVocabPath)) {
    throw "Base relation vocab not found: $baseRelationVocabPath"
}
if (-not (Test-Path $baseNodeTypeVocabPath)) {
    throw "Base node type vocab not found: $baseNodeTypeVocabPath"
}

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

    $outputDir = Join-Path $env:VULCGBT_CHECKPOINTS $ExperimentName
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    $args = @(
        "code/run_with_gat.py",
        "--train-graphs", (Join-Path $env:VULCGBT_GRAPHS $TrainGraphs),
        "--train-embeddings", (Join-Path $env:VULCGBT_EMBEDDINGS $TrainEmbeddings),
        "--valid-graphs", (Join-Path $env:VULCGBT_GRAPHS $ValidGraphs),
        "--valid-embeddings", (Join-Path $env:VULCGBT_EMBEDDINGS $ValidEmbeddings),
        "--test-graphs", (Join-Path $env:VULCGBT_GRAPHS $TestGraphs),
        "--test-embeddings", (Join-Path $env:VULCGBT_EMBEDDINGS $TestEmbeddings),
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

$datasetConfig = @{
    "bigvul" = @{
        Profiles = @("bigvul-ft-train", "bigvul-ft-vale", "bigvul-ft-test")
        TrainGraph = "bigvul_finetune_train_graphs_targeted_v1.jsonl"
        ValGraph = "bigvul_finetune_vale_graphs_targeted_v1.jsonl"
        TestGraph = "bigvul_finetune_test_graphs_targeted_v1.jsonl"
        TrainEmbedding = "bigvul_finetune_train_graphcodebert"
        ValEmbedding = "bigvul_finetune_vale_graphcodebert"
        TestEmbedding = "bigvul_finetune_test_graphcodebert"
    }
    "devign" = @{
        Profiles = @("devign-train", "devign-vale", "devign-test")
        TrainGraph = "devign_train_graphs_targeted_v1.jsonl"
        ValGraph = "devign_vale_graphs_targeted_v1.jsonl"
        TestGraph = "devign_test_graphs_targeted_v1.jsonl"
        TrainEmbedding = "devign_train_graphcodebert"
        ValEmbedding = "devign_vale_graphcodebert"
        TestEmbedding = "devign_test_graphcodebert"
    }
}[$Dataset]

foreach ($profile in $datasetConfig.Profiles) {
    Invoke-PipelineProfile -ProfileName $profile
}

foreach ($target in @(
    @{ Graph = $datasetConfig.TrainGraph; Embedding = $datasetConfig.TrainEmbedding },
    @{ Graph = $datasetConfig.ValGraph; Embedding = $datasetConfig.ValEmbedding },
    @{ Graph = $datasetConfig.TestGraph; Embedding = $datasetConfig.TestEmbedding }
)) {
    Invoke-EmbeddingBuild -GraphFile $target.Graph -EmbeddingDirName $target.Embedding
}

$experimentName = "${ExperimentPrefix}_${Dataset}_finetune"

Invoke-GatRun `
    -TrainGraphs $datasetConfig.TrainGraph `
    -TrainEmbeddings $datasetConfig.TrainEmbedding `
    -ValidGraphs $datasetConfig.ValGraph `
    -ValidEmbeddings $datasetConfig.ValEmbedding `
    -TestGraphs $datasetConfig.TestGraph `
    -TestEmbeddings $datasetConfig.TestEmbedding `
    -ExperimentName $experimentName `
    -ExtraArgs @(
        "--checkpoint-path", $baseCheckpointPath,
        "--relation-vocab-path", $baseRelationVocabPath,
        "--node-type-vocab-path", $baseNodeTypeVocabPath,
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--eval-batch-size", "$EvalBatchSize",
        "--num-workers", "$NumWorkers",
        "--shard-cache-size", "$ShardCacheSize",
        "--tune-threshold-on-valid",
        "--save-explanations"
    )
