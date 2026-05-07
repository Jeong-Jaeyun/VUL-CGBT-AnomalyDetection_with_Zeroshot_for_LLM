param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("bigvul-generic", "devign", "reveal-generic")]
    [string]$Source,

    [switch]$RerunQuery,
    [switch]$RerunGraphs,
    [switch]$FamilyEdgeTypes,
    [switch]$RerunEmbeddings,
    [switch]$SkipReveal,
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$EvalBatchSize = 64,
    [int]$NumWorkers = 0,
    [int]$ShardCacheSize = 8,
    [string]$CheckpointMetric = "balanced_accuracy",
    [string]$ThresholdMetric = "balanced_accuracy",
    [string]$GraphTag = "targeted_v1",
    [string]$EmbeddingTag = "",
    [string]$ExperimentPrefix = "cross_dataset",
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

$pythonExe = if ($PythonExe) {
    $PythonExe
} elseif ($env:CONDA_PREFIX) {
    Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    "python"
}

$pipelineWrapper = Join-Path $env:VULCGBT_REPO "scripts\run_research_codeql_pipeline_windows.ps1"
$effectiveEmbeddingTag = if ($EmbeddingTag) {
    $EmbeddingTag
} elseif ($GraphTag -ne "targeted_v1") {
    $GraphTag
} else {
    ""
}

function Get-GraphFileName {
    param([string]$ArtifactName)

    return "{0}_graphs_{1}.jsonl" -f $ArtifactName, $GraphTag
}

function Get-EmbeddingDirName {
    param([string]$ArtifactName)

    if ($effectiveEmbeddingTag) {
        return "{0}_graphcodebert_{1}" -f $ArtifactName, $effectiveEmbeddingTag
    }
    return "{0}_graphcodebert" -f $ArtifactName
}

function Invoke-PipelineProfile {
    param([string]$ProfileName)

    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $pipelineWrapper,
        $ProfileName,
        "-GraphTag", $GraphTag
    )
    if ($RerunQuery) {
        $args += "-RerunQuery"
    }
    if ($RerunGraphs) {
        $args += "-RerunGraphs"
    }
    if ($FamilyEdgeTypes -or $GraphTag -eq "targeted_v4") {
        $args += "-FamilyEdgeTypes"
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

    $graphPath = Join-Path $env:VULCGBT_GRAPHS $GraphFile
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

$sourceConfig = @{
    "bigvul-generic" = @{
        Profiles = @("bigvul-generic-train", "bigvul-generic-vale", "bigvul-generic-test", "devign-test")
        TrainArtifact = "bigvul_generic_train"
        ValArtifact = "bigvul_generic_vale"
        TestArtifact = "bigvul_generic_test"
        EvalTargets = @(
            @{ Name = "devign"; Artifact = "devign_test" }
        )
    }
    "devign" = @{
        Profiles = @("devign-source-train", "devign-source-vale", "devign-test")
        TrainArtifact = "devign_source_train"
        ValArtifact = "devign_source_vale"
        TestArtifact = "devign_test"
        EvalTargets = @(
            @{ Name = "bigvul_generic"; Artifact = "bigvul_generic_test" }
        )
    }
    "reveal-generic" = @{
        Profiles = @("reveal-generic-train", "reveal-generic-vale", "reveal-generic-test", "devign-test", "bigvul-generic-test")
        TrainArtifact = "reveal_generic_train"
        ValArtifact = "reveal_generic_vale"
        TestArtifact = "reveal_generic_test"
        EvalTargets = @(
            @{ Name = "devign"; Artifact = "devign_test" },
            @{ Name = "bigvul_generic"; Artifact = "bigvul_generic_test" }
        )
    }
}[$Source]

if ((-not $SkipReveal) -and ($Source -ne "reveal-generic")) {
    $sourceConfig.Profiles += "reveal-generic-test"
    $sourceConfig.EvalTargets += @{ Name = "reveal_generic"; Artifact = "reveal_generic_test" }
}

foreach ($profile in $sourceConfig.Profiles) {
    Invoke-PipelineProfile -ProfileName $profile
}

$embeddingTargets = @(
    @{ Graph = (Get-GraphFileName -ArtifactName $sourceConfig.TrainArtifact); Embedding = (Get-EmbeddingDirName -ArtifactName $sourceConfig.TrainArtifact) },
    @{ Graph = (Get-GraphFileName -ArtifactName $sourceConfig.ValArtifact); Embedding = (Get-EmbeddingDirName -ArtifactName $sourceConfig.ValArtifact) },
    @{ Graph = (Get-GraphFileName -ArtifactName $sourceConfig.TestArtifact); Embedding = (Get-EmbeddingDirName -ArtifactName $sourceConfig.TestArtifact) }
)
foreach ($target in $sourceConfig.EvalTargets) {
    $embeddingTargets += @{
        Graph = Get-GraphFileName -ArtifactName $target.Artifact
        Embedding = Get-EmbeddingDirName -ArtifactName $target.Artifact
    }
}

foreach ($target in $embeddingTargets) {
    Invoke-EmbeddingBuild -GraphFile $target.Graph -EmbeddingDirName $target.Embedding
}

$mainExperimentName = if ($GraphTag -eq "targeted_v1") {
    "${ExperimentPrefix}_${Source}_main"
} else {
    "${ExperimentPrefix}_${Source}_${GraphTag}_main"
}

Invoke-GatRun `
    -TrainGraphs (Get-GraphFileName -ArtifactName $sourceConfig.TrainArtifact) `
    -TrainEmbeddings (Get-EmbeddingDirName -ArtifactName $sourceConfig.TrainArtifact) `
    -ValidGraphs (Get-GraphFileName -ArtifactName $sourceConfig.ValArtifact) `
    -ValidEmbeddings (Get-EmbeddingDirName -ArtifactName $sourceConfig.ValArtifact) `
    -TestGraphs (Get-GraphFileName -ArtifactName $sourceConfig.TestArtifact) `
    -TestEmbeddings (Get-EmbeddingDirName -ArtifactName $sourceConfig.TestArtifact) `
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

$mainCheckpointDir = Join-Path $env:VULCGBT_CHECKPOINTS $mainExperimentName
$bestCheckpointPath = Join-Path $mainCheckpointDir "best_model.pt"
$selectedThresholdPath = Join-Path $mainCheckpointDir "selected_threshold.json"
if (-not (Test-Path $bestCheckpointPath)) {
    throw "Best checkpoint not found: $bestCheckpointPath"
}
if (-not (Test-Path $selectedThresholdPath)) {
    throw "Selected threshold not found: $selectedThresholdPath"
}

$thresholdPayload = Get-Content $selectedThresholdPath -Raw | ConvertFrom-Json
$selectedThreshold = [string]$thresholdPayload.selected_threshold

foreach ($target in $sourceConfig.EvalTargets) {
    $evalExperimentName = if ($GraphTag -eq "targeted_v1") {
        "${ExperimentPrefix}_${Source}_eval_$($target.Name)"
    } else {
        "${ExperimentPrefix}_${Source}_${GraphTag}_eval_$($target.Name)"
    }
    Invoke-GatRun `
        -TrainGraphs (Get-GraphFileName -ArtifactName $sourceConfig.TrainArtifact) `
        -TrainEmbeddings (Get-EmbeddingDirName -ArtifactName $sourceConfig.TrainArtifact) `
        -ValidGraphs (Get-GraphFileName -ArtifactName $sourceConfig.ValArtifact) `
        -ValidEmbeddings (Get-EmbeddingDirName -ArtifactName $sourceConfig.ValArtifact) `
        -TestGraphs (Get-GraphFileName -ArtifactName $target.Artifact) `
        -TestEmbeddings (Get-EmbeddingDirName -ArtifactName $target.Artifact) `
        -ExperimentName $evalExperimentName `
        -ExtraArgs @(
            "--epochs", "0",
            "--batch-size", "$BatchSize",
            "--eval-batch-size", "$EvalBatchSize",
            "--num-workers", "$NumWorkers",
            "--shard-cache-size", "$ShardCacheSize",
            "--checkpoint-path", $bestCheckpointPath,
            "--relation-vocab-path", (Join-Path $mainCheckpointDir "relation_vocab.json"),
            "--node-type-vocab-path", (Join-Path $mainCheckpointDir "node_type_vocab.json"),
            "--prediction-threshold", $selectedThreshold,
            "--save-explanations"
        )
}
