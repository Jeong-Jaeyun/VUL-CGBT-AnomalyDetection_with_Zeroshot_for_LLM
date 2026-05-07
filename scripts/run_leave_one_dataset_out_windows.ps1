param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("bigvul-generic", "devign", "reveal-generic")]
    [string]$Heldout,

    [switch]$RerunQuery,
    [switch]$RerunGraphs,
    [switch]$FamilyEdgeTypes,
    [switch]$RerunEmbeddings,
    [switch]$RerunMerged,
    [int]$MetadataPairChunkSizeOverride = 0,
    [int]$QueryThreads = 8,
    [int]$QueryRam = 24576,
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$EvalBatchSize = 64,
    [int]$NumWorkers = 0,
    [int]$ShardCacheSize = 8,
    [string]$CheckpointMetric = "balanced_accuracy",
    [string]$ThresholdMetric = "balanced_accuracy",
    [string]$GraphTag = "targeted_v4",
    [string]$EmbeddingTag = "",
    [string]$ExperimentPrefix = "lodo",
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
        "-File", (Join-Path $env:VULCGBT_REPO "scripts\run_research_codeql_pipeline_windows.ps1"),
        $ProfileName,
        "-GraphTag", $GraphTag,
        "-QueryThreads", ([string][int]$QueryThreads),
        "-QueryRam", ([string][int]$QueryRam)
    )
    if ($MetadataPairChunkSizeOverride -gt 0) {
        $args += @("-MetadataPairChunkSizeOverride", ([string][int]$MetadataPairChunkSizeOverride))
    }
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
    & $pythonExe "scripts/build_graph_embeddings.py" "--input" $graphPath "--output-dir" $embeddingDir
    if ($LASTEXITCODE -ne 0) {
        throw "Embedding build failed: $EmbeddingDirName"
    }
}

function Invoke-MergeSet {
    param(
        [string]$Name,
        [object[]]$Artifacts
    )

    $graphOutput = Join-Path $env:VULCGBT_GRAPHS ("{0}_{1}_graphs.jsonl" -f $Name, $GraphTag)
    $embeddingOutput = Join-Path $env:VULCGBT_EMBEDDINGS ("{0}_graphcodebert_{1}" -f $Name, $effectiveEmbeddingTag)
    $summaryOutput = Join-Path $env:VULCGBT_ARTIFACTS ("reports\{0}_{1}_merge_summary.json" -f $Name, $GraphTag)

    if ((-not $RerunMerged) -and (Test-Path $graphOutput) -and (Test-Path (Join-Path $embeddingOutput "manifest.jsonl"))) {
        return @{
            Graph = Split-Path $graphOutput -Leaf
            Embedding = Split-Path $embeddingOutput -Leaf
        }
    }

    $args = @("scripts/merge_graph_embedding_sets.py", "--graph-output", $graphOutput, "--embedding-output", $embeddingOutput, "--summary-path", $summaryOutput, "--overwrite")
    foreach ($artifact in $Artifacts) {
        $args += "--graph-input"
        $args += (Join-Path $env:VULCGBT_GRAPHS (Get-GraphFileName -ArtifactName $artifact))
        $args += "--embedding-input"
        $args += (Join-Path $env:VULCGBT_EMBEDDINGS (Get-EmbeddingDirName -ArtifactName $artifact))
    }
    & $pythonExe @args
    if ($LASTEXITCODE -ne 0) {
        throw "Merge failed: $Name"
    }

    return @{
        Graph = Split-Path $graphOutput -Leaf
        Embedding = Split-Path $embeddingOutput -Leaf
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
        [string]$ExperimentName
    )

    $outputDir = Join-Path $env:VULCGBT_CHECKPOINTS $ExperimentName
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    & $pythonExe "code/run_with_gat.py" `
        "--train-graphs" (Join-Path $env:VULCGBT_GRAPHS $TrainGraphs) `
        "--train-embeddings" (Join-Path $env:VULCGBT_EMBEDDINGS $TrainEmbeddings) `
        "--valid-graphs" (Join-Path $env:VULCGBT_GRAPHS $ValidGraphs) `
        "--valid-embeddings" (Join-Path $env:VULCGBT_EMBEDDINGS $ValidEmbeddings) `
        "--test-graphs" (Join-Path $env:VULCGBT_GRAPHS $TestGraphs) `
        "--test-embeddings" (Join-Path $env:VULCGBT_EMBEDDINGS $TestEmbeddings) `
        "--output-dir" $outputDir `
        "--epochs" "$Epochs" `
        "--batch-size" "$BatchSize" `
        "--eval-batch-size" "$EvalBatchSize" `
        "--num-workers" "$NumWorkers" `
        "--shard-cache-size" "$ShardCacheSize" `
        "--checkpoint-metric" $CheckpointMetric `
        "--threshold-metric" $ThresholdMetric `
        "--tune-threshold-on-valid" `
        "--save-explanations"
    if ($LASTEXITCODE -ne 0) {
        throw "GAT run failed: $ExperimentName"
    }
}

Set-Location $env:VULCGBT_REPO

$datasets = @{
    "bigvul-generic" = @{
        TrainProfile = "bigvul-generic-train"
        ValeProfile = "bigvul-generic-vale"
        TestProfile = "bigvul-generic-test"
        TrainArtifact = "bigvul_generic_train"
        ValeArtifact = "bigvul_generic_vale"
        TestArtifact = "bigvul_generic_test"
    }
    "devign" = @{
        TrainProfile = "devign-source-train"
        ValeProfile = "devign-source-vale"
        TestProfile = "devign-test"
        TrainArtifact = "devign_source_train"
        ValeArtifact = "devign_source_vale"
        TestArtifact = "devign_test"
    }
    "reveal-generic" = @{
        TrainProfile = "reveal-generic-train"
        ValeProfile = "reveal-generic-vale"
        TestProfile = "reveal-generic-test"
        TrainArtifact = "reveal_generic_train"
        ValeArtifact = "reveal_generic_vale"
        TestArtifact = "reveal_generic_test"
    }
}

$heldoutConfig = $datasets[$Heldout]
$sourceNames = @($datasets.Keys | Where-Object { $_ -ne $Heldout } | Sort-Object)

$profiles = @()
foreach ($sourceName in $sourceNames) {
    $profiles += $datasets[$sourceName].TrainProfile
    $profiles += $datasets[$sourceName].ValeProfile
}
$profiles += $heldoutConfig.TestProfile

foreach ($profile in $profiles) {
    Invoke-PipelineProfile -ProfileName $profile
}

$embeddingArtifacts = @()
foreach ($sourceName in $sourceNames) {
    $embeddingArtifacts += $datasets[$sourceName].TrainArtifact
    $embeddingArtifacts += $datasets[$sourceName].ValeArtifact
}
$embeddingArtifacts += $heldoutConfig.TestArtifact

foreach ($artifact in $embeddingArtifacts) {
    Invoke-EmbeddingBuild -GraphFile (Get-GraphFileName -ArtifactName $artifact) -EmbeddingDirName (Get-EmbeddingDirName -ArtifactName $artifact)
}

$trainArtifacts = @()
$valeArtifacts = @()
foreach ($sourceName in $sourceNames) {
    $trainArtifacts += $datasets[$sourceName].TrainArtifact
    $valeArtifacts += $datasets[$sourceName].ValeArtifact
}

$sourceSlug = ($sourceNames -join "_").Replace("-", "_")
$heldoutSlug = $Heldout.Replace("-", "_")
$trainMerge = Invoke-MergeSet -Name ("lodo_{0}_train_for_{1}" -f $sourceSlug, $heldoutSlug) -Artifacts $trainArtifacts
$valeMerge = Invoke-MergeSet -Name ("lodo_{0}_vale_for_{1}" -f $sourceSlug, $heldoutSlug) -Artifacts $valeArtifacts

$experimentName = "{0}_{1}_{2}" -f $ExperimentPrefix, $GraphTag, $heldoutSlug
Invoke-GatRun `
    -TrainGraphs $trainMerge.Graph `
    -TrainEmbeddings $trainMerge.Embedding `
    -ValidGraphs $valeMerge.Graph `
    -ValidEmbeddings $valeMerge.Embedding `
    -TestGraphs (Get-GraphFileName -ArtifactName $heldoutConfig.TestArtifact) `
    -TestEmbeddings (Get-EmbeddingDirName -ArtifactName $heldoutConfig.TestArtifact) `
    -ExperimentName $experimentName
