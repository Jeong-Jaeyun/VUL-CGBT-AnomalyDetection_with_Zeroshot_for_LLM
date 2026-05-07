param(
    [ValidateSet("all", "reveal-to-bigvul", "bigvul-to-reveal")]
    [string]$Mode = "all",

    [switch]$RerunQuery,
    [switch]$RerunGraphs,
    [switch]$RerunEmbeddings,
    [switch]$FamilyEdgeTypes,
    [int]$Epochs = 10,
    [int]$BatchSize = 64,
    [int]$EvalBatchSize = 64,
    [int]$NumWorkers = 0,
    [int]$ShardCacheSize = 64,
    [string]$CheckpointMetric = "balanced_accuracy",
    [string]$ThresholdMetric = "balanced_accuracy",
    [string]$GraphTag = "targeted_v4",
    [string]$ExperimentPrefix = "main_protocol",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

if (-not $env:VULCGBT_REPO) {
    $env:VULCGBT_REPO = "C:\Users\UCS\Vul-CGBT-main"
}

$crossDatasetScript = Join-Path $env:VULCGBT_REPO "scripts\run_cross_dataset_transfer_windows.ps1"

function Invoke-CrossDatasetRun {
    param(
        [string]$Source,
        [string]$RunSuffix
    )

    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $crossDatasetScript,
        "-Source", $Source,
        "-GraphTag", $GraphTag,
        "-Epochs", ([string][int]$Epochs),
        "-BatchSize", ([string][int]$BatchSize),
        "-EvalBatchSize", ([string][int]$EvalBatchSize),
        "-NumWorkers", ([string][int]$NumWorkers),
        "-ShardCacheSize", ([string][int]$ShardCacheSize),
        "-CheckpointMetric", $CheckpointMetric,
        "-ThresholdMetric", $ThresholdMetric,
        "-ExperimentPrefix", ("{0}_{1}" -f $ExperimentPrefix, $RunSuffix)
    )

    if ($PythonExe) {
        $args += @("-PythonExe", $PythonExe)
    }
    if ($RerunQuery) {
        $args += "-RerunQuery"
    }
    if ($RerunGraphs) {
        $args += "-RerunGraphs"
    }
    if ($RerunEmbeddings) {
        $args += "-RerunEmbeddings"
    }
    if ($FamilyEdgeTypes -or $GraphTag -eq "targeted_v4") {
        $args += "-FamilyEdgeTypes"
    }

    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Cross-dataset run failed: $Source"
    }
}

switch ($Mode) {
    "all" {
        Invoke-CrossDatasetRun -Source "reveal-generic" -RunSuffix "reveal_to_bigvul"
        Invoke-CrossDatasetRun -Source "bigvul-generic" -RunSuffix "bigvul_to_reveal"
    }
    "reveal-to-bigvul" {
        Invoke-CrossDatasetRun -Source "reveal-generic" -RunSuffix "reveal_to_bigvul"
    }
    "bigvul-to-reveal" {
        Invoke-CrossDatasetRun -Source "bigvul-generic" -RunSuffix "bigvul_to_reveal"
    }
}
