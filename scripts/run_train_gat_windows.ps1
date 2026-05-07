param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$TrainGraphs,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$TrainEmbeddings,

    [Parameter(Mandatory = $true, Position = 2)]
    [string]$ValidGraphs,

    [Parameter(Mandatory = $true, Position = 3)]
    [string]$ValidEmbeddings,

    [Parameter(Mandatory = $true, Position = 4)]
    [string]$TestGraphs,

    [Parameter(Mandatory = $true, Position = 5)]
    [string]$TestEmbeddings,

    [Parameter(Mandatory = $true, Position = 6)]
    [string]$ExperimentName,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
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

$pythonExe = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { "python" }

function Resolve-ArtifactPath {
    param(
        [string]$Value,
        [string]$BaseDir
    )

    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }
    return (Join-Path $BaseDir $Value)
}

$trainGraphsPath = Resolve-ArtifactPath $TrainGraphs $env:VULCGBT_GRAPHS
$validGraphsPath = Resolve-ArtifactPath $ValidGraphs $env:VULCGBT_GRAPHS
$testGraphsPath = Resolve-ArtifactPath $TestGraphs $env:VULCGBT_GRAPHS

$trainEmbeddingsPath = Resolve-ArtifactPath $TrainEmbeddings $env:VULCGBT_EMBEDDINGS
$validEmbeddingsPath = Resolve-ArtifactPath $ValidEmbeddings $env:VULCGBT_EMBEDDINGS
$testEmbeddingsPath = Resolve-ArtifactPath $TestEmbeddings $env:VULCGBT_EMBEDDINGS

$outputDir = Resolve-ArtifactPath $ExperimentName $env:VULCGBT_CHECKPOINTS
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Set-Location $env:VULCGBT_REPO

$command = @(
    $pythonExe,
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
    $command += $ExtraArgs
}

& $command[0] $command[1..($command.Length - 1)]
