param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputGraphJsonl,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$OutputDirName,

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

$pythonExe = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { "python" }

if ([System.IO.Path]::IsPathRooted($InputGraphJsonl)) {
    $inputPath = $InputGraphJsonl
} else {
    $inputPath = Join-Path $env:VULCGBT_GRAPHS $InputGraphJsonl
}

if ([System.IO.Path]::IsPathRooted($OutputDirName)) {
    $outputDir = $OutputDirName
} else {
    $outputDir = Join-Path $env:VULCGBT_EMBEDDINGS $OutputDirName
}

New-Item -ItemType Directory -Force -Path $env:VULCGBT_EMBEDDINGS | Out-Null

Set-Location $env:VULCGBT_REPO

$command = @(
    $pythonExe,
    "scripts/build_graph_embeddings.py",
    "--input", $inputPath,
    "--output-dir", $outputDir
)

if ($ExtraArgs) {
    $command += $ExtraArgs
}

& $command[0] $command[1..($command.Length - 1)]
