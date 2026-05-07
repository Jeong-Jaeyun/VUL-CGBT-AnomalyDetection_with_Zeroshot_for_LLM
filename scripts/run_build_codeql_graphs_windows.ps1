param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$DatabaseSummaryPath,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$QuerySummaryPath,

    [Parameter(Mandatory = $true, Position = 2)]
    [string]$OutputPathOrName,

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

$outputPath = if ([System.IO.Path]::IsPathRooted($OutputPathOrName)) {
    $OutputPathOrName
} else {
    Join-Path $env:VULCGBT_GRAPHS $OutputPathOrName
}

$pythonArgs = @(
    "scripts/build_codeql_graphs.py",
    "--database-summary", $DatabaseSummaryPath,
    "--query-summary", $QuerySummaryPath,
    "--output", $outputPath
)
if ($ExtraArgs) {
    $pythonArgs += $ExtraArgs
}

Push-Location $env:VULCGBT_REPO
try {
    python @pythonArgs
}
finally {
    Pop-Location
}
