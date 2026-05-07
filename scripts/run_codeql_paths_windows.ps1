param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$DatabasePathOrName,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$ArtifactPathOrName,

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
if (-not $env:VULCGBT_CODEQL_DB) {
    $env:VULCGBT_CODEQL_DB = Join-Path $env:VULCGBT_ARTIFACTS "codeql-db"
}
if (-not $env:VULCGBT_CODEQL_RESULTS) {
    $env:VULCGBT_CODEQL_RESULTS = Join-Path $env:VULCGBT_ARTIFACTS "codeql-results"
}
if (-not $env:VULCGBT_CODEQL) {
    $candidate = "C:\Users\UCS\AppData\Roaming\Code\User\globalStorage\github.vscode-codeql\distribution1\codeql\codeql.exe"
    if (Test-Path $candidate) {
        $env:VULCGBT_CODEQL = $candidate
    }
}

$databasePath = if ([System.IO.Path]::IsPathRooted($DatabasePathOrName)) {
    $DatabasePathOrName
} else {
    Join-Path $env:VULCGBT_CODEQL_DB $DatabasePathOrName
}

$artifactPath = if ([System.IO.Path]::IsPathRooted($ArtifactPathOrName)) {
    $ArtifactPathOrName
} else {
    Join-Path $env:VULCGBT_CODEQL_RESULTS $ArtifactPathOrName
}

$pythonArgs = @(
    "scripts/codeql_extract_paths.py",
    "--database", $databasePath,
    "--artifact-dir", $artifactPath
)
if ($env:VULCGBT_CODEQL) {
    $pythonArgs += @("--codeql", $env:VULCGBT_CODEQL)
}
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
