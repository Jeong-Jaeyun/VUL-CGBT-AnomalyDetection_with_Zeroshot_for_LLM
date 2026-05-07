param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputJsonl,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$OutputDbName,

    [Parameter(Mandatory = $true, Position = 2)]
    [string]$ArtifactName,

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
    if (Test-Path -LiteralPath $candidate) {
        $env:VULCGBT_CODEQL = $candidate
    }
}

if ([System.IO.Path]::IsPathRooted($InputJsonl)) {
    $inputPath = $InputJsonl
} else {
    $inputPath = Join-Path $env:VULCGBT_REPO $InputJsonl
}

if ([System.IO.Path]::IsPathRooted($OutputDbName)) {
    $outputDbPath = $OutputDbName
} else {
    $outputDbPath = Join-Path $env:VULCGBT_CODEQL_DB $OutputDbName
}

if ([System.IO.Path]::IsPathRooted($ArtifactName)) {
    $artifactDir = $ArtifactName
} else {
    $artifactDir = Join-Path $env:VULCGBT_CODEQL_RESULTS $ArtifactName
}

New-Item -ItemType Directory -Force -Path $env:VULCGBT_CODEQL_DB | Out-Null
New-Item -ItemType Directory -Force -Path $env:VULCGBT_CODEQL_RESULTS | Out-Null
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null

Set-Location $env:VULCGBT_REPO

$command = @(
    "python",
    "scripts/build_codeql_database.py",
    "--input", $inputPath,
    "--output-db", $outputDbPath,
    "--artifact-dir", $artifactDir
)

if ($env:VULCGBT_CODEQL) {
    $command += @("--codeql", $env:VULCGBT_CODEQL)
}

if ($ExtraArgs) {
    $command += $ExtraArgs
}

& $command[0] $command[1..($command.Length - 1)]
