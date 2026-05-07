param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Profile,

    [int]$MaxSamples = 0,
    [int]$MetadataPairChunkSizeOverride = 0,
    [int]$QueryThreads = 8,
    [int]$QueryRam = 24576,

    [switch]$RerunQuery,
    [switch]$RerunGraphs,
    [switch]$FamilyEdgeTypes,
    [string]$GraphTag = "targeted_v1"
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
if (-not $env:VULCGBT_GRAPHS) {
    $env:VULCGBT_GRAPHS = Join-Path $env:VULCGBT_ARTIFACTS "graphs"
}

$pythonExe = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { "python" }

$profiles = @{
    "sard-train-cwe119" = @{
        Input = "dataset\research_v1\sard_cwe119_78\train_cwe119.jsonl"
        DbName = "sard_cwe119_train"
        ArtifactName = "sard_cwe119_train"
        RawSourceRoot = "dataset\SARD"
        AllowCompileErrors = $false
        MetadataPairChunkSize = 1000
    }
    "sard-val-cwe119" = @{
        Input = "dataset\research_v1\sard_cwe119_78\vale_cwe119.jsonl"
        DbName = "sard_cwe119_val"
        ArtifactName = "sard_cwe119_val"
        RawSourceRoot = "dataset\SARD"
        AllowCompileErrors = $false
        MetadataPairChunkSize = 1000
    }
    "sard-test-cwe119" = @{
        Input = "dataset\research_v1\sard_cwe119_78\test_cwe119.jsonl"
        DbName = "sard_cwe119_test"
        ArtifactName = "sard_cwe119_test"
        RawSourceRoot = "dataset\SARD"
        AllowCompileErrors = $false
        MetadataPairChunkSize = 1000
    }
    "bigvul-test-cwe119" = @{
        Input = "dataset\research_v1\bigvul_cwe119_78_binary_subset\test_cwe119.jsonl"
        DbName = "bigvul_cwe119_test"
        ArtifactName = "bigvul_cwe119_test"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 250
    }
    "bigvul-test-cwe78" = @{
        Input = "dataset\research_v1\bigvul_cwe119_78_binary_subset\test_cwe78.jsonl"
        DbName = "bigvul_cwe78_test"
        ArtifactName = "bigvul_cwe78_test"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 50
    }
    "bigvul-ft-train" = @{
        Input = "dataset\research_v1\bigvul_cwe119_78_finetune_subset\train.jsonl"
        DbName = "bigvul_finetune_train"
        ArtifactName = "bigvul_finetune_train"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 250
    }
    "bigvul-ft-vale" = @{
        Input = "dataset\research_v1\bigvul_cwe119_78_finetune_subset\vale.jsonl"
        DbName = "bigvul_finetune_vale"
        ArtifactName = "bigvul_finetune_vale"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 125
    }
    "bigvul-ft-test" = @{
        Input = "dataset\research_v1\bigvul_cwe119_78_finetune_subset\test.jsonl"
        DbName = "bigvul_finetune_test"
        ArtifactName = "bigvul_finetune_test"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 125
    }
    "bigvul-generic-train" = @{
        Input = "dataset\research_v1\bigvul_generic_binary_split\train.jsonl"
        DbName = "bigvul_generic_train"
        ArtifactName = "bigvul_generic_train"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 250
    }
    "bigvul-generic-vale" = @{
        Input = "dataset\research_v1\bigvul_generic_binary_split\vale.jsonl"
        DbName = "bigvul_generic_vale"
        ArtifactName = "bigvul_generic_vale"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 125
    }
    "bigvul-generic-test" = @{
        Input = "dataset\research_v1\bigvul_generic_binary_split\test.jsonl"
        DbName = "bigvul_generic_test"
        ArtifactName = "bigvul_generic_test"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 125
    }
    "devign-train" = @{
        Input = "dataset\devign\train.jsonl"
        DbName = "devign_train"
        ArtifactName = "devign_train"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 250
    }
    "devign-source-train" = @{
        Input = "dataset\research_v1\devign_source_balanced_subset\train.jsonl"
        DbName = "devign_source_train"
        ArtifactName = "devign_source_train"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 50
    }
    "devign-vale" = @{
        Input = "dataset\devign\vale.jsonl"
        DbName = "devign_vale"
        ArtifactName = "devign_vale"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 125
    }
    "devign-source-vale" = @{
        Input = "dataset\research_v1\devign_source_balanced_subset\vale.jsonl"
        DbName = "devign_source_vale"
        ArtifactName = "devign_source_vale"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 25
    }
    "devign-test" = @{
        Input = "dataset\devign\test.jsonl"
        DbName = "devign_test"
        ArtifactName = "devign_test"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 250
    }
    "reveal-test" = @{
        Input = "dataset\reveal\test.jsonl"
        DbName = "reveal_test"
        ArtifactName = "reveal_test"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 250
    }
    "reveal-generic-train" = @{
        Input = "dataset\research_v1\reveal_generic_binary_split\train.jsonl"
        DbName = "reveal_generic_train"
        ArtifactName = "reveal_generic_train"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 250
    }
    "reveal-generic-vale" = @{
        Input = "dataset\research_v1\reveal_generic_binary_split\vale.jsonl"
        DbName = "reveal_generic_vale"
        ArtifactName = "reveal_generic_vale"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 125
    }
    "reveal-generic-test" = @{
        Input = "dataset\research_v1\reveal_generic_binary_split\test.jsonl"
        DbName = "reveal_generic_test"
        ArtifactName = "reveal_generic_test"
        RawSourceRoot = $null
        AllowCompileErrors = $true
        MetadataPairChunkSize = 125
    }
}

function Invoke-PipelineProfile {
    param([string]$ProfileName)

    if (-not $profiles.ContainsKey($ProfileName)) {
        throw "Unknown profile: $ProfileName"
    }

    $config = $profiles[$ProfileName]
    $dbPath = Join-Path $env:VULCGBT_CODEQL_DB $config.DbName
    $artifactDir = Join-Path $env:VULCGBT_CODEQL_RESULTS $config.ArtifactName
    $dbSummaryPath = Join-Path $artifactDir "summary.json"
    $queryArtifactDir = "${artifactDir}_query_targeted_v1"
    $querySummaryPath = Join-Path $queryArtifactDir "summary.json"
    $graphPath = Join-Path $env:VULCGBT_GRAPHS ("{0}_graphs_{1}.jsonl" -f $config.ArtifactName, $GraphTag)
    $graphSummaryPath = $graphPath -replace "\.jsonl$", ".summary.json"

    $dbArgs = @(
        "scripts/build_codeql_database.py",
        "--input", (Join-Path $env:VULCGBT_REPO $config.Input),
        "--output-db", $dbPath,
        "--artifact-dir", $artifactDir
    )
    if ($MaxSamples -gt 0) {
        $dbArgs += @("--max-samples", "$MaxSamples")
    }
    if ($config.RawSourceRoot) {
        $dbArgs += @("--raw-source-root", (Join-Path $env:VULCGBT_REPO $config.RawSourceRoot))
    }
    if ($config.AllowCompileErrors) {
        $dbArgs += "--allow-compile-errors"
    }

    if (-not ((Test-Path $dbPath) -and (Test-Path $dbSummaryPath))) {
        & $pythonExe @dbArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $queryArgs = @(
        "scripts/codeql_extract_paths.py",
        "--database", $dbPath,
        "--database-summary", $dbSummaryPath,
        "--artifact-dir", $queryArtifactDir,
        "--threads", ([string][int]$QueryThreads),
        "--ram", ([string][int]$QueryRam)
    )
    $effectiveMetadataPairChunkSize = 0
    if ($MetadataPairChunkSizeOverride -gt 0) {
        $effectiveMetadataPairChunkSize = [int]$MetadataPairChunkSizeOverride
    } elseif ($config.ContainsKey("MetadataPairChunkSize") -and [int]$config.MetadataPairChunkSize -gt 0) {
        $effectiveMetadataPairChunkSize = [int]$config.MetadataPairChunkSize
    }
    if ($effectiveMetadataPairChunkSize -gt 0) {
        $queryArgs += @("--metadata-pair-chunk-size", ([string][int]$effectiveMetadataPairChunkSize))
    }
    if ($RerunQuery) {
        $queryArgs += "--rerun"
    }

    if ($RerunQuery -or -not (Test-Path $querySummaryPath)) {
        & $pythonExe @queryArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $graphArgs = @(
        "scripts/build_codeql_graphs.py",
        "--database-summary", $dbSummaryPath,
        "--query-summary", $querySummaryPath,
        "--output", $graphPath
    )
    if ($FamilyEdgeTypes -or $GraphTag -eq "targeted_v4") {
        $graphArgs += "--family-edge-types"
    }

    if ($RerunQuery -or $RerunGraphs -or -not (Test-Path $graphSummaryPath)) {
        & $pythonExe @graphArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

Push-Location $env:VULCGBT_REPO
try {
    if ($Profile -eq "all-main") {
        foreach ($name in @(
            "sard-train-cwe119",
            "sard-val-cwe119",
            "sard-test-cwe119",
            "bigvul-test-cwe119",
            "bigvul-test-cwe78",
            "devign-test",
            "reveal-test"
        )) {
            Invoke-PipelineProfile -ProfileName $name
        }
    } elseif ($Profile -eq "all-finetune-prep") {
        foreach ($name in @(
            "bigvul-ft-train",
            "bigvul-ft-vale",
            "bigvul-ft-test",
            "devign-train",
            "devign-vale",
            "devign-test"
        )) {
            Invoke-PipelineProfile -ProfileName $name
        }
    } elseif ($Profile -eq "all-bigvul-generic") {
        foreach ($name in @(
            "bigvul-generic-train",
            "bigvul-generic-vale",
            "bigvul-generic-test"
        )) {
            Invoke-PipelineProfile -ProfileName $name
        }
    } elseif ($Profile -eq "all-reveal-generic") {
        foreach ($name in @(
            "reveal-generic-train",
            "reveal-generic-vale",
            "reveal-generic-test"
        )) {
            Invoke-PipelineProfile -ProfileName $name
        }
    } elseif ($Profile -eq "all-external-tests") {
        foreach ($name in @(
            "bigvul-test-cwe119",
            "bigvul-test-cwe78",
            "devign-test",
            "reveal-test"
        )) {
            Invoke-PipelineProfile -ProfileName $name
        }
    } elseif ($Profile -eq "all-four-datasets") {
        foreach ($name in @(
            "sard-train-cwe119",
            "sard-val-cwe119",
            "sard-test-cwe119",
            "bigvul-test-cwe119",
            "bigvul-test-cwe78",
            "devign-test",
            "reveal-test"
        )) {
            Invoke-PipelineProfile -ProfileName $name
        }
    } elseif ($Profile -eq "all-three-datasets") {
        foreach ($name in @(
            "sard-train-cwe119",
            "sard-val-cwe119",
            "sard-test-cwe119",
            "bigvul-test-cwe119",
            "bigvul-test-cwe78",
            "devign-test"
        )) {
            Invoke-PipelineProfile -ProfileName $name
        }
    } else {
        Invoke-PipelineProfile -ProfileName $Profile
    }
}
finally {
    Pop-Location
}
