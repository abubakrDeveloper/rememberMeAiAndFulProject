param(
    [ValidateSet("sample", "camera", "video")]
    [string]$Mode = "sample",
    [string]$Source = "0",
    [switch]$ShowPreview,
    [switch]$SeedFakeData,
    [switch]$SaveOutput,
    [string]$OutputVideo = "",
    [int]$MaxFrames = 0
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "Python virtual environment not found at .venv\Scripts\python.exe"
    Write-Host "Create it first: python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt"
    exit 1
}

New-Item -ItemType Directory -Path "outputs\logs" -Force | Out-Null
New-Item -ItemType Directory -Path "outputs\alerts" -Force | Out-Null
New-Item -ItemType Directory -Path "outputs\video" -Force | Out-Null

if ($SeedFakeData) {
    & $pythonExe -m app.generate_fake_data --overwrite --rows 28
}

if ($Mode -eq "sample") {
    $Source = "sample"
}
elseif ($Mode -eq "video") {
    if (-not (Test-Path $Source)) {
        Write-Host "Video source path not found: $Source"
        exit 1
    }
}

$recognitionArgs = @("-m", "app.recognition", "--source", $Source)

if ($Mode -eq "sample") {
    $recognitionArgs += "--sample-mode"
}

if ($Mode -eq "video") {
    $recognitionArgs += "--offline"
    $recognitionArgs += "--save-output"
}

if ($SaveOutput) {
    $recognitionArgs += "--save-output"
}

if ($OutputVideo -ne "") {
    $recognitionArgs += @("--output-video", $OutputVideo)
}

if (-not $ShowPreview) {
    $recognitionArgs += "--no-display"
}

if ($MaxFrames -gt 0) {
    $recognitionArgs += @("--max-frames", "$MaxFrames")
}

$dashboardArgs = @("-m", "streamlit", "run", "app/dashboard.py")

$recognitionProcess = Start-Process -FilePath $pythonExe -ArgumentList $recognitionArgs -WorkingDirectory $projectRoot -PassThru
$dashboardProcess = Start-Process -FilePath $pythonExe -ArgumentList $dashboardArgs -WorkingDirectory $projectRoot -PassThru

Write-Host "Recognition PID: $($recognitionProcess.Id)"
Write-Host "Dashboard PID: $($dashboardProcess.Id)"
Write-Host "Dashboard URL: http://localhost:8501"
Write-Host "To stop: Stop-Process -Id $($recognitionProcess.Id),$($dashboardProcess.Id)"
