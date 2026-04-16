param(
    [ValidateSet("sample", "camera", "video")]
    [string]$Mode = "sample",
    [string]$Source = "0",
    [string]$CourseId = "general",
    [string]$SessionId = "session_1",
    [switch]$ShowPreview,
    [switch]$SeedFakeData,
    [switch]$SaveOutput,
    [string]$OutputVideo = "",
    [int]$MaxFrames = 0
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command python).Source
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command py).Source
}
else {
    Write-Host "Python was not found. Install Python 3.11+ and run: python -m pip install -r requirements.txt"
    exit 1
}

Write-Host "Using Python executable: $pythonExe"

New-Item -ItemType Directory -Path "outputs\logs" -Force | Out-Null
New-Item -ItemType Directory -Path "outputs\alerts" -Force | Out-Null
New-Item -ItemType Directory -Path "outputs\video" -Force | Out-Null

if ($SeedFakeData) {
    & $pythonExe -m app.generate_fake_data --overwrite --rows 28 --course-id $CourseId --session-id $SessionId
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
$recognitionArgs += @("--course-id", $CourseId, "--session-id", $SessionId)

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
