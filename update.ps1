# update.ps1
# -----------
# Run this script whenever you download a new neuro_ai.zip
# It copies only the changed .py files into your existing folder
# WITHOUT touching neuro_env — no reinstall needed.
#
# Usage:
#   1. Download new neuro_ai.zip
#   2. Put neuro_ai.zip anywhere (e.g. Desktop)
#   3. Open terminal in your ai-ml-agent folder
#   4. Run: .\update.ps1 -ZipPath "C:\Users\YourName\Desktop\neuro_ai.zip"

param(
    [Parameter(Mandatory=$true)]
    [string]$ZipPath
)

# Current script location = your ai-ml-agent folder
$TargetFolder = $PSScriptRoot

Write-Host ""
Write-Host "Neuro AI Updater" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host "ZIP    : $ZipPath"
Write-Host "Target : $TargetFolder"
Write-Host ""

# Check zip exists
if (-not (Test-Path $ZipPath)) {
    Write-Host "ERROR: ZIP file not found at: $ZipPath" -ForegroundColor Red
    exit 1
}

# Create a temp folder to extract into
$TempDir = Join-Path $env:TEMP "neuro_ai_update_$(Get-Random)"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

Write-Host "Extracting ZIP to temp folder..." -ForegroundColor Yellow
Expand-Archive -Path $ZipPath -DestinationPath $TempDir -Force

# The zip extracts to ai-ml-agent/ subfolder
$SourceFolder = Join-Path $TempDir "ai-ml-agent"

if (-not (Test-Path $SourceFolder)) {
    Write-Host "ERROR: Could not find ai-ml-agent folder inside ZIP" -ForegroundColor Red
    Remove-Item -Recurse -Force $TempDir
    exit 1
}

# Copy all .py files recursively - SKIP neuro_env and __pycache__
$copied  = 0
$skipped = 0

Get-ChildItem -Path $SourceFolder -Recurse -File | ForEach-Object {
    $relativePath = $_.FullName.Substring($SourceFolder.Length + 1)

    # Skip environment and cache folders
    if ($relativePath -match "neuro_env|__pycache__|\.pyc$") {
        $skipped++
        return
    }

    $destPath = Join-Path $TargetFolder $relativePath
    $destDir  = Split-Path $destPath -Parent

    # Create directory if it doesn't exist
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    # Copy file
    Copy-Item -Path $_.FullName -Destination $destPath -Force
    Write-Host "  Updated: $relativePath" -ForegroundColor Green
    $copied++
}

# Cleanup temp folder
Remove-Item -Recurse -Force $TempDir

Write-Host ""
Write-Host "Done! Updated $copied files, skipped $skipped cache files." -ForegroundColor Cyan
Write-Host "Your neuro_env is untouched - no reinstall needed." -ForegroundColor Green
Write-Host ""
Write-Host "Start the app:" -ForegroundColor Yellow
Write-Host "  streamlit run ui/streamlit_app.py" -ForegroundColor White
