<#
    First-time setup on a new machine.
        powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
#>
[CmdletBinding()]
param(
    [switch]$WithDriveApi,
    [string]$GitRemote = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== meetingrec bootstrap ==" -ForegroundColor Cyan

# 1. interpreter
try { $ver = & py --version 2>&1 } catch {
    throw "Python launcher 'py' not found. Install Python from python.org (tick 'Add to PATH')."
}
Write-Host "  interpreter : $ver"

# 2. venv
if (-not (Test-Path ".venv")) {
    Write-Host "  creating venv..."
    & py -m venv .venv
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# 3. deps
$extras = if ($WithDriveApi) { "detect,driveapi,dev" } else { "detect,dev" }
Write-Host "  installing  : .[$extras]"
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -e ".[$extras]" --quiet

# 4. config
if (-not (Test-Path "config.toml")) {
    Copy-Item "config.example.toml" "config.toml"
    Write-Host "  config.toml : created from example" -ForegroundColor Green
} else {
    Write-Host "  config.toml : already exists, left alone"
}

New-Item -ItemType Directory -Force -Path "recordings" | Out-Null

# 5. git
if (-not (Test-Path ".git")) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        & git init -q
        & git add -A
        & git commit -qm "meetingrec scaffold: detector, storage, config"
        if ($GitRemote) {
            & git remote add origin $GitRemote
            Write-Host "  git remote  : $GitRemote (push with: git push -u origin master)"
        }
        Write-Host "  git         : initialized" -ForegroundColor Green
    }
}

# 6. smoke test
Write-Host "`n== checks ==" -ForegroundColor Cyan
& $py -m pytest -q
& $py -m meetingrec storage

Write-Host "`nNext: join a test meeting, then run" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe -m meetingrec list"
Write-Host "  .\.venv\Scripts\python.exe -m meetingrec watch"
