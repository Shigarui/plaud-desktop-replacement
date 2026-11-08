<#
    Put a meetingrec console shortcut on the Desktop and in the Start Menu.

        powershell -ExecutionPolicy Bypass -File .\scripts\install_shortcut.ps1
        powershell -ExecutionPolicy Bypass -File .\scripts\install_shortcut.ps1 -Remove

    The shortcut runs pythonw.exe, so no console window appears — the browser
    opens on its own and the page's Quit button stops the server. Without that
    button there would be no way to stop a windowless process short of Task
    Manager.
#>
[CmdletBinding()]
param(
    [string]$Archive  = "G:\My Drive\Meeting Recordings",
    [string]$Profiles = ".\BIGGBY_Speaker_Profiles",
    [string]$Name     = "meetingrec console",
    [switch]$Remove,
    [switch]$NoStartMenu
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
$desktop   = [Environment]::GetFolderPath("Desktop")
$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$targets = @((Join-Path $desktop "$Name.lnk"))
if (-not $NoStartMenu) { $targets += (Join-Path $startMenu "$Name.lnk") }

if ($Remove) {
    $targets | ForEach-Object { Remove-Item $_ -ErrorAction SilentlyContinue }
    Write-Host "removed '$Name' shortcuts"
    exit 0
}

$pyw = Join-Path $proj ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "not found: $pyw  (run bootstrap.ps1 first)" }
$icon = Join-Path $proj "meetingrec.ico"

$shell = New-Object -ComObject WScript.Shell
foreach ($path in $targets) {
    $parent = Split-Path -Parent $path
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $lnk = $shell.CreateShortcut($path)
    $lnk.TargetPath       = $pyw
    $lnk.Arguments        = "scripts\console.py --archive `"$Archive`" --profiles `"$Profiles`""
    $lnk.WorkingDirectory = $proj
    $lnk.Description      = "Run meetingrec commands and label speakers"
    if (Test-Path $icon) { $lnk.IconLocation = "$icon,0" }
    $lnk.Save()
    Write-Host "created $path"
}

Write-Host ""
Write-Host "Double-click it. The browser opens by itself; use Quit in the page"
Write-Host "to stop the console (the recorder keeps running either way)."
