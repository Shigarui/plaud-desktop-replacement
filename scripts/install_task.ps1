<#
    Register meetingrec to start automatically at log on.

        powershell -ExecutionPolicy Bypass -File .\scripts\install_task.ps1
        powershell -ExecutionPolicy Bypass -File .\scripts\install_task.ps1 -Remove

    Why "only when the user is logged on", and never a Windows service:
    both of this app's signals live in the interactive session. Meeting
    detection reads live WASAPI capture sessions, and audio endpoints are
    per-session. A task configured to "run whether the user is logged on or
    not" gets no interactive token, sees no audio devices, and records
    nothing — silently, which is the worst kind of nothing.
#>
[CmdletBinding()]
param(
    [string]$TaskName = "meetingrec",
    [int]$DelaySeconds = 60,
    [switch]$Remove,
    [switch]$ShowConsole
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "removed scheduled task '$TaskName'"
    exit 0
}

# pythonw.exe has no console window. The log file is what you read instead.
$exe = if ($ShowConsole) { "$proj\.venv\Scripts\python.exe" }
       else { "$proj\.venv\Scripts\pythonw.exe" }
if (-not (Test-Path $exe)) { throw "not found: $exe  (run bootstrap.ps1 first)" }

$action  = New-ScheduledTaskAction -Execute $exe -Argument "-m meetingrec run" `
                                   -WorkingDirectory $proj

$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$trigger.Delay = "PT${DelaySeconds}S"   # let Drive and the audio stack settle

# No RestartCount. Task Scheduler treats any exit as a failure worth
# retrying, including a deliberate Stop-Process — so a restart policy means
# every manual stop spawns a replacement, and you end up with two recorders
# fighting over the audio devices. The app refuses to run twice now, but the
# right fix is not asking Windows to relaunch it in the first place.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

# Interactive, so HKCU and the audio session list are actually visible.
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
                                        -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
                       -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "registered '$TaskName'"
Write-Host "  runs    : $exe -m meetingrec run"
Write-Host "  trigger : at log on, after ${DelaySeconds}s"
Write-Host "  log     : $proj\logs\meetingrec.log"
Write-Host ""
Write-Host "start now : Start-ScheduledTask -TaskName $TaskName"
Write-Host "stop      : Stop-ScheduledTask  -TaskName $TaskName"
Write-Host "check     : Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "watch log : Get-Content '$proj\logs\meetingrec.log' -Wait -Tail 20"
Write-Host "remove    : .\scripts\install_task.ps1 -Remove"
