<#
  Export the assets of a Unity game (or of one bundle or asset file) with AssetRipper, run headless on a free local port.

    .\Export-UnityAssets.ps1 -Source "E:\Games\Some Game" -OutFolder "D:\assets\some-game"
    .\Export-UnityAssets.ps1 -Source "E:\Games\Some Game\Some Game_Data\StreamingAssets" -OutFolder D:\assets\streaming
    .\Export-UnityAssets.ps1 -Source "E:\Games\X\a.bundle" -OutFolder D:\assets\a -Mode UnityProject

  -Mode PrimaryContent (default): the assets as ordinary files (png/jpg images, audio, text, models, json for the rest).
  -Mode UnityProject: a Unity project folder (also the scripts, settings and the scenes), much bigger.

  The export folder must not exist or must be empty (AssetRipper writes into it). The program makes the folder itself.
  It prints a progress line about every 20 seconds, so run it in the background and read its output as it goes.
  Windows PowerShell 5.1 and PowerShell 7 both run it. AssetRipper listens on 127.0.0.1 only and is stopped when this ends.
#>
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$OutFolder,
    [ValidateSet('PrimaryContent', 'UnityProject')][string]$Mode = 'PrimaryContent',
    [string]$AssetRipper = (Join-Path $env:USERPROFILE 'tools\AssetRipper-2.0.0\AssetRipper.GUI.Free.exe'),
    [int]$TimeoutMinutes = 240,
    [int]$ProgressSeconds = 20
)
$ErrorActionPreference = 'Stop'

function Say([string]$text) { Write-Output ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $text) }
function MB([double]$bytes) { '{0:N1} MB' -f ($bytes / 1MB) }

# ---- checks before anything starts
if (-not (Test-Path -LiteralPath $AssetRipper)) { throw "AssetRipper is not at $AssetRipper (download AssetRipper_win_x64.zip from github.com/AssetRipper/AssetRipper/releases and unpack it there)." }
if (-not (Test-Path -LiteralPath $Source)) { throw "The source $Source does not exist." }
$Source = (Resolve-Path -LiteralPath $Source).ProviderPath
$isFolder = (Get-Item -LiteralPath $Source).PSIsContainer
$OutFolder = [IO.Path]::GetFullPath($OutFolder)
if (Test-Path -LiteralPath $OutFolder) {
    if (-not (Get-Item -LiteralPath $OutFolder).PSIsContainer) { throw "$OutFolder is a file, not a folder." }
    if (@(Get-ChildItem -LiteralPath $OutFolder -Force).Count -gt 0) { throw "$OutFolder is not empty. Give a new folder or an empty one: AssetRipper writes into it and may replace what is there." }
}
$parent = Split-Path -Parent $OutFolder
if (-not $parent -or -not (Test-Path -LiteralPath $parent)) { throw "The parent folder of $OutFolder does not exist. Make that folder first (only the parent), then run this again." }

$sourceBytes = if ($isFolder) { (Get-ChildItem -LiteralPath $Source -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum } else { (Get-Item -LiteralPath $Source).Length }
$drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($OutFolder).Substring(0, 1)) -ErrorAction SilentlyContinue
if ($drive -and $sourceBytes -gt 0 -and $drive.Free -lt ($sourceBytes * 1.2)) {
    throw ("Not enough free space on {0}: {1} free, the source is {2} and an export is usually about as big or bigger." -f $drive.Name, (MB $drive.Free), (MB $sourceBytes))
}
Say ("source: {0} ({1}, {2}); export mode: {3}; into: {4}" -f $Source, $(if ($isFolder) { 'folder' } else { 'file' }), (MB $sourceBytes), $Mode, $OutFolder)

# ---- a free port, and the log
$listener = New-Object System.Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
$listener.Start(); $port = $listener.LocalEndpoint.Port; $listener.Stop()
$logPath = Join-Path ([IO.Path]::GetTempPath()) ('assetripper-{0}.log' -f [guid]::NewGuid().ToString('N').Substring(0, 8))
$base = "http://127.0.0.1:$port"

Add-Type -AssemblyName System.Net.Http
$client = New-Object System.Net.Http.HttpClient
$client.Timeout = [TimeSpan]::FromMinutes($TimeoutMinutes)

function Post-Form([string]$path, [string]$value) {
    $data = New-Object 'System.Collections.Generic.Dictionary[string,string]'
    $data.Add('Path', $value)
    $content = New-Object System.Net.Http.FormUrlEncodedContent($data)
    return $client.PostAsync("$base$path", $content)
}
function Log-Lines { if (Test-Path -LiteralPath $logPath) { try { return @(Get-Content -LiteralPath $logPath -ErrorAction Stop) } catch { return @() } } else { return @() } }

$started = Get-Date
$proc = Start-Process -FilePath $AssetRipper -ArgumentList @('--headless', '--port', "$port", '--log-path', $logPath) -WindowStyle Hidden -PassThru -WorkingDirectory ([IO.Path]::GetTempPath())
try {
    # ---- wait for the web server
    $up = $false
    for ($i = 0; $i -lt 120 -and -not $up; $i++) {
        if ($proc.HasExited) { throw "AssetRipper stopped by itself (exit code $($proc.ExitCode)). Its log: $logPath" }
        try { $null = $client.GetAsync("$base/").Result; $up = $true } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $up) { throw 'AssetRipper did not start answering within a minute.' }
    Say "AssetRipper is up on 127.0.0.1:$port (log: $logPath)"

    # ---- load
    $null = (Post-Form $(if ($isFolder) { '/LoadFolder' } else { '/LoadFile' }) $Source).Result
    $loaded = $false
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    $nextTell = Get-Date
    while (-not $loaded -and (Get-Date) -lt $deadline) {
        $lines = Log-Lines
        if ($lines -match 'Finished processing assets') { $loaded = $true; break }
        if ($proc.HasExited) { throw "AssetRipper stopped while loading (exit code $($proc.ExitCode)). Its log: $logPath" }
        if ((Get-Date) -ge $nextTell) { Say ('loading... ' + $(if ($lines.Count) { $lines[-1] } else { 'starting' })); $nextTell = (Get-Date).AddSeconds($ProgressSeconds) }
        Start-Sleep -Milliseconds 700
    }
    if (-not $loaded) { throw "Loading did not finish within $TimeoutMinutes minutes. Its log: $logPath" }
    $backend = (Log-Lines | Where-Object { $_ -match 'scripting backend' } | Select-Object -First 1)
    if ($backend) { Say $backend }
    Say 'loaded; exporting (this call blocks until the export is finished)'

    # ---- export, with a progress line from the log while it runs
    $task = Post-Form "/Export/$Mode" $OutFolder
    $nextTell = Get-Date
    while (-not $task.IsCompleted) {
        Start-Sleep -Milliseconds 700
        if ($proc.HasExited) { throw "AssetRipper stopped during the export (exit code $($proc.ExitCode)). Its log: $logPath" }
        if ((Get-Date) -ge $nextTell) {
            $last = Log-Lines | Where-Object { $_ -match 'ExportProgress' -and $_ -notmatch 'Warning' } | Select-Object -Last 1
            Say $(if ($last) { $last } else { 'exporting...' })
            $nextTell = (Get-Date).AddSeconds($ProgressSeconds)
        }
    }
    if ($task.IsFaulted) { throw ('The export request failed: ' + $task.Exception.GetBaseException().Message) }

    # ---- summary
    $lines = Log-Lines
    $failed = @($lines | Where-Object { $_ -match 'Failed to export' }).Count
    $files = @(Get-ChildItem -LiteralPath $OutFolder -Recurse -File -ErrorAction SilentlyContinue)
    Say ('finished in {0:N1} minutes: {1} files, {2}, {3} items AssetRipper could not export (warnings, usually Unity built-in resources)' -f ((Get-Date) - $started).TotalMinutes, $files.Count, (MB (($files | Measure-Object Length -Sum).Sum)), $failed)
    Write-Output 'by type:'
    $files | Group-Object { if ($_.Extension) { $_.Extension.ToLower() } else { '(none)' } } | Sort-Object Count -Descending | Select-Object -First 12 |
        ForEach-Object { Write-Output ('  {0,7}  {1,-9} {2}' -f $_.Count, $_.Name, (MB (($_.Group | Measure-Object Length -Sum).Sum))) }
    Write-Output 'top-level folders:'
    Get-ChildItem -LiteralPath $OutFolder -Directory | ForEach-Object { Write-Output ('  ' + $_.Name) }
    Write-Output "log: $logPath"
}
finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    $client.Dispose()
}
