param([string]$Root = '$VPS_ROOT\uhwr\stage\exp\FMD\xs\runs\fmdx_monitor\dispatch', [int]$PollSeconds = 600, [string]$Deadline = '2026-11-07T06:00:00Z')
# FMDX warm-monitor dispatcher (VPS 4090). Reason: <author> decision 3 Nov ~23:50Z (FMDX sampling first on the 4090), relayed by the lead.
# Runs inside the runner as ONE cmd (411_fxm_dispatcher). Loop: take the lowest-named *.cmd in ready\ (names carry the priority.json order), run it, move it to done\ or failed\.
# Poll for new jobs every $PollSeconds (checks EXIT / deadline every 30 s). Exit when: UTC now >= $Deadline (checked between jobs), or ready\ is empty and the file EXIT exists (written by the monitor
# when every PASS_XS warm arm + NOFM + PEINIT is sampled or declared not-assessed). It never touches the runner queue or any other track's files.
$ErrorActionPreference = 'Continue'
foreach ($d in 'ready','running','done','failed','logs') { New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null }
$log = Join-Path $Root 'dispatcher_log.txt'
function Log($m) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), $m) }
$dl = [DateTime]::Parse($Deadline, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::AdjustToUniversal -bor [Globalization.DateTimeStyles]::AssumeUniversal)
Log "dispatcher start pid=$PID root=$Root poll=$PollSeconds deadline=$($dl.ToString('yyyy-MM-ddTHH:mm:ssZ')) reason='<author> decision 3 Nov ~23:50Z'"
Get-ChildItem (Join-Path $Root 'running') -Filter *.cmd | ForEach-Object { Move-Item $_.FullName (Join-Path $Root 'ready') -Force; Log "requeued after restart: $($_.Name)" }
while ($true) {
  if ((Get-Date).ToUniversalTime() -ge $dl) { Log 'exit: deadline reached'; break }
  $job = Get-ChildItem (Join-Path $Root 'ready') -Filter *.cmd | Sort-Object Name | Select-Object -First 1
  if ($job) {
    $name = [IO.Path]::GetFileNameWithoutExtension($job.Name)
    $runPath = Join-Path (Join-Path $Root 'running') $job.Name
    Move-Item $job.FullName $runPath -Force
    $jl = Join-Path (Join-Path $Root 'logs') "$name.log"
    $t0 = (Get-Date).ToUniversalTime(); Log "job start $name"
    $p = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$runPath`"" -RedirectStandardOutput $jl -RedirectStandardError ($jl + '.err') -NoNewWindow -PassThru -Wait
    $code = $p.ExitCode; $secs = [int]((Get-Date).ToUniversalTime() - $t0).TotalSeconds
    $dest = if ($code -eq 0) { 'done' } else { 'failed' }
    Move-Item $runPath (Join-Path (Join-Path $Root $dest) $job.Name) -Force
    Log "job end $name exit=$code seconds=$secs -> $dest"
    continue
  }
  if (Test-Path (Join-Path $Root 'EXIT')) { Log 'exit: EXIT file and ready\ empty'; break }
  for ($i = 0; $i -lt [int]($PollSeconds / 30); $i++) {
    if ((Get-Date).ToUniversalTime() -ge $dl) { break }
    if ((Test-Path (Join-Path $Root 'EXIT')) -and -not (Get-ChildItem (Join-Path $Root 'ready') -Filter *.cmd)) { break }
    Start-Sleep -Seconds 30
  }
}
Log 'dispatcher end'
