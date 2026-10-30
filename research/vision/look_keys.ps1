# Look marks for research/vision/gaze_calib.py (T-031 step 1b): the owner holds Right Ctrl while looking at the skull.
# Global (GetAsyncKeyState): works whatever window has the focus — the owner keeps working. Prints «ready», then «down» /
# «up» on every change; the WSL side stamps the lines on arrival (~20 ms polling). Right Ctrl alone does nothing in apps.
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File look_keys.ps1 <seconds>
Add-Type -Name K -Namespace W -MemberDefinition '[DllImport("user32.dll")] public static extern short GetAsyncKeyState(int vKey);'
$down = $false
$t_end = (Get-Date).AddSeconds([double]$args[0])
[Console]::Out.WriteLine("ready"); [Console]::Out.Flush()
while ((Get-Date) -lt $t_end) {
  $now = ([W.K]::GetAsyncKeyState(0xA3) -band 0x8000) -ne 0
  if ($now -ne $down) {
    $down = $now
    [Console]::Out.WriteLine($(if ($down) { "down" } else { "up" })); [Console]::Out.Flush()
  }
  Start-Sleep -Milliseconds 20
}
