<#
Edit on Windows, run in WSL.

Every command runs inside the WSL distro (default Ubuntu-24.04) from this
project folder, with ROS 2 Jazzy, ~/venvs/uav-sitl and the project
PYTHONPATH loaded (scripts/setup_wsl_env.sh). Files are shared through
/mnt/c, so saving a file here is all it takes; there is nothing to sync.

  .\wsl.ps1 check                      toolchain health check
  .\wsl.ps1 test [pytest args]         pytest in WSL (default: tests)
  .\wsl.ps1 run "<bash command>"       any command, e.g.
        .\wsl.ps1 run "python experiments/milestone5/smoke_test_m5_env.py"
  .\wsl.ps1 run -Log <name> "<cmd>"    same, also saved to logs\<time>_<name>.log
  .\wsl.ps1 build                      colcon build ROS package uav_autonomy
  .\wsl.ps1 setup                      ROS package + PX4 build + LiDAR airframe
  .\wsl.ps1 sitl [-World <w>] [-Gui]   PX4 SITL + Gazebo (Ctrl+C stops);
                                       log in logs\sitl_<w>.log
  .\wsl.ps1 stop                       stop PX4 / Gazebo / uav_autonomy nodes
  .\wsl.ps1 status                     running sims, CPU/RAM, recent logs
  .\wsl.ps1 shell                      interactive WSL shell with the env

Put a whole bash command line in one quoted string; it is passed to WSL
through an environment variable, so quotes inside it survive.
Set $env:UAV_WSL_DISTRO to use a different distro.
#>
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Distro = if ($env:UAV_WSL_DISTRO) { $env:UAV_WSL_DISTRO } else { 'Ubuntu-24.04' }

$Verb = if ($args.Count -gt 0) { [string]$args[0] } else { 'help' }
$Rest = @()
if ($args.Count -gt 1) { $Rest = @($args[1..($args.Count - 1)] | ForEach-Object { [string]$_ }) }

function Pop-Option([string]$Name, [switch]$Flag) {
    # Removes "-Name value" (or just "-Name" with -Flag) from $script:Rest.
    $i = -1
    for ($j = 0; $j -lt $script:Rest.Count; $j++) { if ($script:Rest[$j] -ieq "-$Name") { $i = $j; break } }
    if ($i -lt 0) { if ($Flag) { return $false } else { return $null } }
    $take = if ($Flag) { 1 } else { 2 }
    if (-not $Flag -and $i + 1 -ge $script:Rest.Count) { throw "-$Name needs a value" }
    $value = if ($Flag) { $true } else { $script:Rest[$i + 1] }
    $keep = @()
    for ($j = 0; $j -lt $script:Rest.Count; $j++) { if ($j -lt $i -or $j -ge $i + $take) { $keep += $script:Rest[$j] } }
    $script:Rest = $keep
    return $value
}

function Invoke-InWsl([string]$Command, [string]$LogFile) {
    # Call as a statement (never capture the result) so wsl.exe keeps the
    # console: live output, Ctrl+C and interactive programs all work.
    # The exit code is left in $LASTEXITCODE.
    # This file is CRLF on disk, so multi-line here-strings carry `r; bash must not see it.
    $env:UAV_CMD = $Command -replace "`r", ''
    $env:UAV_LOG = $LogFile
    $keep = @(($env:WSLENV -split ':') | Where-Object { $_ -and $_ -notmatch '^UAV_(CMD|LOG)' })
    $env:WSLENV = (@($keep) + 'UAV_CMD/u' + 'UAV_LOG/u') -join ':'
    try {
        & wsl.exe -d $Distro --cd $Root --exec bash scripts/wsl_exec.sh
    } finally {
        Remove-Item Env:UAV_CMD, Env:UAV_LOG -ErrorAction SilentlyContinue
    }
}

function New-LogPath([string]$Name) {
    $safe = ($Name -replace '[^A-Za-z0-9_.-]', '_')
    return "logs/{0}_{1}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $safe
}

function ConvertTo-BashWords([string[]]$Words) {
    # Quote each argument for bash so "-k 'not slow'" stays one argument.
    ($Words | ForEach-Object {
        if ($_ -match '^[A-Za-z0-9_./:=@%+,-]+$') { $_ } else { "'" + ($_ -replace "'", "'\''") + "'" }
    }) -join ' '
}

switch ($Verb.ToLower()) {
    'check' {
        Invoke-InWsl 'bash scripts/wsl_check.sh'; exit $LASTEXITCODE
    }
    'test' {
        $target = if ($Rest.Count) { ConvertTo-BashWords $Rest } else { 'tests' }
        # ROS pytest plugins (launch_testing etc.) are on the path once ROS is
        # sourced; keep them out of the project's unit tests.
        Invoke-InWsl "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q $target"; exit $LASTEXITCODE
    }
    'run' {
        $logName = Pop-Option 'Log'
        if (-not $Rest.Count) { throw 'usage: .\wsl.ps1 run [-Log name] "<bash command>"' }
        $log = if ($logName) { New-LogPath $logName } else { '' }
        Invoke-InWsl ($Rest -join ' ') $log; exit $LASTEXITCODE
    }
    'build' {
        Invoke-InWsl 'bash scripts/setup_wsl_sitl.sh --ros-only'; exit $LASTEXITCODE
    }
    'setup' {
        Invoke-InWsl 'bash scripts/setup_wsl_sitl.sh' (New-LogPath 'setup'); exit $LASTEXITCODE
    }
    'sitl' {
        $world = Pop-Option 'World'
        if (-not $world) { $world = 'default' }
        $headless = if (Pop-Option 'Gui' -Flag) { '0' } else { '1' }
        if ($world -match '^m5_obstacle_world_\d$' -and -not (Test-Path (Join-Path $Root "worlds\$world.sdf"))) {
            Invoke-InWsl 'python scripts/generate_m5_sitl_worlds.py'
        }
        $log = "logs/sitl_$world.log"
        Write-Host "PX4 SITL + Gazebo, world=$world headless=$headless; Ctrl+C to stop. Log: $log"
        Invoke-InWsl "HEADLESS=$headless PX4_GZ_WORLD=$world bash scripts/run_px4_sitl.sh" $log; exit $LASTEXITCODE
    }
    'stop' {
        Invoke-InWsl 'bash scripts/wsl_procs.sh stop'; exit $LASTEXITCODE
    }
    'status' {
        Invoke-InWsl 'bash scripts/wsl_procs.sh status'; exit $LASTEXITCODE
    }
    'shell' {
        & wsl.exe -d $Distro --cd $Root --exec bash --rcfile scripts/wsl_shellrc.sh -i
        exit $LASTEXITCODE
    }
    default {
        Get-Content -LiteralPath $PSCommandPath -TotalCount 27 | Select-Object -Skip 1
        if ($Verb -notin @('help', '-h', '--help', '/?')) { Write-Host "Unknown command: $Verb"; exit 2 }
    }
}
