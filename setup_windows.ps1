$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pythonLauncher) {
    & py -3.12 --version
    if ($LASTEXITCODE -ne 0) {
        throw 'Python 3.12 is required. Install 64-bit Python 3.12 and retry.'
    }
    $python = 'py'
    $pythonArgs = @('-3.12')
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw 'Python 3.12 is required. Install 64-bit Python 3.12 and retry.' }
    $version = & python --version 2>&1
    if ($version -notmatch '^Python 3\.12\.') { throw "Python 3.12 is required; found $version" }
    $python = 'python'
    $pythonArgs = @()
}

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $python @pythonArgs -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv.' }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip.' }
& $venvPython -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Failed to install CPU PyTorch.' }
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-windows.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install pinned Python dependencies.' }
& $venvPython -m pip install --no-deps -e $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the project in editable mode.' }

Write-Host 'Windows Python environment is ready. Activate it with .\.venv\Scripts\Activate.ps1'
