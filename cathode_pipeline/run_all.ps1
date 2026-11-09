<#
Headless end-to-end run on a Windows workstation (PowerShell 5.1 or 7).

  .\run_all.ps1                                   # all samples, all stages, resume where possible
  .\run_all.ps1 -Force                            # recompute everything
  .\run_all.ps1 -Stage segment -Sample LCO50      # one stage / one sample
  .\run_all.ps1 -Config config\default.yaml       # another config
  .\run_all.ps1 -Python "C:\Python311\python.exe" # choose the interpreter used to create the venv

If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#>
param(
    [string]$Config = "config\workstation.yaml",
    [string]$Python = "python",
    [string]$Venv = ".venv",
    [string]$Stage = "all",
    [string[]]$Sample = @(),
    [switch]$Force,
    [switch]$DryRun,
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "creating virtual environment $Venv with $Python"
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
if (-not $SkipInstall) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

& $venvPython -m cathode_analysis.cli validate --config $Config
if ($LASTEXITCODE -ne 0) { throw "config validation failed" }

$runArgs = @("run", "--config", $Config, "--stage", $Stage)
foreach ($s in $Sample) { $runArgs += @("--sample", $s) }
if ($Force)  { $runArgs += "--force" }
if ($DryRun) { $runArgs += "--dry-run" }

& $venvPython -m cathode_analysis.cli @runArgs
exit $LASTEXITCODE
