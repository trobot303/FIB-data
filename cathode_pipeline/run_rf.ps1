<#
Run the 3D analysis on the random-forest segmented volumes.

  .\run_rf.ps1                                        # data in D:\microscopy\Jehad\FIB data\segmented
  .\run_rf.ps1 -DataRoot "E:\other\segmented"         # another data folder
  .\run_rf.ps1 -Sample LCO70                          # one sample
  .\run_rf.ps1 -Force                                 # recompute everything
  .\run_rf.ps1 -SkipConfig                            # keep a hand-edited config\workstation_rf.yaml
  .\run_rf.ps1 -Python "C:\Python311\python.exe"      # interpreter for the venv (3.10 - 3.12)

If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\run_rf.ps1
#>
param(
    [string]$DataRoot = "D:\microscopy\Jehad\FIB data\segmented",
    [string]$Python = "python",
    [string]$Venv = ".venv",
    [string]$Stage = "all",
    [string[]]$Sample = @(),
    [string]$Device = "auto",
    [int]$TortuosityDownsample = 2,
    [int]$GrainsDownsample = 2,
    [switch]$Force,
    [switch]$SkipConfig,
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
if (-not $SkipConfig) {
    & $venvPython make_rf_config.py $DataRoot --device $Device --tortuosity-downsample $TortuosityDownsample --grains-downsample $GrainsDownsample
    if ($LASTEXITCODE -ne 0) { throw "config generation failed" }
}
$args2 = @("-Config", "config\workstation_rf.yaml", "-Stage", $Stage, "-Python", $Python, "-Venv", $Venv, "-SkipInstall")
foreach ($s in $Sample) { $args2 += @("-Sample", $s) }
if ($Force) { $args2 += "-Force" }
& "$PSScriptRoot\run_all.ps1" @args2
exit $LASTEXITCODE
