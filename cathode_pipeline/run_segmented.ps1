<#
Headless run on ALREADY SEGMENTED stacks (config\workstation_segmented.yaml).
Same switches as run_all.ps1:

  .\run_segmented.ps1                        # all samples in the config, all stages, resume
  .\run_segmented.ps1 -Force                 # recompute everything
  .\run_segmented.ps1 -Stage segment         # only import the masks (labels.npy + segment.json + qc\)
  .\run_segmented.ps1 -Sample LCO50 -DryRun  # list planned outputs, run nothing

If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\run_segmented.ps1
#>
param(
    [string]$Python = "python",
    [string]$Venv = ".venv",
    [string]$Stage = "all",
    [string[]]$Sample = @(),
    [switch]$Force,
    [switch]$DryRun,
    [switch]$SkipInstall
)
& (Join-Path $PSScriptRoot "run_all.ps1") -Config "config\workstation_segmented.yaml" -Python $Python -Venv $Venv `
    -Stage $Stage -Sample $Sample -Force:$Force -DryRun:$DryRun -SkipInstall:$SkipInstall
exit $LASTEXITCODE
