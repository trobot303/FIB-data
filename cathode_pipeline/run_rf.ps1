# Run the 3D analysis on the random-forest segmented volumes (config\workstation_rf.yaml).
#   .\run_rf.ps1                  # all samples, all stages, resumes where outputs exist
#   .\run_rf.ps1 -Force           # recompute everything
#   .\run_rf.ps1 -Sample LCO70    # one sample
# If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\run_rf.ps1
param([string]$Stage = "all", [string[]]$Sample = @(), [switch]$Force, [string]$Python = "python")
$args2 = @("-Config", "config\workstation_rf.yaml", "-Stage", $Stage, "-Python", $Python)
foreach ($s in $Sample) { $args2 += @("-Sample", $s) }
if ($Force) { $args2 += "-Force" }
& "$PSScriptRoot\run_all.ps1" @args2
exit $LASTEXITCODE
