@echo off
REM Re-run ONLY the tortuosity stage at full resolution (no downsampling) from a Miniforge / conda prompt,
REM after run_rf_conda.bat has finished (it needs labels_connected.npy of each sample).
REM   run_tortuosity_full.bat                                   LLZTO only, all samples, data in D:\microscopy\Jehad\FIB data\segmented
REM   run_tortuosity_full.bat "D:\...\segmented" llzto --sample LCO70
REM   run_tortuosity_full.bat "D:\...\segmented" "lco llzto"     both phases
REM The previous tortuosity.json of each sample is kept as tortuosity_downsampled.json; phases not re-solved
REM are taken from it, the full-resolution result is also saved as tortuosity_<phase>_full.json, and the
REM report is refreshed.
setlocal
set DATA=%~1
if "%DATA%"=="" set DATA=D:\microscopy\Jehad\FIB data\segmented
set PHASES=%~2
if "%PHASES%"=="" set PHASES=llzto
shift
shift
set EXTRA=
:loop
if "%~1"=="" goto run
set EXTRA=%EXTRA% %1
shift
goto loop
:run
cd /d "%~dp0"
call conda activate cathode || exit /b 1
python make_rf_config.py "%DATA%" --out config\workstation_rf_tau_full.yaml --tortuosity-downsample 1 --tortuosity-phases %PHASES% || exit /b 1
python -m cathode_analysis.cli validate --config config\workstation_rf_tau_full.yaml || exit /b 1
for /d %%S in (..\results_rf\*) do if exist "%%S\tortuosity.json" if not exist "%%S\tortuosity_downsampled.json" copy /Y "%%S\tortuosity.json" "%%S\tortuosity_downsampled.json" >nul
python -m cathode_analysis.cli run --config config\workstation_rf_tau_full.yaml --stage tortuosity --force %EXTRA%
for /d %%S in (..\results_rf\*) do if exist "%%S\tortuosity.json" python merge_tortuosity.py "%%S" --phases %PHASES%
python -m cathode_analysis.cli run --config config\workstation_rf_tau_full.yaml --stage report --force
endlocal
