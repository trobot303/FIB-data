@echo off
REM Re-run ONLY the tortuosity stage at full resolution (no downsampling) from a Miniforge / conda prompt,
REM after run_rf_conda.bat has finished (it needs labels_connected.npy of each sample).
REM   run_tortuosity_full.bat                        data in D:\microscopy\Jehad\FIB data\segmented, all samples
REM   run_tortuosity_full.bat "D:\...\segmented" --sample LCO70
REM The previous tortuosity.json of each sample is kept as tortuosity_downsampled.json; the report is refreshed.
setlocal
set DATA=%~1
if "%DATA%"=="" set DATA=D:\microscopy\Jehad\FIB data\segmented
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
python make_rf_config.py "%DATA%" --out config\workstation_rf_tau_full.yaml --tortuosity-downsample 1 || exit /b 1
python -m cathode_analysis.cli validate --config config\workstation_rf_tau_full.yaml || exit /b 1
for /d %%S in (..\results_rf\*) do if exist "%%S\tortuosity.json" if not exist "%%S\tortuosity_downsampled.json" copy /Y "%%S\tortuosity.json" "%%S\tortuosity_downsampled.json" >nul
python -m cathode_analysis.cli run --config config\workstation_rf_tau_full.yaml --stage tortuosity --force %EXTRA%
python -m cathode_analysis.cli run --config config\workstation_rf_tau_full.yaml --stage report --force
endlocal
