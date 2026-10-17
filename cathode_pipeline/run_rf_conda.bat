@echo off
REM Run the 3D analysis on the random-forest segmented volumes from a Miniforge / conda prompt.
REM   run_rf_conda.bat                       data in D:\microscopy\Jehad\FIB data\segmented
REM   run_rf_conda.bat "E:\other\segmented"  another data folder
REM Extra arguments after the folder go to the pipeline, e.g.:  run_rf_conda.bat "D:\...\segmented" --sample LCO70 --force
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
call mamba env list | findstr /B /C:"cathode " >nul 2>&1 || call conda env list | findstr /B /C:"cathode " >nul 2>&1
if errorlevel 1 (
  echo creating conda environment "cathode" from environment.yml
  call mamba env create -f environment.yml || call conda env create -f environment.yml || exit /b 1
)
call conda activate cathode || exit /b 1
python make_rf_config.py "%DATA%" || exit /b 1
python -m cathode_analysis.cli validate --config config\workstation_rf.yaml || exit /b 1
python -m cathode_analysis.cli run --config config\workstation_rf.yaml --stage all %EXTRA%
endlocal
