@echo off
REM Segment the LCO50 FIB-SEM stack into pore / LCO / LLZTO with the committed random forest.
REM Run from the repository folder (the one containing fib_rf_seg\).  Resumes if interrupted.
set INPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\processed_LCO50
set OUTPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\processed_LCO50_segmented

REM Leave SMOOTH empty for denoised ("processed") slices; set it to --smooth 1.0 for raw, noisy slices.
set SMOOTH=

python -m fib_rf_seg predict --model models\rf_lco60.joblib --input "%INPUT%" --out "%OUTPUT%" ^
    %SMOOTH% --pore-level auto --min-size 20 --per-phase --stack
pause
