@echo off
REM Segment the LCO60 FIB-SEM stack into pore / LCO / LLZTO with the committed random forest.
REM Run from the repository folder (the one containing fib_rf_seg\).  Resumes if interrupted.
set INPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\A970_6040\processed_LCO60
set OUTPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\A970_6040\processed_LCO60_segmented

python -m fib_rf_seg predict --model models\rf_lco60.joblib --input "%INPUT%" --out "%OUTPUT%" ^
    --pore-level auto --min-size 20 --per-phase --stack
pause
