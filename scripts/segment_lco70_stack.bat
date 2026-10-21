@echo off
REM Segment the LCO70 FIB-SEM stack (already cropped, HistBal-Gauss-Kuwahara processed) with the combined random forest.
REM Run from the repository folder (the one containing fib_rf_seg\).  Resumes if interrupted.
REM  rf_all.joblib : trained on 4 LCO60 + 3 LCO70 slices with small-particle-aware seeds; keeps the many small LLZTO
REM                  particles of LCO70 (the LCO60-only model labelled half of them LCO).
REM  --smooth 1.0  : the Kuwahara-processed slices still carry a blocky texture; a light blur gives cleaner edges.
REM  --norm phase  : per-slice grey normalisation anchored on the pore and LLZTO levels.
set INPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\processed_LCO70
set OUTPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\processed_LCO70_segmented

python -m fib_rf_seg predict --model models\rf_all.joblib --input "%INPUT%" --out "%OUTPUT%" ^
    --smooth 1.0 --norm phase --pore-level auto --min-size 10 --per-phase --stack
pause
