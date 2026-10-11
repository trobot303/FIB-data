@echo off
REM Segment the LCO60_2 FIB-SEM stack (raw, uncropped slices) with the committed random forest.
REM Run from the repository folder (the one containing fib_rf_seg\).  Resumes if interrupted.
REM  --crop 400 1100 600 2300 : 700 rows x 1700 cols = 35 x 85 um at 50 nm/px, centred in the cathode of slice-4.
REM                             Order is Y0 Y1 X0 X1 (end exclusive).  Adjust if the sample moves along z.
REM  --smooth 1.5             : the slices are raw and noisy (noise std ~3400 vs ~460 in the processed LCO60 stack).
REM  --norm phase             : per-slice grey normalisation anchored on the pore and LLZTO levels; this stack has
REM                             a different contrast from LCO60 and its brightness drifts along z.
set INPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\LCO60_2
set OUTPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\LCO60_2_segmented

python -m fib_rf_seg predict --model models\rf_lco60.joblib --input "%INPUT%" --out "%OUTPUT%" ^
    --crop 400 1100 600 2300 --smooth 1.5 --norm phase --pore-level auto --min-size 20 --per-phase --stack
pause
