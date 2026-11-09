@echo off
REM Segment the LCO60 FIB-SEM stack into pore / LCO / LLZTO with the committed random forest.
REM Run from the repository folder (the one containing fib_rf_seg\).  Resumes if interrupted.
set INPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\A970_6040\processed_LCO60
set OUTPUT=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\A970_6040\processed_LCO60_segmented

REM Leave SMOOTH empty for denoised ("processed") slices; set it to --smooth 1.0 for raw, noisy slices.
set SMOOTH=

REM Pixel size of the slices in nm; the model is 50 nm and resamples other sizes (e.g. --pixel-size 100).
REM "auto" reads it from the TIFF metadata; set the number explicitly if the processed slices lost it.
set PIXEL=--pixel-size auto

python -m fib_rf_seg predict --model models\rf_lco60.joblib --input "%INPUT%" --out "%OUTPUT%" ^
    %SMOOTH% %PIXEL% --pore-level auto --min-size 20 --per-phase --stack
pause
