@echo off
REM Crop the segmented LCO70 and LCO50 volumes to the size of the segmented LCO60_2 volume
REM (same voxel size, 50 nm isotropic, in all stacks).  Run from the repository folder.
REM Step 1 prints the sizes; step 2 writes <folder>_cropped next to each input.
REM The three folder names below (processed_*_segmented) must match the names inside the segmented folder.
REM  --z-anchor start  keeps the first Z slices (use center/end, or --z-start N, to pick another range).
REM  --xy-anchor center cuts the x/y window from the middle of each slice.
set BASE=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\segmented
set REF=%BASE%\processed_LCO60_2_segmented

python -m fib_rf_seg.cropvol report "%REF%" "%BASE%\processed_LCO70_segmented" "%BASE%\processed_LCO50_segmented"
python -m fib_rf_seg.cropvol crop "%BASE%\processed_LCO70_segmented" "%BASE%\processed_LCO50_segmented" --match "%REF%" --z-anchor start --stack
pause
