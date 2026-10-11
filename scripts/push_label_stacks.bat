@echo off
REM Copy the compressed label stacks (labels_stack.tif, ~50 MB per sample) of the four segmented volumes
REM into the repository and push them to GitHub, so the 3D analysis can run in the cloud session.
REM Run from the repository folder.  Uses the CROPPED folders for LCO70 and LCO50 (common volume with LCO60_2).
set SEG=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\segmented

if not exist data\segmented mkdir data\segmented
for %%S in (LCO60 LCO50 LCO60_2 LCO70) do if not exist data\segmented\%%S mkdir data\segmented\%%S

copy /Y "%SEG%\processed_LCO60_segmented\labels_stack.tif"          data\segmented\LCO60\labels_stack.tif
copy /Y "%SEG%\processed_LCO50_segmented_cropped\labels_stack.tif"  data\segmented\LCO50\labels_stack.tif
copy /Y "%SEG%\processed_LCO60_2_segmented\labels_stack.tif"        data\segmented\LCO60_2\labels_stack.tif
copy /Y "%SEG%\processed_LCO70_segmented_cropped\labels_stack.tif"  data\segmented\LCO70\labels_stack.tif
copy /Y "%SEG%\processed_LCO70_segmented_cropped\crop_info.json"    data\segmented\LCO70\crop_info.json
copy /Y "%SEG%\processed_LCO50_segmented_cropped\crop_info.json"    data\segmented\LCO50\crop_info.json

dir data\segmented /s
git add data\segmented
git commit -m "Add compressed label stacks of the four segmented volumes"
git push
pause
