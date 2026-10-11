@echo off
REM Push the segmented volumes (one binary multipage TIFF per phase, 255 = phase) to GitHub so the
REM 3D analysis can run in the cloud session.  Run from the repository folder.
REM Edit the four folder names if they differ inside ...\FIB data\segmented.
set SEG=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\segmented

for %%P in ("LCO60=processed_LCO60_segmented" "LCO50=processed_LCO50_segmented_cropped" "LCO60_2=processed_LCO60_2_segmented" "LCO70=processed_LCO70_segmented_cropped") do (
  for /f "tokens=1,2 delims==" %%A in (%%P) do (
    if not exist data\segmented\%%A mkdir data\segmented\%%A
    copy /Y "%SEG%\%%B\pore_stack.tif"  data\segmented\%%A\pore_stack.tif
    copy /Y "%SEG%\%%B\LCO_stack.tif"   data\segmented\%%A\LCO_stack.tif
    copy /Y "%SEG%\%%B\LLZTO_stack.tif" data\segmented\%%A\LLZTO_stack.tif
    if exist "%SEG%\%%B\crop_info.json" copy /Y "%SEG%\%%B\crop_info.json" data\segmented\%%A\crop_info.json
  )
)
dir data\segmented /s
git add data\segmented
git commit -m "Add segmented phase stacks of the four volumes"
git push
pause
