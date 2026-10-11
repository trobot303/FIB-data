@echo off
REM Push the segmented volumes (one binary multipage TIFF per phase, 255 = phase) to GitHub so the
REM 3D analysis can run in the cloud session.  Run from the repository folder.
REM Cropping to a common volume is done in the cloud, so the full stacks are pushed.
set SEG=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\segmented
REM Folder of the LCO60_2 segmentation (leave as is if it does not exist yet; it is then skipped).
set SEG60_2=C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\processed_LCO60_2_segmented

for %%P in ("LCO50=%SEG%\processed_LCO50_segmented" "LCO60=%SEG%\processed_LCO60_segmented" "LCO70=%SEG%\processed_LCO70_segmented" "LCO60_2=%SEG60_2%") do (
  for /f "tokens=1,* delims==" %%A in (%%P) do (
    if exist "%%B\pore_stack.tif" (
      if not exist data\segmented\%%A mkdir data\segmented\%%A
      copy /Y "%%B\pore_stack.tif"  data\segmented\%%A\pore_stack.tif
      copy /Y "%%B\LCO_stack.tif"   data\segmented\%%A\LCO_stack.tif
      copy /Y "%%B\LLZTO_stack.tif" data\segmented\%%A\LLZTO_stack.tif
    ) else (
      echo skipped %%A: no pore_stack.tif in "%%B"
    )
  )
)
dir data\segmented /s
git add data\segmented
git commit -m "Add segmented phase stacks"
git push
pause
