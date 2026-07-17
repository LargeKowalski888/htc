@echo off
cd /d D:\HTC_github\htc
call HSI_ML\Scripts\activate.bat

REM Choose one: gpu_smoke, gpu_practical, gpu_paper
set PROFILE=gpu_paper

REM Matching output folder suffix: gpu_smoke, gpu_practical, gpu_paper
set OUTPUT_NAME=median_pixel_yaml_gpu_paper

for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-ChildItem 'D:\HTC_github\htc_results\training\median_pixel' -Recurse -Filter 'test_predictions.csv' | Where-Object { $_.FullName -like '*%PROFILE%*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.Directory.Parent.FullName }"') do set RUN_DIR=%%i

echo Latest run folder:
echo %RUN_DIR%

python "D:\HTC_github\htc\scripts\newer_scripts\show_confusion_matrix_from_htc_run.py" ^
  --run-dir "%RUN_DIR%" ^
  --label-mapping "D:\HTC_github\outputs\%OUTPUT_NAME%\label_mapping.json" ^
  --output-png "D:\HTC_github\outputs\%OUTPUT_NAME%\confusion_matrix_%PROFILE%.png" ^
  --output-csv "D:\HTC_github\outputs\%OUTPUT_NAME%\confusion_matrix_%PROFILE%.csv" ^
  --normalize ^
  --list-candidates

pause