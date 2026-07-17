@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM SETTINGS TO EDIT
REM ==========================

REM Full project root path
set "PROJECT_ROOT=D:\HTC_github"

REM Full scenario folder path. Example: D:\HTC_github\scenario_1
set "SCENARIO_DIR=D:\HTC_github\scenario_3_ManUSPECTRAL"

REM Must match the training profile used in the training batch file
REM Example: gpu_smoke, gpu_practical, gpu_paper
set "TRAINING_PROFILE=gpu_smoke"

REM Output folder used by the training run, usually scenario\output
set "OUTPUT_DIR=D:\HTC_github\scenario_3_ManUSPECTRAL"

REM DO NOT EDIT BELOW
REM ==========================

set "RUN_CONFIG_DIR=%SCENARIO_DIR%\run_config"
set "PATH_HTC_RESULTS=%PROJECT_ROOT%\htc_results"

if not exist "%SCENARIO_DIR%" mkdir "%SCENARIO_DIR%"
if not exist "%RUN_CONFIG_DIR%" mkdir "%RUN_CONFIG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
copy "%~f0" "%RUN_CONFIG_DIR%\%~n0_used_%TS%.bat" >nul

cd /d "%PROJECT_ROOT%\htc"
call HSI_ML\Scripts\activate.bat

REM Prefer the run folder saved by the training batch file.
set "RUN_DIR_FILE=%RUN_CONFIG_DIR%\last_run_dir_%TRAINING_PROFILE%.txt"
if exist "%RUN_DIR_FILE%" (
  set /p RUN_DIR=<"%RUN_DIR_FILE%"
) else (
  for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-ChildItem '%PATH_HTC_RESULTS%\training\median_pixel' -Recurse -Filter 'test_predictions.csv' | Where-Object { $_.FullName -like '*%TRAINING_PROFILE%*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.Directory.Parent.FullName }"') do set "RUN_DIR=%%i"
)

echo Using run folder:
echo %RUN_DIR%

python "%PROJECT_ROOT%\htc\scripts\newer_scripts\show_confusion_matrix_from_htc_run.py" ^
  --run-dir "%RUN_DIR%" ^
  --label-mapping "%OUTPUT_DIR%\output\label_mapping.json" ^
  --output-png "%SCENARIO_DIR%\confusion_matrix_%TRAINING_PROFILE%.png" ^
  --output-csv "%SCENARIO_DIR%\confusion_matrix_%TRAINING_PROFILE%.csv" ^
  --normalize ^
  --list-candidates

pause
