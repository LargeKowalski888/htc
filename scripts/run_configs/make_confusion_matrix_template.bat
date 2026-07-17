@echo on
setlocal EnableExtensions EnableDelayedExpansion

REM HTC confusion matrix from one YAML path

set "YAML_PATH=D:\HTC_github\scenario_3_ManUSPECTRAL\config\manifold_settings.yaml"

REM DO NOT EDIT BELOW

echo Starting confusion matrix batch...
echo YAML_PATH=%YAML_PATH%

if not exist "%YAML_PATH%" (
  echo ERROR: YAML file does not exist:
  echo %YAML_PATH%
  pause
  exit /b 1
)

for /f "delims=" %%i in ('powershell -NoProfile -Command "(Resolve-Path '%YAML_PATH%').Path"') do set "YAML_PATH_ABS=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "Split-Path -Parent (Split-Path -Parent '%YAML_PATH_ABS%')"') do set "SCENARIO_DIR=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "Split-Path -Parent '%SCENARIO_DIR%'"') do set "PROJECT_ROOT=%%i"

set "OUTPUT_DIR=%SCENARIO_DIR%\output"
set "RUN_CONFIG_DIR=%SCENARIO_DIR%\run_config"
set "PATH_HTC_RESULTS=%PROJECT_ROOT%\htc_results"
set "LOG_FILE=%RUN_CONFIG_DIR%\confusion_matrix_debug_log.txt"

if not exist "%RUN_CONFIG_DIR%" mkdir "%RUN_CONFIG_DIR%"

echo YAML_PATH_ABS=%YAML_PATH_ABS% > "%LOG_FILE%"
echo PROJECT_ROOT=%PROJECT_ROOT% >> "%LOG_FILE%"
echo SCENARIO_DIR=%SCENARIO_DIR% >> "%LOG_FILE%"
echo OUTPUT_DIR=%OUTPUT_DIR% >> "%LOG_FILE%"
echo RUN_CONFIG_DIR=%RUN_CONFIG_DIR% >> "%LOG_FILE%"

echo PROJECT_ROOT=%PROJECT_ROOT%
echo SCENARIO_DIR=%SCENARIO_DIR%
echo OUTPUT_DIR=%OUTPUT_DIR%
echo RUN_CONFIG_DIR=%RUN_CONFIG_DIR%
echo LOG_FILE=%LOG_FILE%

if not exist "%PROJECT_ROOT%\htc\HSI_ML\Scripts\activate.bat" (
  echo ERROR: Could not find virtual environment:
  echo %PROJECT_ROOT%\htc\HSI_ML\Scripts\activate.bat
  echo ERROR: Could not find virtual environment >> "%LOG_FILE%"
  pause
  exit /b 2
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
copy "%~f0" "%RUN_CONFIG_DIR%\%~n0_used_%TS%.bat" >nul

cd /d "%PROJECT_ROOT%\htc"
call HSI_ML\Scripts\activate.bat

REM Read training_profile from YAML. Default to gpu_smoke if not found.
set "TRAINING_PROFILE="

for /f "tokens=2 delims=:" %%i in ('findstr /R /C:"^[ ]*training_profile[ ]*:" "%YAML_PATH_ABS%"') do set "TRAINING_PROFILE=%%i"

set "TRAINING_PROFILE=%TRAINING_PROFILE: =%"

if not defined TRAINING_PROFILE (
  echo ERROR: training_profile not found in YAML.
  pause
  exit /b 1
)

echo TRAINING_PROFILE=%TRAINING_PROFILE%
echo TRAINING_PROFILE=%TRAINING_PROFILE% >> "%LOG_FILE%"

set "RUN_DIR="
set "RUN_DIR_FILE=%RUN_CONFIG_DIR%\last_run_dir_%TRAINING_PROFILE%.txt"

if exist "%RUN_DIR_FILE%" (
  set /p RUN_DIR=<"%RUN_DIR_FILE%"
)

if not defined RUN_DIR (
  if exist "%RUN_CONFIG_DIR%\last_run_dir.txt" (
    set /p RUN_DIR=<"%RUN_CONFIG_DIR%\last_run_dir.txt"
  )
)

if not defined RUN_DIR (
  for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-ChildItem '%PATH_HTC_RESULTS%\training\median_pixel' -Recurse -Filter 'test_predictions.csv' | Where-Object { $_.FullName -like '*%TRAINING_PROFILE%*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.Directory.Parent.FullName }"') do set "RUN_DIR=%%i"
)

echo RUN_DIR=%RUN_DIR%
echo RUN_DIR=%RUN_DIR% >> "%LOG_FILE%"

if not defined RUN_DIR (
  echo ERROR: Could not find a run folder for %TRAINING_PROFILE%.
  echo ERROR: Could not find RUN_DIR >> "%LOG_FILE%"
  pause
  exit /b 3
)

python "%PROJECT_ROOT%\htc\scripts\newer_scripts\show_confusion_matrix_from_htc_run.py" ^
  --run-dir "%RUN_DIR%" ^
  --label-mapping "%OUTPUT_DIR%\label_mapping.json" ^
  --output-png "%SCENARIO_DIR%\confusion_matrix_%TRAINING_PROFILE%.png" ^
  --output-csv "%SCENARIO_DIR%\confusion_matrix_%TRAINING_PROFILE%.csv" ^
  --normalize ^
  --list-candidates

set "CM_EXIT=%ERRORLEVEL%"
echo Confusion matrix exit code: %CM_EXIT%
echo Confusion matrix exit code: %CM_EXIT% >> "%LOG_FILE%"

pause
exit /b %CM_EXIT%