@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM HTC confusion matrix from one YAML path
REM Supervisor edits ONLY this line.
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

if not exist "%PROJECT_ROOT%\htc\HSI_ML\Scripts\activate.bat" (
  echo ERROR: Could not find virtual environment:
  echo %PROJECT_ROOT%\htc\HSI_ML\Scripts\activate.bat
  pause
  exit /b 2
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
copy "%~f0" "%RUN_CONFIG_DIR%\%~n0_used_%TS%.bat" >nul

cd /d "%PROJECT_ROOT%\htc"
call HSI_ML\Scripts\activate.bat

REM Read training_profile from YAML using findstr.
set "TRAINING_PROFILE="

for /f "tokens=2 delims=:" %%i in ('findstr /R /C:"^[ ]*training_profile[ ]*:" "%YAML_PATH_ABS%"') do set "TRAINING_PROFILE=%%i"

set "TRAINING_PROFILE=%TRAINING_PROFILE: =%"

if not defined TRAINING_PROFILE (
  echo ERROR: training_profile not found in YAML.
  echo Expected line like:
  echo   training_profile: gpu_practical
  pause
  exit /b 3
)

echo TRAINING_PROFILE=%TRAINING_PROFILE%
echo TRAINING_PROFILE=%TRAINING_PROFILE% >> "%LOG_FILE%"

REM Use profile-specific run file first.
set "RUN_DIR="
set "RUN_DIR_FILE=%RUN_CONFIG_DIR%\last_run_dir_%TRAINING_PROFILE%.txt"

if exist "%RUN_DIR_FILE%" (
  set /p RUN_DIR=<"%RUN_DIR_FILE%"
)

REM Reject corrupted or invalid RUN_DIR.
if defined RUN_DIR (
  if /I "%RUN_DIR%"=="ECHO ist ausgeschaltet (OFF)." (
    echo WARNING: Ignoring corrupted RUN_DIR file.
    set "RUN_DIR="
  )
)

if defined RUN_DIR (
  if not exist "%RUN_DIR%" (
    echo WARNING: Saved RUN_DIR does not exist. Ignoring:
    echo %RUN_DIR%
    set "RUN_DIR="
  )
)

REM If no valid saved run, search latest matching prediction file.
if not defined RUN_DIR (
  for /f "delims=" %%i in ('powershell -NoProfile -Command "$p=Get-ChildItem '%PATH_HTC_RESULTS%\training\median_pixel' -Recurse -Filter 'test_predictions.csv' | Where-Object { $_.FullName -like '*%TRAINING_PROFILE%*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if($p){$p.Directory.Parent.FullName}"') do set "RUN_DIR=%%i"
)

if not defined RUN_DIR (
  echo ERROR: Could not find a valid run folder for %TRAINING_PROFILE%.
  echo Expected file:
  echo %RUN_DIR_FILE%
  echo Or matching test_predictions.csv under:
  echo %PATH_HTC_RESULTS%\training\median_pixel
  pause
  exit /b 4
)

if not exist "%RUN_DIR%" (
  echo ERROR: RUN_DIR does not exist:
  echo %RUN_DIR%
  pause
  exit /b 5
)

echo RUN_DIR=%RUN_DIR%
echo RUN_DIR=%RUN_DIR% >> "%LOG_FILE%"

REM Save corrected run-dir files.
echo %RUN_DIR%> "%RUN_CONFIG_DIR%\last_run_dir.txt"
echo %RUN_DIR%> "%RUN_CONFIG_DIR%\last_run_dir_%TRAINING_PROFILE%.txt"

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