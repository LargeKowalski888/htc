@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM HTC training from one YAML path

REM Full path to scenario YAML file.
set "YAML_PATH=D:\HTC_github\scenario_3_ManUSPECTRAL\config\manifold_settings.yaml"

REM DO NOT EDIT BELOW

REM Resolve YAML path and derive folders robustly.
for /f "delims=" %%i in ('powershell -NoProfile -Command "(Resolve-Path '%YAML_PATH%').Path"') do set "YAML_PATH_ABS=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "Split-Path -Parent (Split-Path -Parent '%YAML_PATH_ABS%')"') do set "SCENARIO_DIR=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "Split-Path -Parent '%SCENARIO_DIR%'"') do set "PROJECT_ROOT=%%i"

set "OUTPUT_DIR=%SCENARIO_DIR%\output"
set "RUN_CONFIG_DIR=%SCENARIO_DIR%\run_config"
set "PATH_HTC_RESULTS=%PROJECT_ROOT%\htc_results"
set "HTC_ADD_NETWORK_ALTERNATIVES=false"

if not exist "%PROJECT_ROOT%\htc\HSI_ML\Scripts\activate.bat" (
  echo ERROR: Could not find virtual environment at "%PROJECT_ROOT%\htc\HSI_ML\Scripts\activate.bat"
  echo PROJECT_ROOT was detected as: %PROJECT_ROOT%
  pause
  exit /b 2
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%RUN_CONFIG_DIR%" mkdir "%RUN_CONFIG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
copy "%~f0" "%RUN_CONFIG_DIR%\%~n0_used_%TS%.bat" >nul

cd /d "%PROJECT_ROOT%\htc"
call HSI_ML\Scripts\activate.bat

REM Read training_profile from YAML without requiring PyYAML in the batch file.
for /f "delims=" %%i in ('powershell -NoProfile -Command "$y=Get-Content -Raw '%YAML_PATH_ABS%'; if($y -match '(?m)^\s*training_profile\s*:\s*[''\"'']?([^''\"''\r\n#]+)') {$matches[1].Trim()} else {'gpu_smoke'}"') do set "TRAINING_PROFILE=%%i"

echo YAML_PATH=%YAML_PATH_ABS%
echo PROJECT_ROOT=%PROJECT_ROOT%
echo SCENARIO_DIR=%SCENARIO_DIR%
echo OUTPUT_DIR=%OUTPUT_DIR%
echo TRAINING_PROFILE=%TRAINING_PROFILE%

python "%PROJECT_ROOT%\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py" ^
  --settings-yaml "%YAML_PATH_ABS%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --path-prefix-from "Z:/TIVITA_Cat" ^
  --path-prefix-to "Z:/TIVITA_Cat" ^
  --require-standardized-or-001 ^
  --label-mode official_atlas ^
  --label-mapping generated_dict ^
  --build-htc-adapter ^
  --adapter-dataset-name "Cat_HTC_Adapter" ^
  --annotation-name "semantic#primary" ^
  --training-profile gpu_smoke ^
  --accelerator gpu ^
  --devices 1 ^
  --precision 32-true ^
  --batch-size 512 ^
  --num-workers 0 ^
  --epoch-size 2000 ^
  --max-epochs 1 ^
  --wavelength-min 500 ^
  --wavelength-max 995 ^
  --results-dir "%PATH_HTC_RESULTS%" ^
  --run-training ^
  --test

set "TRAIN_EXIT=%ERRORLEVEL%"
echo Training exit code: %TRAIN_EXIT%

if "%TRAIN_EXIT%"=="0" (
  for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-ChildItem '%PATH_HTC_RESULTS%\training\median_pixel' -Recurse -Filter 'test_predictions.csv' | Where-Object { $_.FullName -like '*%TRAINING_PROFILE%*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.Directory.Parent.FullName }"') do set "RUN_DIR=%%i"
  if defined RUN_DIR (
    echo %RUN_DIR%> "%RUN_CONFIG_DIR%\last_run_dir.txt"
    echo %RUN_DIR%> "%RUN_CONFIG_DIR%\last_run_dir_%TRAINING_PROFILE%.txt"
    echo Saved run folder: %RUN_DIR%
  ) else (
    echo WARNING: Training succeeded but no test_predictions.csv was found for %TRAINING_PROFILE%.
  )
) else (
  echo Training failed. last_run_dir files were not updated.
)

pause
exit /b %TRAIN_EXIT%
