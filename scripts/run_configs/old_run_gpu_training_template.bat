@echo off
setlocal EnableExtensions EnableDelayedExpansion


REM SETTINGS TO EDIT
REM Full project root path
set "PROJECT_ROOT=D:\HTC_github"

REM Full scenario folder path. Example: D:\HTC_github\scenario_1
set "SCENARIO_DIR=D:\HTC_github\scenario_3_ManUSPECTRAL"

REM Full YAML file path inside the scenario config folder
set "YAML_PATH=D:\HTC_github\scenario_3_ManUSPECTRAL\config\manifold_settings.yaml"

REM Output folder for this scenario. Usually keep this under the scenario folder.
set "OUTPUT_DIR=D:\HTC_github\scenario_3_ManUSPECTRAL\output"

REM The level of complexity you want the confusion matrix to be. Examples: gpu_smoke, gpu_practical, gpu_paper, cpu_smoke, cpu_practical
set "TRAINING_PROFILE=gpu_smoke"

REM Full range: 500 to 995.
set "WAVELENGTH_MIN=500"
set "WAVELENGTH_MAX=995"


REM MACHINE LEARNING SETTINGS TO EDIT
REM Choose one: gpu or cpu
set "ACCELERATOR=gpu"

REM Usually 1
set "DEVICES=1"

REM GTX 1080 stable: 32-true. Newer RTX GPUs can try: 16-mixed
set "PRECISION=32-true"

REM GTX 1080 stable: 512. Newer stronger GPUs can try: 1024, 2048, 4096, 8192, 20000
set "BATCH_SIZE=512"

REM Windows stable: 0. Stronger/stable workstation can try: 2, 4, 8
set "NUM_WORKERS=0"

REM smoke: 2000, practical: 10369 or leave profile default, paper: 10000000
set "EPOCH_SIZE=2000"

REM smoke: 1, practical/paper: 10
set "MAX_EPOCHS=1"

REM Path replacement used for raw data paths in YAML
set "PATH_PREFIX_FROM=Z:/TIVITA_Cat"
set "PATH_PREFIX_TO=Z:/TIVITA_Cat"


REM DO NOT EDIT BELOW
REM Full list of commands that are activated
set "RUN_CONFIG_DIR=%SCENARIO_DIR%\run_config"
set "PATH_HTC_RESULTS=%PROJECT_ROOT%\htc_results"
set "HTC_ADD_NETWORK_ALTERNATIVES=false"

if not exist "%SCENARIO_DIR%" mkdir "%SCENARIO_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%RUN_CONFIG_DIR%" mkdir "%RUN_CONFIG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
copy "%~f0" "%RUN_CONFIG_DIR%\%~n0_used_%TS%.bat" >nul

cd /d "%PROJECT_ROOT%\htc"
call HSI_ML\Scripts\activate.bat

python "%PROJECT_ROOT%\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py" ^
  --settings-yaml "%YAML_PATH%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --path-prefix-from "%PATH_PREFIX_FROM%" ^
  --path-prefix-to "%PATH_PREFIX_TO%" ^
  --require-standardized-or-001 ^
  --label-mode official_atlas ^
  --label-mapping generated_dict ^
  --build-htc-adapter ^
  --adapter-dataset-name "Cat_HTC_Adapter" ^
  --annotation-name "semantic#primary" ^
  --training-profile %TRAINING_PROFILE% ^
  --accelerator %ACCELERATOR% ^
  --devices %DEVICES% ^
  --precision %PRECISION% ^
  --batch-size %BATCH_SIZE% ^
  --num-workers %NUM_WORKERS% ^
  --epoch-size %EPOCH_SIZE% ^
  --max-epochs %MAX_EPOCHS% ^
  --wavelength-min %WAVELENGTH_MIN% ^
  --wavelength-max %WAVELENGTH_MAX% ^
  --results-dir "%PATH_HTC_RESULTS%" ^
  --run-training ^
  --test

set "TRAIN_EXIT=%ERRORLEVEL%"

REM Save the newest run folder for this profile so the confusion matrix script can reuse it.
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-ChildItem '%PATH_HTC_RESULTS%\training\median_pixel' -Recurse -Filter 'test_predictions.csv' | Where-Object { $_.FullName -like '*%TRAINING_PROFILE%*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.Directory.Parent.FullName }"') do set "RUN_DIR=%%i"

if defined RUN_DIR (
  echo %RUN_DIR%> "%RUN_CONFIG_DIR%\last_run_dir_%TRAINING_PROFILE%.txt"
  echo Latest run folder saved to: "%RUN_CONFIG_DIR%\last_run_dir_%TRAINING_PROFILE%.txt"
  echo %RUN_DIR%
) else (
  echo WARNING: No test_predictions.csv found for %TRAINING_PROFILE% yet.
)

echo Training exit code: %TRAIN_EXIT%
pause
exit /b %TRAIN_EXIT%