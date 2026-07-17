@echo off
REM HTC median_pixel training launcher
REM Edit only the SETTINGS section below.

REM Usually: D:\HTC_github
set PROJECT_ROOT=D:\HTC_github

REM YAML config file
set YAML_FILE=manifold_settings.yaml

REM A name for the training profile you are trying to run:
REM gpu_smoke, gpu_practical, gpu_paper, cpu_smoke, cpu_practical, etc
set TRAINING_PROFILE=gpu_paper

REM Wavelength range in nm
REM Full CSV1: 500 to 995
REM Example narrow range: 500 to 600
set WAVELENGTH_MIN=500
set WAVELENGTH_MAX=995

REM Choose one:
REM gpu or cpu
set ACCELERATOR=gpu

REM GPU/CPU device count.
REM Usually 1.
set DEVICES=1

REM Precision:
REM GTX 1080 stable: 32-true
REM Newer RTX GPUs can try: 16-mixed
set PRECISION=32-true

REM Batch size:
REM GTX 1080 stable: 512
REM Better GPUs: 1024, 2048, 4096, 8192, maybe higher
set BATCH_SIZE=512

REM Num workers:
REM Windows stable: 0
REM If stable on newer machine: 2, 4, 8
set NUM_WORKERS=0

REM Epoch size:
REM smoke: 2000
REM practical: leave blank or use 10369
REM paper-style: 10000000
set EPOCH_SIZE=10000000

REM Max epochs:
REM smoke: 1
REM practical/paper: 10
set MAX_EPOCHS=10

REM Data path replacement.
REM If YAML and machine both use Z:, keep both Z:/TIVITA_Cat
set PATH_PREFIX_FROM=Z:/TIVITA_Cat
set PATH_PREFIX_TO=Z:/TIVITA_Cat

REM Output folder name.
REM Usually match training profile.
REM gpu_smoke, gpu_practical, gpu_paper, cpu_smoke, cpu_practical, etc
set OUTPUT_NAME=median_pixel_yaml_gpu_paper

REM DO NOT EDIT BELOW

cd /d %PROJECT_ROOT%\htc

call HSI_ML\Scripts\activate.bat

set PATH_HTC_RESULTS=%PROJECT_ROOT%\htc_results
set HTC_ADD_NETWORK_ALTERNATIVES=false

python "%PROJECT_ROOT%\htc\scripts\newer_scripts\htc_median_pixel_from_yaml_official.py" ^
  --settings-yaml "%PROJECT_ROOT%\config\%YAML_FILE%" ^
  --output-dir "%PROJECT_ROOT%\outputs\%OUTPUT_NAME%" ^
  --wavelength-min %WAVELENGTH_MIN% ^
  --wavelength-max %WAVELENGTH_MAX% ^
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
  --results-dir "%PROJECT_ROOT%\htc_results" ^
  --run-training ^
  --test

pause