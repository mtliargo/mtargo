@echo off

set DATA_DIR=D:\Data
set PYTHONPATH=%CD%;..\slim;%PYTHONPATH%

:: Set up the working directories.
set CITYSCAPES_DIR=%DATA_DIR%\Cityscapes
set EXP_FOLDER=%DATA_DIR%\Exp\DeepLabCityscapes\out-of-the-shelf2
set INIT_FOLDER=%EXP_FOLDER%\init_models
set TRAIN_LOGDIR=%EXP_FOLDER%\trainlog\model.ckpt
set EVAL_LOGDIR=%EXP_FOLDER%\eval
set VIS_LOGDIR=%EXP_FOLDER%\vis
set EXPORT_DIR=%EXP_FOLDER%\export
mkdir %EVAL_LOGDIR%
mkdir %VIS_LOGDIR%
mkdir %EXPORT_DIR%

set CITYSCAPES_DATASET=%CITYSCAPES_DIR%\tfrecord


python deeplab/eval.py ^
  --dataset cityscapes ^
  --logtostderr ^
  --eval_split="val" ^
  --model_variant="xception_65" ^
  --atrous_rates=6 ^
  --atrous_rates=12 ^
  --atrous_rates=18 ^
  --output_stride=16 ^
  --decoder_output_stride=4 ^
  --eval_crop_size=1025 ^
  --eval_crop_size=2049 ^
  --checkpoint_dir="%TRAIN_LOGDIR%" ^
  --eval_logdir="%EVAL_LOGDIR%" ^
  --dataset_dir="%CITYSCAPES_DATASET%" ^
  
