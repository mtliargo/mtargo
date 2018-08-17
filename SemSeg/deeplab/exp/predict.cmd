@echo off

set DATA_DIR=D:\Data
set PYTHONPATH=%CD%;..\slim;%PYTHONPATH%

:: Set up the working directories.
set CITYSCAPES_DIR=%DATA_DIR%\Cityscapes
set EXP_FOLDER=%DATA_DIR%\Exp\DeepLabCityscapes\out-of-the-shelf2
set TRAIN_LOGDIR=%EXP_FOLDER%\train\model.ckpt
set VIS_LOGDIR=%EXP_FOLDER%\predict

set CITYSCAPES_DATASET=%CITYSCAPES_DIR%\tfrecord

python deeplab\predict.py ^
    --logtostderr ^
    --vis_split=val ^
    --model_variant=xception_65 ^
    --atrous_rates=6 ^
    --atrous_rates=12 ^
    --atrous_rates=18 ^
    --output_stride=16 ^
    --decoder_output_stride=4 ^
    --vis_crop_size=1025 ^
    --vis_crop_size=2049 ^
    --dataset=cityscapes ^
    --colormap_type=cityscapes ^
    --checkpoint_dir=%TRAIN_LOGDIR% ^
    --vis_logdir=%VIS_LOGDIR% ^
    --dataset_dir=%CITYSCAPES_DATASET% ^
  
