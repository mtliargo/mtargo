@echo off

set DATA_DIR=D:\Data
set PYTHONPATH=%CD%;..\slim;%PYTHONPATH%

:: Set up the working directories.
set EXP_NAME=retrain-4V100
set CITYSCAPES_DIR=%DATA_DIR%\Cityscapes
set EXP_FOLDER=%DATA_DIR%\Exp\DeepLabCityscapes\%EXP_NAME%
set TRAIN_LOGDIR=%EXP_FOLDER%\model\model.ckpt-90000
set PREDICT_DIR=%EXP_FOLDER%\predict

set CITYSCAPES_DATASET=%CITYSCAPES_DIR%\tfrecord

python deeplab\mt_predict.py ^
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
    --vis_logdir=%PREDICT_DIR% ^
    --dataset_dir=%CITYSCAPES_DATASET% ^
  && ^
python deeplab\mt_eval.py ^
    --exp-name=%EXP_NAME% ^
  && ^
python deeplab\mt_vis.py ^
    --exp-name=%EXP_NAME% ^
