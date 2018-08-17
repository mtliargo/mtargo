@echo off

set DATA_DIR=D:\Data
set PYTHONPATH=%CD%;..\slim;%PYTHONPATH%

:: Set up the working directories.
set CITYSCAPES_DIR=%DATA_DIR%\Cityscapes
set EXP_FOLDER=%DATA_DIR%\Exp\DeepLabCityscapes\debug
set INIT_FOLDER=%DATA_DIR%\Exp\DeepLabCityscapes\pretrained-ImageNet\model.ckpt
set TRAIN_LOGDIR=%EXP_FOLDER%\trainlog
mkdir %TRAIN_LOGDIR%

set CITYSCAPES_DATASET=%CITYSCAPES_DIR%\tfrecord

python deeplab\train.py ^
    --logtostderr ^
    --training_number_of_steps=90000 ^
    --train_split=train ^
    --model_variant=xception_65 ^
    --atrous_rates=6 ^
    --atrous_rates=12 ^
    --atrous_rates=18 ^
    --output_stride=16 ^
    --decoder_output_stride=4 ^
    --train_crop_size=769 ^
    --train_crop_size=769 ^
    --train_batch_size=1 ^
    --fine_tune_batch_norm=False ^
    --dataset=cityscapes ^
    --colormap_type=cityscapes ^
    --tf_initial_checkpoint=%INIT_FOLDER% ^
    --train_logdir=%TRAIN_LOGDIR% ^
    --dataset_dir=%CITYSCAPES_DATASET% ^
    --log_steps=1 ^
    --save_summaries_secs=5 ^
    --save_summaries_images=True ^
