#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

DATA_DIR=/home/mli/Data
export PYTHONPATH=`pwd`:../slim:$PYTHONPATH

# Set up the working directories.
CITYSCAPES_DIR=$DATA_DIR/Cityscapes
EXP_FOLDER=$DATA_DIR/Exp/DeepLabCityscapes/pretrained
INIT_FOLDER="${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${EXP_FOLDER}/train/model.ckpt"
EVAL_LOGDIR="${EXP_FOLDER}/eval"
VIS_LOGDIR="${EXP_FOLDER}/vis"
EXPORT_DIR="${EXP_FOLDER}/export"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

CITYSCAPES_DATASET=${CITYSCAPES_DIR}/tfrecord


python deeplab/my_eval.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=1025 \
    --vis_crop_size=2049 \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir=${TRAIN_LOGDIR} \
    --vis_logdir=${VIS_LOGDIR} \
    --dataset_dir=${CITYSCAPES_DATASET} \
    --also_save_raw_predictions=True \
  

# Visualize the results.
# python deeplab/vis.py \
#     --logtostderr \
#     --vis_split="val" \
#     --model_variant="xception_65" \
#     --atrous_rates=6 \
#     --atrous_rates=12 \
#     --atrous_rates=18 \
#     --output_stride=16 \
#     --decoder_output_stride=4 \
#     --vis_crop_size=1025 \
#     --vis_crop_size=2049 \
#     --dataset="cityscapes" \
#     --colormap_type="cityscapes" \
#     --checkpoint_dir=${TRAIN_LOGDIR} \
#     --vis_logdir=${VIS_LOGDIR} \
#     --dataset_dir=${CITYSCAPES_DATASET} \

# python deeplab/eval.py \
#   --dataset cityscapes \
#   --logtostderr \
#   --eval_split="val" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --eval_crop_size=1025 \
#   --eval_crop_size=2049 \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --eval_logdir="${EVAL_LOGDIR}" \
#   --dataset_dir="${CITYSCAPES_DATASET}" \