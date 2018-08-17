#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

DATA_DIR=/home/mli/Data
export PYTHONPATH=`pwd`:../slim:$PYTHONPATH

# Set up the working directories.
CITYSCAPES_DIR=$DATA_DIR/Cityscapes
EXP_FOLDER=$DATA_DIR/Exp/DeepLabCityscapes/pretrained
INIT_FOLDER="${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${EXP_FOLDER}/train"
EVAL_LOGDIR="${EXP_FOLDER}/eval"
VIS_LOGDIR="${EXP_FOLDER}/vis"
EXPORT_DIR="${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

CITYSCAPES_DATASET=${CITYSCAPES_DIR}/tfrecord

# Train 10 iterations.
NUM_ITERATIONS=10
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="trainval" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=512 \
  --train_crop_size=1024 \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${CITYSCAPES_DATASET}"

python deeplab/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=512 \
  --eval_crop_size=1024 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${CITYSCAPES_DATASET}" \
  --max_number_of_evaluations=1

# Visualize the results.
python deeplab/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=512 \
  --eval_crop_size=1024 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${CITYSCAPES_DATASET}" \
  --max_number_of_iterations=1

