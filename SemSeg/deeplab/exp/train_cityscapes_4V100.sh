#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

DATA_DIR=/home/ubuntu/Data
export PYTHONPATH=`pwd`:../slim:$PYTHONPATH

# Set up the working directories.
CITYSCAPES_DIR=$DATA_DIR/Cityscapes
EXP_FOLDER=$DATA_DIR/Exp/DeepLabCityscapes/retrain-4V100
INIT_FOLDER=$DATA_DIR/Exp/DeepLabCityscapes/pretrained-ImageNet/model.ckpt
TRAIN_LOGDIR="${EXP_FOLDER}/train"
EVAL_LOGDIR="${EXP_FOLDER}/eval"
VIS_LOGDIR="${EXP_FOLDER}/vis"
EXPORT_DIR="${EXP_FOLDER}/export"

mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"


CITYSCAPES_DATASET=${CITYSCAPES_DIR}/tfrecord

NUM_ITERATIONS=90000
python deeplab/train.py \
  --num_clones 4 \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=769 \
  --train_crop_size=769 \
  --train_batch_size=8 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --dataset=cityscapes \
  --colormap_type=cityscapes \
  --tf_initial_checkpoint="${INIT_FOLDER}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${CITYSCAPES_DATASET}"

python deeplab/my_eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=1025 \
  --eval_crop_size=2049 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${CITYSCAPES_DATASET}" \
  --max_number_of_evaluations=1


