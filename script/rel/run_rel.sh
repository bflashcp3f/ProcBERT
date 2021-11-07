#!/bin/bash

MODEL_NAME=${1}
DATA_NAME=${2}
BATCH_SIZE=${3}
EPOCH=${4}
LR=${5}
MAX_LEN=${6}
GPU_ID=${7}
SAMP_RATE=${8}
OUTPUT_DIR=${9}

python code/rel.py \
  --lm_model "$MODEL_NAME" \
  --data_name "$DATA_NAME" \
  --gpu_ids "$GPU_ID"  \
  --output_dir "$OUTPUT_DIR"/"$DATA_NAME" \
  --learning_rate "$LR" \
  --task_name rel \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --epochs "$EPOCH" \
  --down_sample \
  --down_sample_rate "$SAMP_RATE" \
  --save_model