#!/bin/bash

MODEL_NAME=${1}
BUDGET=${2}
DATA_NAME=${3}
BATCH_SIZE=${4}
EPOCH=${5}
LR=${6}
MAX_LEN=${7}
GPU_ID=${8}
SAMP_RATE=${9}
OUTPUT_DIR=${10}

python code/rel_budget.py \
  --lm_model "$MODEL_NAME" \
  --data_name "$DATA_NAME" \
  --gpu_ids "$GPU_ID"  \
  --output_dir "$OUTPUT_DIR"/"$DATA_NAME" \
  --learning_rate "$LR" \
  --task_name rel \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --epochs "$EPOCH" \
  --budget "$BUDGET" \
  --down_sample \
  --down_sample_rate "$SAMP_RATE" \
  --save_model