#!/bin/bash

MODEL_NAME=${1}
DATA_NAME=${2}
BATCH_SIZE=${3}
EPOCH=${4}
LR=${5}
MAX_LEN=${6}
GPU_ID=${7}
OUTPUT_DIR=${8}

python code/ner.py \
  --lm_model "$MODEL_NAME" \
  --data_name "$DATA_NAME" \
  --gpu_ids "$GPU_ID"  \
  --output_dir "$OUTPUT_DIR"/"$DATA_NAME" \
  --learning_rate "$LR" \
  --task_name ner \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --epochs "$EPOCH" \
  --save_model