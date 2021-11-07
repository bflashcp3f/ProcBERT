#!/bin/bash

MODEL_NAME=${1}
BUDGET=${2}
DATA_NAME=${3}
BATCH_SIZE=${4}
EPOCH=${5}
LR=${6}
MAX_LEN=${7}
GPU_ID=${8}
OUTPUT_DIR=${9}

python code/ner_budget.py \
  --lm_model "$MODEL_NAME" \
  --data_name "$DATA_NAME" \
  --gpu_ids "$GPU_ID"  \
  --output_dir "$OUTPUT_DIR"/"$DATA_NAME" \
  --learning_rate "$LR" \
  --task_name ner \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --epochs "$EPOCH" \
  --budget "$BUDGET" \
  --save_model