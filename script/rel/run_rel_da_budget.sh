#!/bin/bash

MODEL_NAME=${1}
BUDGET=${2}
SRC_DATA=${3}
TGT_DATA=${4}
BATCH_SIZE=${5}
EPOCH=${6}
LR=${7}
MAX_LEN=${8}
GPU_ID=${9}
SAMP_RATE=${10}
OUTPUT_DIR=${11}

python code/rel_da_budget.py \
  --lm_model "$MODEL_NAME" \
  --src_data "$SRC_DATA"     \
  --tgt_data "$TGT_DATA"     \
  --gpu_ids "$GPU_ID"  \
  --output_dir "$OUTPUT_DIR"/da/"$SRC_DATA"_"$TGT_DATA" \
  --learning_rate "$LR" \
  --task_name fa_rel \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --epochs "$EPOCH" \
  --budget "$BUDGET" \
  --alpha 1 \
  --down_sample \
  --down_sample_rate "$SAMP_RATE" \
  --save_model