#!/bin/bash

cd /mydata/vocim/zachary/color_prediction/

# Only install if not already present (quiet mode)
pip install --quiet matplotlib || true
pip install --quiet scikit-learn || true
pip install --quiet pandas || true

# Define log files
LOG_FILE="logs/output.log"
ERROR_LOG="logs/error.log"

# Create logs directory if needed
mkdir -p logs

# Clear previous logs
> "$LOG_FILE"
> "$ERROR_LOG"

# Start training
nohup python train.py \
  --train_json_data data/vocim_yolopose_train_vidsplit.json \
  --eval_json_data data/vocim_yolopose_val_vidsplit.json \
  --img_dir /mydata/vocim/zachary/data/cropped \
  > "$LOG_FILE" 2> "$ERROR_LOG" &

echo "Training started. Monitor logs with:"
echo "tail -f $LOG_FILE"