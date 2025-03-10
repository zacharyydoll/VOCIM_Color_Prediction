#!/bin/bash

# USE: ./train.sh (keeps old checkpoints) OR ./train.sh true (Clear checkpoints)
CLEAN_CHECKPOINTS=${1:-false}

if [ "$CLEAN_CHECKPOINTS" = true ]; then
    rm -f top_colorid_ckpt.pth top_colorid_best_model.pth
fi

cd /mydata/vocim/zachary/color_prediction/

# ----- Install System Dependencies -----
# Update package lists (quietly)
apt-get update -qq > /dev/null

# Install bc if not already present
if ! command -v bc &> /dev/null; then
    echo "Installing bc calculator..."
    apt-get install -qq -y bc > /dev/null
fi

# ----- Python Setup -----
# Only install if not already present (quiet mode)
pip install --quiet matplotlib || true
pip install --quiet scikit-learn || true
pip install --quiet pandas || true

# Define log files
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/output.log"
ERROR_LOG="$LOG_DIR/error.log"
SUMMARY_FILE="$LOG_DIR/output_summary.log"

# Clear previous logs
> "$LOG_FILE"
> "$ERROR_LOG"
> "$SUMMARY_FILE"

# Start training silently
nohup python -u train.py \
  --train_json_data data/vocim_yolopose_train_vidsplit.json \
  --eval_json_data data/vocim_yolopose_val_vidsplit.json \
  --img_dir /mydata/vocim/zachary/data/cropped \
  > "$LOG_FILE" 2> "$ERROR_LOG" &

# Start enhanced summary generator
(
  START_TIME=$(date +%s)
  tail -f "$LOG_FILE" | grep --line-buffered -E \
  '^Accuracy: |^Epoch |^Checkpoint saved|^Best model saved|Total training time' | \
  while read line
  do
    # Add epoch timing
    if [[ "$line" == *"Epoch "* ]]; then
      EPOCH_TIME=$(date +%s)
      echo "$line | Epoch duration: $((EPOCH_TIME - PREV_EPOCH_TIME))s" >> "$SUMMARY_FILE"
      PREV_EPOCH_TIME=$EPOCH_TIME
    elif [[ "$line" == *"Accuracy: "* ]]; then
      PREV_EPOCH_TIME=$(date +%s)
      echo "$line" >> "$SUMMARY_FILE"
    else
      echo "$line" >> "$SUMMARY_FILE"
    fi
    
    # Add empty line after "Best model saved" lines
    [[ "$line" == *"Best model saved"* ]] && echo >> "$SUMMARY_FILE"
  done
  
  # Calculate total time when process exits
  TOTAL_TIME=$(( $(date +%s) - START_TIME ))
  echo -e "\nTotal training time: ${TOTAL_TIME}s" >> "$SUMMARY_FILE"
) &

echo "Training started. Monitor logs with:"
echo "  - Full log: tail -f $LOG_FILE"
echo "  - Clean summary: tail -f $SUMMARY_FILE"
echo "  - Errors: tail -f $ERROR_LOG"
echo "Check training with: ps aux | grep "python train.py""
echo "Kill the process with: pkill -f "python train.py""
