#!/bin/bash

# USE: ./train.sh (keeps old checkpoints) OR ./train.sh true (Clear checkpoints)
CLEAN_CHECKPOINTS=${1:-false}

if [ "$CLEAN_CHECKPOINTS" = true ]; then
    echo "Replacing previously saved checkpoints."
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
echo "Checking Python dependencies..."
while IFS= read -r package; do
    # Extract package name (remove version constraints)
    pkg_name=$(echo "$package" | sed -E 's/[<>=].*$//')
    if ! pip show "$pkg_name" &>/dev/null; then
        echo "Installing $pkg_name..."
        pip install --quiet "$package" || true
    fi
done < requirements.txt


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
nohup python3 -u train.py \
  --train_json_data data/newdata_cls_train_vidsplit_n.json \
  --eval_json_data data/newdata_cls_val_vidsplit_n.json \
  --img_dir /mydata/vocim/zachary/data/cropped \
  > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "Monitor logs with:"
echo "  - Full log: tail -f $LOG_FILE"
echo "  - Clean summary: tail -f $SUMMARY_FILE"
echo "  - Errors: tail -f $ERROR_LOG"
echo "Check training with: ps aux | grep \"python3 train.py\""
echo "Kill the process with: pkill -f \"python3 -u train.py\""
{
echo "=========================================================================="
echo "Training started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "PID: $TRAIN_PID"
echo "=========================================================================="
} >> "$LOG_FILE"
wait $TRAIN_PID

grep -E '^Accuracy: |^Epoch |^Checkpoint saved|^Best model saved|Total training time' "$LOG_FILE" > "$SUMMARY_FILE"
echo "Training completed. Summary file created at: $SUMMARY_FILE"
