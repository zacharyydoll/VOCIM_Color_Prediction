#!/bin/bash

set -e

ENSEMBLE_SIZE=5
ENSEMBLE_DIR="deep_ensembles/ensembles"

for SEED in $(seq 1 $ENSEMBLE_SIZE); do
    MODEL_DIR="$ENSEMBLE_DIR/model${SEED}"
    CKPT_DIR="$MODEL_DIR/checkpoints"
    LOG_DIR="$MODEL_DIR/logs"
    mkdir -p "$CKPT_DIR" "$LOG_DIR"
    # Check for best model or final checkpoint
    BEST_MODEL=$(ls "$CKPT_DIR"/*best_model*.pth 2>/dev/null | head -n 1)
    FINAL_CKPT=$(ls "$CKPT_DIR"/ckpt_seed_${SEED}.pth 2>/dev/null | head -n 1)
    if [ -f "$BEST_MODEL" ]; then
        echo "Model $SEED already completed (best model found). Skipping."
        continue
    fi
    if [ -f "$FINAL_CKPT" ]; then
        echo "Resuming training for model $SEED from latest checkpoint."
        python deep_ensembles/train_ensemble.py \
            --seed $SEED \
            --train_json_data data/newdata_cls_train_vidsplit_n.json \
            --eval_json_data data/newdata_cls_val_vidsplit_n.json \
            --img_dir /mydata/vocim/zachary/data/cropped \
            --output_dir "$CKPT_DIR" \
            > "$LOG_DIR/output.log" 2>&1
    else
        echo "Starting training for model $SEED from scratch."
        python deep_ensembles/train_ensemble.py \
            --seed $SEED \
            --train_json_data data/newdata_cls_train_vidsplit_n.json \
            --eval_json_data data/newdata_cls_val_vidsplit_n.json \
            --img_dir /mydata/vocim/zachary/data/cropped \
            --output_dir "$CKPT_DIR" \
            > "$LOG_DIR/output.log" 2>&1
    fi
    # After finishing, continue to next model
    echo "Model $SEED training complete."
done

echo "To monitor running ensemble training jobs:"
echo "  ps aux | grep train_ensemble.py"
echo "To kill all running ensemble training jobs:"
echo "  pkill -f train_ensemble.py" 