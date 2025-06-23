#!/usr/bin/env bash
set -e

trap '' SIGINT

ENSEMBLE_SIZE=5
ENSEMBLE_DIR="/mydata/vocim/zachary/color_prediction/deep_ensembles/ensembles"

# If "true" as first argument, clean out existing models:
CLEAN_ENSEMBLE=${1:-false}
if [ "$CLEAN_ENSEMBLE" = true ]; then
    echo "You are about to delete all existing ensemble models in $ENSEMBLE_DIR."
    echo "Type 'yes' to confirm, anything else to abort."
    read CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted. No models were deleted."
        exit 1
    fi
    echo "Replacing previously saved ensemble models."
    rm -rf "$ENSEMBLE_DIR"
fi

mkdir -p "$ENSEMBLE_DIR"

for SEED in $(seq 1 $ENSEMBLE_SIZE); do
    MODEL_DIR="$ENSEMBLE_DIR/model${SEED}"
    mkdir -p "$MODEL_DIR/logs" "$MODEL_DIR/checkpoints"

    echo "────────────────────────────────────────────────────────────────"
    echo "Launching training for ensemble member $SEED (seed=$SEED)…"
    echo "  Logs → $MODEL_DIR/logs/output.log"
    echo "  Checkpoints → $MODEL_DIR/checkpoints"
    echo "────────────────────────────────────────────────────────────────"

    nohup python3 /mydata/vocim/zachary/color_prediction/deep_ensembles/train_ensemble.py \
        --seed $SEED \
        --train_json_data /mydata/vocim/zachary/color_prediction/data/newdata_cls_train_vidsplit_n.json \
        --eval_json_data /mydata/vocim/zachary/color_prediction/data/newdata_cls_val_vidsplit_n.json \
        --img_dir /mydata/vocim/zachary/data/cropped \
        --output_dir "$MODEL_DIR/checkpoints" \
        > "$MODEL_DIR/logs/output.log" 2>&1 &

    PID=$!

    echo " → Ensemble member $SEED started as PID $PID."

    wait $PID

    echo "Ensemble member $SEED (PID $PID) has completed training."
done

echo "────────────────────────────────────────────────────────────────"
echo "All $ENSEMBLE_SIZE ensemble members have finished training."
echo "If you need to kill any lingering jobs, run:  pkill -f train_ensemble.py"
echo "────────────────────────────────────────────────────────────────"
