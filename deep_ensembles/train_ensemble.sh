#!/usr/bin/env bash
set -e

# If you press Ctrl+C in this shell, we want to ignore it,
# so that the current Python training process isn't killed.
trap '' SIGINT

ENSEMBLE_SIZE=5
ENSEMBLE_DIR="/mydata/vocim/zachary/color_prediction/deep_ensembles/ensembles"

# If you pass "true" as the first argument, wipe out existing models:
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

    # (1) Start the training with nohup, redirecting stdout/stderr into the log file.
    #     The trailing '&' backgrounds the job, so we can capture its PID.
    nohup python3 /mydata/vocim/zachary/color_prediction/deep_ensembles/train_ensemble.py \
        --seed $SEED \
        --train_json_data /mydata/vocim/zachary/color_prediction/data/newdata_cls_train_vidsplit_n.json \
        --eval_json_data /mydata/vocim/zachary/color_prediction/data/newdata_cls_val_vidsplit_n.json \
        --img_dir /mydata/vocim/zachary/data/cropped \
        --output_dir "$MODEL_DIR/checkpoints" \
        > "$MODEL_DIR/logs/output.log" 2>&1 &

    # (2) Capture the PID of that backgrounded process
    PID=$!

    echo " → Ensemble member $SEED started as PID $PID."

    # (3) Wait for that PID to finish before moving to the next SEED
    wait $PID

    # When we reach here, seed $SEED has fully finished (or crashed).
    # We can optionally check its exit status via $?; because we used "set -e",
    # if the Python script crashes, this wrapper will exit immediately.
    echo "Ensemble member $SEED (PID $PID) has completed training."
done

echo "────────────────────────────────────────────────────────────────"
echo "All $ENSEMBLE_SIZE ensemble members have finished training."
echo "If you need to kill any lingering jobs, run:  pkill -f train_ensemble.py"
echo "────────────────────────────────────────────────────────────────"
