#!/bin/bash

set -e

ENSEMBLE_SIZE=5
ENSEMBLE_DIR="deep_ensembles/ensembles"

# confirm if using the "true" argument 
CLEAN_ENSEMBLE=${1:-false}
if [ "$CLEAN_ENSEMBLE" = true ]; then
    echo "You are about to delete all existing ensemble models in $ENSEMBLE_DIR. Are you sure? Type 'yes' to confirm, anything else to abort."
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
    echo "Training ensemble member $SEED (seed=$SEED)"
    python3 deep_ensembles/train_ensemble.py \
        --seed $SEED \
        --train_json_data data/newdata_cls_train_vidsplit_n.json \
        --eval_json_data data/newdata_cls_val_vidsplit_n.json \
        --img_dir /mydata/vocim/zachary/data/cropped \
        --output_dir "$MODEL_DIR/checkpoints" \
        > "$MODEL_DIR/logs/output.log" 2>&1
done

echo "To monitor running ensemble training jobs:"
echo "  ps aux | grep train_ensemble.py"
echo "To kill all running ensemble training jobs:"
echo "  pkill -f train_ensemble.py" 