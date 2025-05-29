#!/bin/bash

ENSEMBLE_DIR="deep_ensembles/ensembles"
TEST_JSON="data/newdata_test_vidsplit_n.json"
IMG_DIR="/mydata/vocim/zachary/data/cropped"

for MODEL_DIR in $ENSEMBLE_DIR/model*/; do
    CKPT=$(ls "$MODEL_DIR/checkpoints/"*best_model*.pth 2>/dev/null | head -n 1)
    if [ -f "$CKPT" ]; then
        echo "Evaluating $CKPT"
        python eval.py \
            --model_path "$CKPT" \
            --data_path "$TEST_JSON" \
            --img_dir "$IMG_DIR" \
            --output_dir "$MODEL_DIR"
    else
        echo "No best model found in $MODEL_DIR"
    fi
done 