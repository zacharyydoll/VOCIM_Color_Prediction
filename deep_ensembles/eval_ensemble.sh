#!/usr/bin/env bash
set -e 

ENSEMBLE_DIR="deep_ensembles/ensembles"
TEST_JSON="data/newdata_test_vidsplit_n.json"
IMG_DIR="/mydata/vocim/zachary/data/cropped"

echo "Checking Python dependencies..."
while IFS= read -r package; do
    # Extract package name (remove version constraints)
    pkg_name=$(echo "$package" | sed -E 's/[<>=].*$//')
    if ! pip show "$pkg_name" &>/dev/null; then
        echo "Installing $pkg_name..."
        pip install --quiet "$package" || true
    fi
done < requirements.txt

clear

for MODEL_DIR in $ENSEMBLE_DIR/model*/; do
    CKPT=$(ls "$MODEL_DIR/checkpoints/"*best_model*.pth 2>/dev/null | head -n 1)
    if [ -f "$CKPT" ]; then
        echo "Evaluating $CKPT"
        python3 eval.py \
            --model_path "$CKPT" \
            --data_path "$TEST_JSON" \
            --img_dir "$IMG_DIR" \
            --output_dir "$MODEL_DIR"
    else
        echo "No best model found in $MODEL_DIR"
    fi
done 