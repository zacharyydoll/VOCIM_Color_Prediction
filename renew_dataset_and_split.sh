#!/bin/bash

# 1. Cleanup existing files
#rm -f /mydata/vocim/zachary/color_prediction/output.log
#rm -f /mydata/vocim/zachary/color_prediction/top_colorid_best_model.pth
#rm -f /mydata/vocim/zachary/color_prediction/top_colorid_ckpt.pth

# 2. Remove JSON files in data directory while preserving the directory
find /mydata/vocim/zachary/color_prediction/data/ -name "*.json" -delete

# 3. Remove cropped directory
rm -rf /mydata/vocim/zachary/data/cropped

# 4. Install pandas
pip install pandas

clear 

# 5. Run crop annotations
python3 /mydata/vocim/zachary/scripts/crop_annotations.py

# 6. Run merge JSON files and ensure output location
python3 /mydata/vocim/zachary/scripts/merge_json_files.py
mv /mydata/vocim/zachary/scripts/cropped_merged_annotations.json /mydata/vocim/zachary/data/

# 7. Run video split and ensure output location
cd /mydata/vocim/zachary/color_prediction/
python split_by_video.py

# 8. Start training
#nohup python train.py \
#  --train_json_data data/vocim_yolopose_train_vidsplit.json \
#  --eval_json_data data/vocim_yolopose_val_vidsplit.json \
#  --img_dir /mydata/vocim/zachary/data/cropped \
#  > output.log 2>&1 &