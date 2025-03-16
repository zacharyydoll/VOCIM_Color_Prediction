#!/bin/bash

# 1. Remove JSON files in data directory while preserving the directory
echo "Removing stored JSONs..."
find /mydata/vocim/zachary/color_prediction/data/ -name "*.json" -delete

# 2. Remove cropped directory
echo "Removing existing cropped directory..."
rm -rf /mydata/vocim/zachary/data/cropped

# 3. Install pandas
pip install pandas

clear 

# 4. Run crop annotations
echo "Cropping existing images..."
python3 /mydata/vocim/zachary/scripts/crop_annotations.py

# 5. Run merge JSON files and ensure output location
echo "Merging JSON files..."
python3 /mydata/vocim/zachary/scripts/merge_json_files.py
mv /mydata/vocim/zachary/scripts/cropped_merged_annotations.json /mydata/vocim/zachary/data/

# 6. Run video split and ensure output location
cd /mydata/vocim/zachary/color_prediction/
python split_by_video.py

echo "Data renewal complete."