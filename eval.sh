#!/bin/bash

conda init bash
source ~/.bashrc

# Define log files
LOG_FILE="logs/train.log"
ERROR_LOG="logs/error.log"

# Clear previous error log (optional, if you want fresh logs each time)
> $ERROR_LOG

# Exit script on any error and capture the error message
set -e

# Function to handle errors and write to the error log
error_handler() {
    echo "Error occurred. Check $ERROR_LOG for details."
    echo "Error on line $1" >> $ERROR_LOG
}
trap 'error_handler $LINENO' ERR  # Trap any error, write the error line to error log

# Step 1: Activate the conda environment
conda activate /myhome/mambaforge/envs/mmpose
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

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

# Step 3: Run the Python script and log output to train.log
python eval.py 2>&1 | tee $LOG_FILE

# Optional: Notify that the script has finished
echo "Training job finished. Logs are saved in $LOG_FILE."
