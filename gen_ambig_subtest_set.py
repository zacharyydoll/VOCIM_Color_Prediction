import os
import json
import re
import yaml
import pandas as pd
from PIL import Image

TEST_JSON_PATH = "/mydata/vocim/zachary/color_prediction/data/newdata_test_vidsplit_n.json"
SHARED_ANNOTATIONS_BASE = "/mydata/vocim/zachary/data/shared/KeypointAnnotations"
OUTPUT_JSON_PATH = "/mydata/vocim/zachary/color_prediction/data/mult_bkpk_sub_test_set.json"
OUTPUT_EXAMPLES_DIR = "/mydata/vocim/zachary/color_prediction/output_examples"  

with open("/mydata/vocim/zachary/color_prediction/newdata_bird_identity.yaml", "r") as f:
    bird_identity_mapping = yaml.safe_load(f)
with open("/mydata/vocim/zachary/color_prediction/newdata_colormap.yaml", "r") as f:
    color_map = yaml.safe_load(f)

def process_directory_for_ambiguity(annotation_dir, ambiguous_files):
    """
    Process one annotation directory (which contains a CSV and images).
    For each CSV row, if there are two or more valid bird groups (i.e. each with a backpack coordinate),
    then for each identity group in that row (that would produce a crop), compute its output file name
    (the same as your cropping script does) and add it to the set `ambiguous_files`.
    """
    csv_file = None
    for file in os.listdir(annotation_dir):
        if file.startswith("CollectedData_") and file.endswith(".csv"):
            csv_file = os.path.join(annotation_dir, file)
            break
    if csv_file is None:
        print(f"No CSV file found in {annotation_dir}")
        return

    try:
        header_df = pd.read_csv(csv_file, header=None, nrows=4)
    except Exception as e:
        print(f"Error reading header from {csv_file}: {e}")
        return
    individuals = header_df.iloc[1, 3:].tolist()
    bodyparts = header_df.iloc[2, 3:].tolist()

    try:
        df = pd.read_csv(csv_file, header=None, skiprows=4)
    except Exception as e:
        print(f"Error reading CSV data from {csv_file}: {e}")
        return

    for idx, row in df.iterrows():
        file_name = row[2]
        img_path = os.path.join(annotation_dir, file_name)
        if not os.path.exists(img_path):
            continue

        # Group keypoints by identity
        identity_groups = {}
        for i in range(3, len(row), 2):
            x = row[i]
            y = row[i+1]
            if pd.notna(x) and pd.notna(y):
                try:
                    coord = (float(x), float(y))
                    identity = individuals[i - 3]
                    part = bodyparts[i - 3]
                    if identity not in identity_groups:
                        identity_groups[identity] = {'coords': [], 'backpack_coords': []}
                    identity_groups[identity]['coords'].append(coord)
                    # Store backpack coordinate 
                    if isinstance(part, str) and part.lower() == "backpack":
                        if not identity_groups[identity]['backpack_coords']:
                            identity_groups[identity]['backpack_coords'].append(coord)
                except Exception as e:
                    continue

        num_valid = sum(1 for group in identity_groups.values() if group['backpack_coords'])
        if num_valid >= 2:
            
            base, ext = os.path.splitext(file_name)
            rel_path = os.path.relpath(annotation_dir, SHARED_ANNOTATIONS_BASE)
            for primary_id, group in identity_groups.items():
                if not group['coords'] or not group['backpack_coords']:
                    continue
                xs = [pt[0] for pt in group['coords']]
                ys = [pt[1] for pt in group['coords']]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                margin_x = 0.2 * (max_x - min_x)
                margin_y = 0.2 * (max_y - min_y)
                crop_min_x = max(min_x - margin_x, 0)
                crop_min_y = max(min_y - margin_y, 0)
                try:
                    image = Image.open(img_path)
                except Exception as e:
                    continue
                crop_max_x = min(max_x + margin_x, image.width)
                crop_max_y = min(max_y + margin_y, image.height)

                output_img_name = f"{base}_{primary_id}{ext}"
                relative_output_file = os.path.join(rel_path, output_img_name)
                ambiguous_files.add(relative_output_file)

def main():
    ambiguous_files = set()

    try:
        with open(TEST_JSON_PATH, "r") as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Error loading test JSON file: {e}")
        return

    # Get unique directories from the "images" file_name values
    test_dirs = set()
    for img_entry in test_data.get("images", []):
        dir_part = os.path.dirname(img_entry.get("file_name", ""))
        if dir_part:
            full_dir = os.path.join(SHARED_ANNOTATIONS_BASE, dir_part)
            test_dirs.add(full_dir)

    print(f"Found {len(test_dirs)} directories in test set.")
    for directory in test_dirs:
        if os.path.exists(directory):
            print(f"Processing {directory} …")
            process_directory_for_ambiguity(directory, ambiguous_files)
        else:
            print(f"Directory {directory} does not exist; skipping.")

    print(f"Identified {len(ambiguous_files)} ambiguous image file names.")

    # Filter test JSON images and annotations based on ambiguous_files
    filtered_images = [img for img in test_data.get("images", []) if img.get("file_name") in ambiguous_files]
    kept_ids = {img["id"] for img in filtered_images}
    filtered_annotations = [ann for ann in test_data.get("annotations", []) if ann.get("image_id") in kept_ids]

    new_data = {
        "images": filtered_images,
        "annotations": filtered_annotations
    }

    try:
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(new_data, f, indent=2)
        print(f"Saved sub test set JSON to {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"Error saving sub test set JSON: {e}")

if __name__ == "__main__":
    main()
