import os
import json
import yaml
import pandas as pd
from PIL import Image

TEST_JSON_PATH = "/mydata/vocim/zachary/color_prediction/data/newdata_cls_train_vidsplit_n.json" # /mydata/vocim/zachary/color_prediction/data/newdata_test_vidsplit_n.json
SHARED_ANNOTATIONS_BASE = "/mydata/vocim/zachary/data/shared/KeypointAnnotations"
OUTPUT_JSON_PATH = "/mydata/vocim/zachary/color_prediction/data/ambig_train_samples.json" # /mydata/vocim/zachary/color_prediction/data/mult_bkpk_sub_test_set.json

# Load bird identity mapping and color map if needed
with open("/mydata/vocim/zachary/color_prediction/newdata_bird_identity.yaml", "r") as f:
    bird_identity_mapping = yaml.safe_load(f)
with open("/mydata/vocim/zachary/color_prediction/newdata_colormap.yaml", "r") as f:
    color_map = yaml.safe_load(f)

def process_directory_for_ambiguity(annotation_dir, ambiguous_files):
    """
    Process one annotation directory. For each CSV row, for each bird group (crop),
    determine if the computed crop contains at least one extra backpack coordinate 
    from another bird in that same image.
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
                    # For backpack, record the coordinate if not already recorded
                    if isinstance(part, str) and part.lower() == "backpack":
                        if not identity_groups[identity]['backpack_coords']:
                            identity_groups[identity]['backpack_coords'].append(coord)
                except Exception as e:
                    continue

        # Open the image for cropping computation
        try:
            image = Image.open(img_path)
        except Exception as e:
            continue
        
        base, ext = os.path.splitext(file_name)
        rel_path = os.path.relpath(annotation_dir, SHARED_ANNOTATIONS_BASE)
        
        # Process each group individually. Instead of marking the entire row as ambiguous,
        # check if the crop for the primary identity contains another bird's backpack.
        for primary_id, primary_group in identity_groups.items():
            if not primary_group['coords'] or not primary_group['backpack_coords']:
                continue

            # Compute bounding box for the primary identity crop:
            xs = [pt[0] for pt in primary_group['coords']]
            ys = [pt[1] for pt in primary_group['coords']]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            margin_x = 0.2 * (max_x - min_x)
            margin_y = 0.2 * (max_y - min_y)
            crop_min_x = max(min_x - margin_x, 0)
            crop_min_y = max(min_y - margin_y, 0)
            crop_max_x = min(max_x + margin_x, image.width)
            crop_max_y = min(max_y + margin_y, image.height)
            
            # Check if any other backpack coordinate falls within this crop
            crop_is_ambiguous = False
            for other_id, other_group in identity_groups.items():
                if other_id == primary_id:
                    continue
                if not other_group['backpack_coords']:
                    continue
                bx, by = other_group['backpack_coords'][0]
                if crop_min_x <= bx <= crop_max_x and crop_min_y <= by <= crop_max_y:
                    crop_is_ambiguous = True
                    break

            if crop_is_ambiguous:
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

    # Get unique directories from "file_name" entries in the test JSON.
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

    # Filter test JSON images and annotations based on ambiguous_files,
    # and filter out any images missing the "id" field.
    filtered_images = [
        img for img in test_data.get("images", [])
        if "id" in img and img.get("file_name") in ambiguous_files
    ]
    filtered_annotations = [
        ann for ann in test_data.get("annotations", [])
        if ann.get("image_id") in {img["id"] for img in filtered_images}
    ]

    # Reindex filtered_images and create an old-to-new id mapping.
    new_images = []
    new_id_map = {}
    for new_id, img in enumerate(filtered_images):
        new_images.append(img)
        old_id = img["id"]
        new_id_map[old_id] = new_id

    # Update the annotations to use the new image ids.
    new_annotations = []
    for ann in filtered_annotations:
        old_img_id = ann["image_id"]
        if old_img_id in new_id_map:
            ann["image_id"] = new_id_map[old_img_id]
            new_annotations.append(ann)

    new_data = {
        "images": new_images,
        "annotations": new_annotations
    }

    try:
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(new_data, f, indent=2)
        print(f"Saved sub test set JSON to {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"Error saving sub test set JSON: {e}")

if __name__ == "__main__":
    main()
