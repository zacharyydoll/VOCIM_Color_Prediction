import os
import json
import re
import yaml
import pandas as pd
from PIL import Image, ImageDraw

TEST_JSON_PATH = "/mydata/vocim/zachary/color_prediction/data/newdata_test_vidsplit_n.json"
SHARED_ANNOTATIONS_BASE = "/mydata/vocim/zachary/data/shared/KeypointAnnotations"
OUTPUT_EXAMPLES_DIR = "/mydata/vocim/zachary/color_prediction/output_examples"
LOG_FILE_PATH = "/mydata/vocim/zachary/color_prediction/data/test_set_cooccurrence_matrix.log"
MAX_EXAMPLES_TO_SAVE = 15

# Load YAML mappings for bird identities and colormap.
with open("/mydata/vocim/zachary/color_prediction/newdata_bird_identity.yaml", "r") as f:
    bird_identity_mapping = yaml.safe_load(f)
with open("/mydata/vocim/zachary/color_prediction/newdata_colormap.yaml", "r") as f:
    color_map = yaml.safe_load(f)
# create reverse mapping from numeric label to color name.
reverse_color_map = {v: k for k, v in color_map.items()}

def get_effective_label_for_identity(identity, rel_dir, bird_identity_mapping, color_map):
    """
    Mimics the effective label extraction from dataset.py.
    Given an identity string and the image's relative directory,
    returns the effective label (a numeric class) using the YAML mappings.
    """
    m = re.search(r'(bird(?:_[a-z])?_) *(\d+)', identity, re.IGNORECASE)
    if m:
        bird_key = f"bird_{m.group(2)}"
    else:
        raise ValueError(f"Could not parse identity from: {identity}")
    
    # use relative directory as the key. If not found, try the basename
    if rel_dir not in bird_identity_mapping:
        alt_dir = os.path.basename(rel_dir)
        if alt_dir in bird_identity_mapping:
            rel_dir = alt_dir
        else:
            raise ValueError(f"Directory {rel_dir} not found in bird identity mapping.")
    
    bird_mapping = bird_identity_mapping[rel_dir]
    if bird_key not in bird_mapping:
        raise ValueError(f"Bird key {bird_key} not found for directory {rel_dir}.")
    color_name = bird_mapping[bird_key]
    if color_name not in color_map:
        raise ValueError(f"Color {color_name} not found in colormap.")
    return color_map[color_name]

def process_directory(annotation_dir, cooccurrence_matrix, primary_counts, examples_saved, global_stats, bird_identity_mapping, color_map):
    """
    Process a single annotation directory (which should contain a CSV file and images).
    For each CSV row:
      - Group keypoints by bird identity (from the header's individuals and bodyparts).
      - Compute a bounding box (with 20% margin) from all keypoints.
      - Use the one provided backpack coordinate for each bird.
      - Determine the effective (color) label for each identity using YAML mappings.
      - For each primary crop, check if any other (secondary) bird's backpack coordinate falls inside its bounding box.
      - Update the co-occurrence matrix and primary crop counts.
      - Also update a global counter for images that contain 2 or more visible backpacks.
      - Save up to MAX_EXAMPLES_TO_SAVE composite cropped examples (with annotation banner).
    """
    csv_file = None
    for file in os.listdir(annotation_dir):
        if file.startswith("CollectedData_") and file.endswith(".csv"):
            csv_file = os.path.join(annotation_dir, file)
            break
    if csv_file is None:
        print(f"No CSV file found in {annotation_dir}")
        return
    
    print(f"Processing CSV: {csv_file}")
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

    # Compute the relative directory (used in effective label mapping)
    rel_dir = os.path.relpath(annotation_dir, SHARED_ANNOTATIONS_BASE)
    
    for idx, row in df.iterrows():
        file_name = row[2]  # Image file name is in column 2
        img_path = os.path.join(annotation_dir, file_name)
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist; skipping.")
            continue

        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue
        
        # group keypoints by identity
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
                    # for backpack, just store the coordinate
                    if isinstance(part, str) and part.lower() == "backpack":
                        if not identity_groups[identity]['backpack_coords']:
                            identity_groups[identity]['backpack_coords'].append(coord)
                except Exception as e:
                    print(f"Error processing coordinate at index {i}: {e}")
                    continue

        # count number of valid birds (with backpack coordinate) in that row
        num_valid = sum(1 for group in identity_groups.values() if group['backpack_coords'])
        if num_valid >= 2:
            global_stats["multiple"] += 1

        # for each PRIMARY identity group 
        for primary_id, group in identity_groups.items():
            if not group['coords'] or not group['backpack_coords']:
                continue  # Skip if no keypoints or no backpack coordinate.
            
            # Compute bounding box from all keypoints (c.f. crop_annotations.py)
            xs = [pt[0] for pt in group['coords']]
            ys = [pt[1] for pt in group['coords']]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            margin_x = 0.2 * (max_x - min_x)
            margin_y = 0.2 * (max_y - min_y)
            crop_min_x = max(min_x - margin_x, 0)
            crop_min_y = max(min_y - margin_y, 0)
            crop_max_x = min(max_x + margin_x, image.width)
            crop_max_y = min(max_y + margin_y, image.height)
            crop_bbox = (crop_min_x, crop_min_y, crop_max_x, crop_max_y)

            try:
                primary_class = get_effective_label_for_identity(primary_id, rel_dir, bird_identity_mapping, color_map)
            except Exception as e:
                print(f"Error getting effective label for primary {primary_id} in {file_name}: {e}")
                continue

            primary_counts[primary_class] += 1

            # accumulate secondary classes visible.
            secondary_visible = False
            secondary_classes = set()
            for secondary_id, sec_group in identity_groups.items():
                if secondary_id == primary_id or not sec_group['backpack_coords']:
                    continue
                sec_bp = sec_group['backpack_coords'][0]
                if (crop_min_x <= sec_bp[0] <= crop_max_x) and (crop_min_y <= sec_bp[1] <= crop_max_y):
                    try:
                        sec_class = get_effective_label_for_identity(secondary_id, rel_dir, bird_identity_mapping, color_map)
                        secondary_classes.add(sec_class)
                    except Exception as e:
                        print(f"Error getting effective label for secondary {secondary_id} in {file_name}: {e}")
                        continue
                    secondary_visible = True

            # Update cooccurrence matrix if any secondary birds are visible
            if secondary_visible:
                for sec_class in secondary_classes:
                    cooccurrence_matrix[primary_class][sec_class] += 1

            # save the crop example if at least one secondary is visible
            if secondary_visible and examples_saved["count"] < MAX_EXAMPLES_TO_SAVE:
                try:
                    cropped_image = image.crop(crop_bbox)
                    
                    # Annotations for saved imgs (for debugging)
                    primary_color = reverse_color_map.get(primary_class, "N/A")
                    sec_text_parts = []
                    for cls in sorted(secondary_classes):
                        sec_color = reverse_color_map.get(cls, "N/A")
                        sec_text_parts.append(f"{cls} ({sec_color})")
                    sec_text = ", ".join(sec_text_parts) if sec_text_parts else "None"
                    text_lines = [
                        f"File: {img_path}",
                        f"Cropped around class: {primary_class} ({primary_color})",
                        f"Other bird(s) visible: {sec_text}"
                    ]
                    
                    banner_height = 50
                    crop_width, crop_height = cropped_image.size
                    banner = Image.new("RGB", (crop_width, banner_height), color=(255, 255, 255))
                    draw = ImageDraw.Draw(banner)
                    y_text = 5
                    for line in text_lines:
                        draw.text((5, y_text), line, fill=(0, 0, 0))
                        y_text += 15
                        
                    composite = Image.new("RGB", (crop_width, crop_height + banner_height), color=(255, 255, 255))
                    composite.paste(cropped_image, (0, 0))
                    composite.paste(banner, (0, crop_height))
                    
                    base, ext = os.path.splitext(file_name)
                    output_img_name = f"{base}_{primary_id}{ext}"
                    os.makedirs(OUTPUT_EXAMPLES_DIR, exist_ok=True)
                    output_img_path = os.path.join(OUTPUT_EXAMPLES_DIR, output_img_name)
                    composite.save(output_img_path)
                    print(f"Saved annotated crop: {output_img_path}")
                    examples_saved["count"] += 1
                except Exception as e:
                    print(f"Error saving composite cropped image for {file_name}: {e}")

def main():
    # Initialize 8x8 cooccurrence matrix for classes 0-7
    cooccurrence_matrix = [[0 for _ in range(8)] for _ in range(8)]
    primary_counts = [0 for _ in range(8)]
    examples_saved = {"count": 0}
    # global stats for counting imgs with 2 or more visible backpacks
    global_stats = {"multiple": 0}

    # load test JSON file to determine which directories to process
    try:
        with open(TEST_JSON_PATH, "r") as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Error loading test JSON file {TEST_JSON_PATH}: {e}")
        return

    total_test_images = len(test_data.get("images", []))
    print(f"Total test images (from JSON): {total_test_images}")

    test_dirs = set()
    for img_entry in test_data.get("images", []):
        rel_dir = os.path.dirname(img_entry.get("file_name", ""))
        if rel_dir:
            test_dirs.add(rel_dir)
    print(f"Found {len(test_dirs)} unique directories from test JSON.")

    # Process each directory from the test set
    for rel_dir in test_dirs:
        full_dir = os.path.join(SHARED_ANNOTATIONS_BASE, rel_dir)
        if os.path.exists(full_dir):
            print(f"Processing directory: {full_dir}")
            process_directory(full_dir, cooccurrence_matrix, primary_counts, examples_saved, global_stats, bird_identity_mapping, color_map)
        else:
            print(f"Directory {full_dir} does not exist; skipping.")

    # Build log output
    log_lines = []
    log_lines.append("Co-occurrence matrix (rows: primary crop class, columns: secondary visible backpack class):")
    header_line = "\t" + "\t".join(str(i) for i in range(8))
    log_lines.append(header_line)
    for i, row in enumerate(cooccurrence_matrix):
        row_str = "\t".join(str(val) for val in row)
        log_lines.append(f"{i}\t{row_str}")
    log_lines.append("")
    log_lines.append("Total images per primary crop class:")
    for cls, count in enumerate(primary_counts):
        log_lines.append(f"Class {cls}: {count} images")
    log_lines.append("")
    
    #Summary lines
    percent = (global_stats["multiple"] / total_test_images * 100) if total_test_images > 0 else 0
    summary_line = (f"Out of {total_test_images} image files in the test set, "
                    f"{global_stats['multiple']} contain 2 or more backpacks visible, "
                    f"i.e. {percent:.1f}% of them.")
    log_lines.append(summary_line)

    try:
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        with open(LOG_FILE_PATH, "w") as log_file:
            log_file.write("\n".join(log_lines))
        print(f"Saved co-occurrence matrix log to {LOG_FILE_PATH}")
    except Exception as e:
        print(f"Error writing log file {LOG_FILE_PATH}: {e}")

if __name__ == "__main__":
    main()
