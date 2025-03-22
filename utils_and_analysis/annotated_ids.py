#!/usr/bin/env python3
import csv
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

def get_annotation_for_image(csv_path, target_img):
    """
    Reads the CSV file and returns the list of coordinate values (as floats)
    for the row corresponding to the given target image filename.
    Assumes that:
      - The CSV contains header rows.
      - Each data row has the image filename in the third column.
      - Coordinate values start at column index 3.
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 2 and row[2].strip() == target_img:
                coords_str = row[3:]
                # Convert non-empty strings to floats.
                coords = [float(val) for val in coords_str if val.strip() != '']
                return coords
    raise ValueError(f"Image '{target_img}' not found in CSV file: {csv_path}")

def annotate_image(image_path, coords, output_path, y_offset=10):
    """
    Annotates the image at image_path with bird indices (1-8) using backpack coordinates.
    It assumes each bird has 10 coordinate values in the order:
      beak(x), beak(y), head(x), head(y), backpack(x), backpack(y),
      tailbe(x), tailbe(y), tailend(x), tailend(y).
    The index (1-8) is drawn a few pixels above the backpack coordinate.
    
    This function does NOT modify the input image; it creates a new image file.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 1000)
    except IOError:
        print("using default font")
        font = ImageFont.load_default()


    group_size = 10  
    num_birds = len(coords) // group_size
    for i in range(num_birds):
        start_idx = i * group_size

        if start_idx + group_size > len(coords):
            print(f"Incomplete coordinates for bird {i+1}, skipping.")
            continue
        backpack_x = coords[start_idx + 4]
        backpack_y = coords[start_idx + 5]
        x = int(round(backpack_x))
        y = int(round(backpack_y)) - y_offset

        text = str(i + 1)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill="black")
        draw.text((x, y), text, font=font, fill="white")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image.save(output_path)
    print(f"Annotated image saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Annotate bird IDs on an image using backpack coordinates from a CSV file."
    )
    parser.add_argument("csv_path", help="Path to the CSV annotation file containing multiple entries")
    parser.add_argument("--offset", type=int, default=20,
                        help="Vertical offset (in pixels) to place the text above the backpack (default: 10)")
    args = parser.parse_args()

    image_path = ("/mydata/vocim/zachary/data/shared/KeypointAnnotations/VOCIM_juvExpBP05/labeled-data_topview/BP_2023-07-06_08-54-48_131465_0500000/img02643.png")
    output_path = ("/mydata/vocim/zachary/color_prediction/utils_and_analysis/img15622_ANNOTATED.png")
    target_img = "img02643.png"

    coords = get_annotation_for_image(args.csv_path, target_img)
    annotate_image(image_path, coords, output_path, y_offset=args.offset)
