import json

# Paths to your files
original_train_json = "/mydata/vocim/zachary/color_prediction/data/newdata_cls_train_vidsplit_n.json"
ambig_train_json = "/mydata/vocim/zachary/color_prediction/data/ambig_train_samples.json"
original_test_json = "/mydata/vocim/zachary/color_prediction/data/newdata_test_vidsplit_n.json"
ambig_test_json = "/mydata/vocim/zachary/color_prediction/data/mult_bkpk_sub_test_set.json"

def count_images(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return len(data["images"])

def print_ambiguous_stats(original_json, ambiguous_json, label, file=None):
    total = count_images(original_json)
    ambiguous = count_images(ambiguous_json)
    percent = 100 * ambiguous / total if total > 0 else 0
    output = (
        f"{label}:\n"
        f"  Total cropped images: {total}\n"
        f"  Ambiguous cropped images: {ambiguous}\n"
        f"  Percentage ambiguous: {percent:.2f}%\n\n"
    )
    print(output, end='')  # Print to console
    if file is not None:
        file.write(output)

with open("count.log", "w") as log_file:
    print_ambiguous_stats(original_train_json, ambig_train_json, "Training set", file=log_file)
    print_ambiguous_stats(original_test_json, ambig_test_json, "Test set", file=log_file)
    
print("counts saved to count.log")