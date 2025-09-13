import os
import shutil
import random

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(BASE_DIR, "allDataYOLO")

images_dir = os.path.join(base_dir, r"images")
labels_dir = os.path.join(base_dir, r"labels")
print(images_dir)
print(labels_dir)

output_dir = os.path.join(base_dir, "split_dataset")
train_ratio = 0.8
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

# === STEP 1: Filter only labeled images ===
label_files = set()  # create an empty set to store label names without extensions

# loop through every file in the labels folder
for filename in os.listdir(labels_dir):
    print(filename)
    # check if the file ends with '.txt'
    if filename.endswith(".txt"):
        print('here')
        # split the filename into (name, extension)
        name_without_ext, ext = os.path.splitext(filename)
        # add just the name (no extension) to our set
        label_files.add(name_without_ext)
print(label_files)
labeled_images = []

for img_file in os.listdir(images_dir):
    name, ext = os.path.splitext(img_file)
    if ext.lower() not in image_exts:
        continue
    if name in label_files:
        labeled_images.append(img_file)
    else:
        os.remove(os.path.join(images_dir, img_file))  # remove unlabeled

print(f"Found {len(labeled_images)} labeled images after cleanup.")

# === STEP 2: Split into train/val sets ===
random.shuffle(labeled_images)
split_idx = int(len(labeled_images) * train_ratio)
train_imgs = labeled_images[:split_idx]
val_imgs = labeled_images[split_idx:]

def copy_files(img_list, subset):
    img_out_dir = os.path.join(output_dir, subset, "images")
    lbl_out_dir = os.path.join(output_dir, subset, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    for img_file in img_list:
        name, _ = os.path.splitext(img_file)
        label_file = name + ".txt"

        # Copy image
        shutil.copy2(os.path.join(images_dir, img_file), os.path.join(img_out_dir, img_file))
        # Copy label
        shutil.copy2(os.path.join(labels_dir, label_file), os.path.join(lbl_out_dir, label_file))

copy_files(train_imgs, "train")
copy_files(val_imgs, "val")

print(f"✅ Done! Train set: {len(train_imgs)} images, Val set: {len(val_imgs)} images.")
print(f"YOLO format ready in: {output_dir}")
