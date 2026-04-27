import os
import shutil
import random

# Path to your extracted dataset
SOURCE_DIR = "dataset"

# Output folders
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Train-test split ratio
split_ratio = 0.8

# Loop through each class
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    # Skip if not a folder
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create folders
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

    # Copy train images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TRAIN_DIR, class_name, img)
        shutil.copy(src, dst)

    # Copy test images
    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TEST_DIR, class_name, img)
        shutil.copy(src, dst)

print("✅ Dataset split completed!")