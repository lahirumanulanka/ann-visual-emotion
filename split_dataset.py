import os
import random
from pathlib import Path
from shutil import copy2

random.seed(42)
src_root = Path("/Users/lahirumunasinghe/Documents/DataScience/ann-visual-emotion/data/raw/EmoSet")
train_root = Path("/Users/lahirumunasinghe/Documents/DataScience/ann-visual-emotion/data/raw/train")
val_root = Path("/Users/lahirumunasinghe/Documents/DataScience/ann-visual-emotion/data/raw/validation")
split_ratio = 0.8  # 80% train, 20% val

for class_dir in src_root.iterdir():
    if not class_dir.is_dir():
        continue
    images = list(class_dir.glob("*.jpg"))
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Create target dirs
    (train_root / class_dir.name).mkdir(parents=True, exist_ok=True)
    (val_root / class_dir.name).mkdir(parents=True, exist_ok=True)

    # Copy files
    for img in train_imgs:
        copy2(img, train_root / class_dir.name / img.name)
    for img in val_imgs:
        copy2(img, val_root / class_dir.name / img.name)

print("Split complete.")
