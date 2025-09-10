"""Split a class-folder dataset into train/val subsets with stratified ratios.

Usage:
  python scripts/split_train_val.py \
      --data-root data/newDataset_cropped_gray \
      --out-root data/newDataset_cropped_gray_splits \
      --val-ratio 0.2 \
      --seed 42

Outputs:
  out-root/
    train/<class>/*.jpg
    val/<class>/*.jpg
    train.csv  (image_path, emotion)
    val.csv
    label_map.json
    stats.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    p = argparse.ArgumentParser(description="Split dataset into train/val.")
    p.add_argument('--data-root', required=True, help='Input dataset root with class folders')
    p.add_argument('--out-root', required=True, help='Output root for splits')
    p.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of symlink (default symlink)'
    )
    return p.parse_args()


def collect(data_root: Path) -> Dict[str, List[Path]]:
    classes = {}
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and not d.name.startswith('.'):
            files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if files:
                classes[d.name] = files
    return classes


def ensure_action(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        if dst.exists():
            return
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    class_map = collect(data_root)
    if not class_map:
        raise SystemExit("No class folders found.")

    label_map = {cls: idx for idx, cls in enumerate(sorted(class_map.keys()))}

    rows_train = []
    rows_val = []
    stats = {"per_class": {}, "val_ratio": args.val_ratio, "seed": args.seed}

    for cls, files in class_map.items():
        random.shuffle(files)
        n = len(files)
        val_count = max(1, int(n * args.val_ratio)) if n > 5 else max(1, int(n * args.val_ratio))
        val_files = files[:val_count]
        train_files = files[val_count:]
        stats["per_class"][cls] = {"train": len(train_files), "val": len(val_files), "total": n}

        for f in train_files:
            rel_out = Path('train') / cls / f.name
            ensure_action(f, out_root / rel_out, args.copy)
            rows_train.append({"image_path": str(rel_out), "emotion": cls})
        for f in val_files:
            rel_out = Path('val') / cls / f.name
            ensure_action(f, out_root / rel_out, args.copy)
            rows_val.append({"image_path": str(rel_out), "emotion": cls})

    # Write CSVs
    pd.DataFrame(rows_train).to_csv(out_root / 'train.csv', index=False)
    pd.DataFrame(rows_val).to_csv(out_root / 'val.csv', index=False)

    # Write label map & stats
    with open(out_root / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    stats['label_map'] = label_map
    stats['counts'] = {"train": len(rows_train), "val": len(rows_val)}
    with open(out_root / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("Split complete:")
    print(f" Train samples: {len(rows_train)}")
    print(f" Val samples:   {len(rows_val)}")
    print(f" Classes: {list(label_map.keys())}")
    print(f" Output root: {out_root}")


if __name__ == '__main__':
    main()
