"""Merge YOLO emotion dataset valid+test splits and categorize images into class folders.

Actions:
  1. Read train/_annotations.csv, valid/_annotations.csv, test/_annotations.csv
  2. Merge valid + test into a single split (name: merged)
  3. Create output structure:
        out_root/
            train/<class>/*.jpg
            merged/<class>/*.jpg
            train.csv (image_path, emotion)
            merged.csv
            label_map.json
            stats.json
  4. Copy (or symlink) images by class (default copy for portability)

Usage:
  python scripts/merge_yolo_emotion.py \
      --in-root data/yolo-emotion \
      --out-root data/yolo-emotion_categorized \
      --copy

Notes:
  - Expects each split directory to contain images and _annotations.csv
  - Annotation schema: filename,width,height,class,xmin,ymin,xmax,ymax
  - Class names are normalized to lowercase
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge YOLO emotion dataset and categorize into class folders."
    )
    p.add_argument('--in-root', required=True, help='Input root with train/valid/test folders')
    p.add_argument('--out-root', required=True, help='Output root for categorized dataset')
    p.add_argument(
        '--merged-name',
        default='merged',
        help='Name for merged valid+test split folder'
    )
    p.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of symlink (default symlink)'
    )
    return p.parse_args()


def read_annotations(split_dir: Path) -> pd.DataFrame:
    csv_path = split_dir / '_annotations.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing annotations: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {'filename', 'class'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Annotations missing required columns {required_cols}: {csv_path}")
    df['class'] = df['class'].str.strip().str.lower()
    return df


def copy_or_link(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)


def categorize(
    split_name: str,
    df: pd.DataFrame,
    images_dir: Path,
    out_root: Path,
    do_copy: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    grouped = df.groupby('class')
    for cls, g in grouped:
        cls_str = str(cls)
        for _, r in g.iterrows():
            fname = str(r['filename'])
            src = images_dir / fname
            if not src.exists():
                # Skip missing
                continue
            rel_path = Path(split_name) / cls_str / fname
            dst = out_root / rel_path
            copy_or_link(src, dst, do_copy)
            rows.append({"image_path": str(rel_path), "emotion": cls_str})
    return rows


def build_label_map(classes: List[str]) -> Dict[str, int]:
    return {cls: idx for idx, cls in enumerate(sorted(classes))}


def main():
    args = parse_args()
    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    train_dir = in_root / 'train'
    valid_dir = in_root / 'valid'
    test_dir = in_root / 'test'

    train_df = read_annotations(train_dir)
    valid_df = read_annotations(valid_dir)
    test_df = read_annotations(test_dir)

    merged_df = pd.concat([valid_df, test_df], ignore_index=True)

    # Deduplicate by filename,class keeping first occurrence
    train_df = train_df.drop_duplicates(subset=['filename', 'class'])
    merged_df = merged_df.drop_duplicates(subset=['filename', 'class'])

    classes = sorted(set(train_df['class']).union(merged_df['class']))
    label_map = build_label_map(classes)

    # Categorize
    rows_train = categorize('train', train_df, train_dir, out_root, args.copy)
    rows_merged = categorize(
        args.merged_name,
        merged_df,
        valid_dir,
        out_root,
        args.copy,
    )  # valid/test images referenced by their original dirs
    # For test images we need to override path source; re-run categorize for test images
    rows_extra = categorize(args.merged_name, test_df, test_dir, out_root, args.copy)
    # rows_merged includes valid/test; rows_extra ensures any test-only paths are added

    # Combine rows for merged (deduplicate by image_path)
    seen = set()
    merged_rows_final = []
    for r in rows_merged + rows_extra:
        if r['image_path'] not in seen:
            seen.add(r['image_path'])
            merged_rows_final.append(r)

    # Write CSVs
    pd.DataFrame(rows_train).to_csv(out_root / 'train.csv', index=False)
    pd.DataFrame(merged_rows_final).to_csv(out_root / f'{args.merged_name}.csv', index=False)

    stats = {
        'splits': {
            'train': len(rows_train),
            args.merged_name: len(merged_rows_final),
        },
        'classes': classes,
        'label_map': label_map,
    }
    with open(out_root / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    with open(out_root / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print('Done.')
    print(f" Train images categorized: {len(rows_train)}")
    print(f" {args.merged_name.capitalize()} images categorized: {len(merged_rows_final)}")
    print(f" Classes: {classes}")
    print(f" Output: {out_root}")


if __name__ == '__main__':
    main()
