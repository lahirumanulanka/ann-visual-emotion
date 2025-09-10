"""Regenerate full EmoSet train/val/test splits from raw class folders.

Source: data/raw/EmoSet/<class>/*.jpg
Target: data/processed/EmoSet_splits/{train,val,test}/<class>/*.jpg (symlinks by default)

CSV schema: filepath,label,label_id,split

Usage:
  python scripts/regenerate_emoset_splits.py \
      --raw-root data/raw/EmoSet \
      --out-root data/processed/EmoSet_splits \
      --val-ratio 0.1 --test-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate EmoSet splits from raw data.")
    p.add_argument('--raw-root', required=True, help='Root raw EmoSet directory')
    p.add_argument('--out-root', required=True, help='Output splits root (will be updated)')
    p.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    p.add_argument('--test-ratio', type=float, default=0.1, help='Test ratio')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--copy', action='store_true', help='Copy files instead of symlink')
    p.add_argument('--verify', action='store_true', help='Open images to check for corruption')
    return p.parse_args()


def collect(raw_root: Path) -> Dict[str, List[Path]]:
    data: Dict[str, List[Path]] = {}
    for d in sorted(raw_root.iterdir()):
        if d.is_dir() and not d.name.startswith('.'):
            files = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
            if files:
                data[d.name] = files
    return data


def verify_images(paths: List[Path]) -> List[Path]:
    good = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            good.append(p)
        except Exception:
            pass
    return good


def place(src: Path, dst: Path, do_copy: bool):
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


def stratified_split(files: List[Path], val_ratio: float, test_ratio: float) -> Dict[str, List[Path]]:
    n = len(files)
    test_count = max(1, int(n * test_ratio)) if n > 5 else max(1, int(n * test_ratio))
    val_count = max(1, int(n * val_ratio)) if n > 5 else max(1, int(n * val_ratio))
    test_files = files[:test_count]
    val_files = files[test_count:test_count + val_count]
    train_files = files[test_count + val_count:]
    return {'train': train_files, 'val': val_files, 'test': test_files}


def main():
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()
    random.seed(args.seed)

    class_files = collect(raw_root)
    if not class_files:
        raise SystemExit("No class folders found in raw root.")

    # Normalize class names (lowercase)
    norm_map = {cls: cls.lower() for cls in class_files}
    if 'fear' in norm_map:
        norm_map['fear'] = 'fearful'
    if 'surprise' in norm_map:
        norm_map['surprise'] = 'surprised'

    # Prepare containers
    rows = {'train': [], 'val': [], 'test': []}
    per_class_counts = {split: {} for split in rows}

    # Clear existing split dirs (keep CSV backups optional)
    for sub in ['train', 'val', 'validation', 'test']:
        dir_path = out_root / sub
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)

    # Build new splits
    for original_cls, files in class_files.items():
        cls = norm_map[original_cls]
        files = sorted(files)
        if args.verify:
            files = verify_images(files)
        random.shuffle(files)
        parts = stratified_split(files, args.val_ratio, args.test_ratio)
        for split, split_files in parts.items():
            per_class_counts[split][cls] = len(split_files)
            for f in split_files:
                rel_path = Path(split) / cls / f.name
                place(f, out_root / rel_path, args.copy)
                rows[split].append({
                    'filepath': str(rel_path),
                    'label': cls,
                    'split': split,
                })

    # Build label map (sorted alphabetically)
    labels = sorted({r['label'] for split_rows in rows.values() for r in split_rows})
    label_map = {lbl: i for i, lbl in enumerate(labels)}
    for split in rows:
        for r in rows[split]:
            r['label_id'] = label_map[r['label']]

    # Write CSVs
    for split in ['train', 'val', 'test']:
        pd.DataFrame(rows[split]).to_csv(out_root / f'{split}.csv', index=False)

    # Stats
    stats = {
        'dataset': 'EmoSet_Full',
        'splits': {
            split: {
                'total': len(rows[split]),
                'by_label': per_class_counts[split]
            } for split in ['train', 'val', 'test']
        },
        'ratios': {
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio
        },
        'seed': args.seed
    }

    with open(out_root / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    with open(out_root / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print('Regeneration complete:')
    print(' Label map:', label_map)
    for split in ['train', 'val', 'test']:
        print(f" {split}: {stats['splits'][split]['total']}")


if __name__ == '__main__':
    main()
