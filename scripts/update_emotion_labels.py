"""Update emotion label names across processed split CSVs, label_map.json, and stats.json.

Example:
  python scripts/update_emotion_labels.py \
      --splits-root data/processed/EmoSet_splits \
      --rename fear:fearful --rename surprise:surprised

Actions:
  * Load train.csv, val.csv, test.csv and replace label values.
  * Recompute label_map.json (sorted by label name) unless --keep-ids given.
  * Update label_id column to match new mapping.
  * Update stats.json keys (splits.*.by_label) accordingly.
  * Optionally rename on-disk class directories if present (train/, validation/ prefixes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Rename emotion labels in existing splits.")
    p.add_argument(
        '--splits-root',
        required=True,
        help='Root folder containing train.csv, val.csv, test.csv'
    )
    p.add_argument(
        '--rename',
        action='append',
        default=[],
        help='Rename mapping old:new (can repeat)'
    )
    p.add_argument(
        '--keep-ids',
        action='store_true',
        help='Keep existing numeric ids (only remap names)'
    )
    p.add_argument('--dry-run', action='store_true', help='Show planned changes without writing')
    return p.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]):
    df['label'] = df['label'].map(lambda x: mapping.get(x, x))
    return df


def update_label_ids(df: pd.DataFrame, label_map: Dict[str, int]):
    df['label_id'] = df['label'].map(label_map)
    return df


def maybe_rename_dirs(root: Path, mapping: Dict[str, str]):
    # Rename directories under train/, validation/, test/ (if exist)
    for split_dir_name in ['train', 'validation', 'val', 'test']:
        d = root / split_dir_name
        if not d.exists() or not d.is_dir():
            continue
        for old, new in mapping.items():
            if old == new:
                continue
            src_dir = d / old
            dst_dir = d / new
            if src_dir.exists() and not dst_dir.exists():
                src_dir.rename(dst_dir)


def main():
    args = parse_args()
    root = Path(args.splits_root).resolve()
    mapping = {}
    for item in args.rename:
        if ':' not in item:
            raise SystemExit(f"Bad --rename value: {item}")
        old, new = item.split(':', 1)
        mapping[old.strip()] = new.strip()
    if not mapping:
        raise SystemExit("No rename mappings provided.")

    train_csv = root / 'train.csv'
    val_csv = root / 'val.csv'
    test_csv = root / 'test.csv'
    label_map_json = root / 'label_map.json'
    stats_json = root / 'stats.json'

    train_df = load_csv(train_csv)
    val_df = load_csv(val_csv)
    test_df = load_csv(test_csv)

    # Apply rename
    for df in (train_df, val_df, test_df):
        apply_mapping(df, mapping)

    # Build new label map
    if args.keep_ids:
        # Load existing and adjust keys
        with open(label_map_json) as f:
            old_map = json.load(f)
        # Reverse to id->name, then rebuild preserving ids where possible
        id_to_name = {v: k for k, v in old_map.items()}
        for k, v in mapping.items():
            for id_, name in list(id_to_name.items()):
                if name == k:
                    id_to_name[id_] = v
        ordered = sorted(((i, n) for i, n in id_to_name.items()), key=lambda x: x[0])
        label_map = {name: idx for idx, name in ordered}
    else:
        labels = sorted(set(train_df['label']) | set(val_df['label']) | set(test_df['label']))
        label_map = {lbl: i for i, lbl in enumerate(labels)}

    for df in (train_df, val_df, test_df):
        update_label_ids(df, label_map)

    # Update stats.json
    with open(stats_json) as f:
        stats = json.load(f)
    for split_name in ['train', 'val', 'test']:
        if split_name in stats.get('splits', {}):
            by_label = stats['splits'][split_name].get('by_label', {})
            new_by_label = {}
            for k, v in by_label.items():
                new_by_label[mapping.get(k, k)] = v
            stats['splits'][split_name]['by_label'] = new_by_label

    if args.dry_run:
        print("--- Dry Run ---")
        print("Mapping:", mapping)
        print("New label_map:", label_map)
        print(train_df.head())
        return

    # Write outputs
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    with open(label_map_json, 'w') as f:
        json.dump(label_map, f, indent=2)
    with open(stats_json, 'w') as f:
        json.dump(stats, f, indent=2)

    # Rename directories (best-effort)
    maybe_rename_dirs(root, mapping)

    print("Updates complete.")
    print("Mapping applied:", mapping)
    print("Label map:", label_map)


if __name__ == '__main__':
    main()
