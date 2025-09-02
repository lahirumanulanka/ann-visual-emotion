import argparse
import csv
import json
import warnings
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit


def collect_pairs(raw_dir: Path, img_exts: list[str], filter_facial: bool = False):
    """
    Returns (relative_image_path, label) where image path is forced to:
      data/raw/EmoSet/<class>/<file>
    Scans nested subfolders and supports multiple extensions.
    """
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in img_exts}
    rows = []  # (image_path, label)
    for class_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        # scan nested too
        for img_path in (p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts):
            ann_path = img_path.with_suffix(".json")
            label = class_dir.name  # fallback: folder name
            if ann_path.exists():
                try:
                    meta = json.loads(ann_path.read_text())
                    # prefer explicit emotion in JSON if present
                    label = (meta.get("emotion") or label).strip()
                    if filter_facial and not meta.get("facial_expression"):
                        continue
                except Exception as e:
                    warnings.warn(f"Bad JSON {ann_path}: {e}; using folder label '{label}'")
            # Force relative path style: data/raw/EmoSet/<class>/<file>
            rel_path = Path("data/raw/EmoSet") / img_path.relative_to(raw_dir)
            rows.append((rel_path.as_posix(), label))
    # de-dup (just in case)
    seen, dedup = set(), []
    for p, lb in rows:
        if p not in seen:
            dedup.append((p, lb))
            seen.add(p)
    return dedup


def write_csv(out_path: Path, rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir",   default="data/raw/EmoSet", type=str,
                    help="Dataset root that has class folders (e.g., data/raw/EmoSet/amusement/*)")
    ap.add_argument("--out_dir",   default="data/processed/EmoSet_splits", type=str)
    ap.add_argument("--img_exts",  default=".jpg,.jpeg,.png", type=str,
                    help="Comma-separated list of image extensions (e.g. .jpg,.jpeg,.png,.bmp)")
    ap.add_argument("--filter_facial", action="store_true",
                    help="Keep only samples that have 'facial_expression' in JSON")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    img_exts = [e.strip() for e in args.img_exts.split(",") if e.strip()]

    rows = collect_pairs(raw_dir, img_exts, args.filter_facial)
    if not rows:
        raise SystemExit(f"No samples found in {raw_dir} with exts {img_exts}")

    # labels + mapping
    labels = sorted({lb for _, lb in rows})
    label_map = {lb: i for i, lb in enumerate(labels)}
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))

    # stratified splits
    X = [p for p, _ in rows]
    y = [label_map[lb] for _, lb in rows]

    rnd = args.seed
    test_split = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=rnd)
    train_idx, test_idx = next(test_split.split(X, y))

    # val from remaining
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    val_ratio_adj = args.val_ratio / (1 - args.test_ratio)  # portion of the remaining
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adj, random_state=rnd)
    train_idx2, val_idx2 = next(val_split.split(X_train, y_train))

    # reindex to original indices
    train_rows = [rows[train_idx[i]] for i in train_idx2]
    val_rows   = [rows[train_idx[i]] for i in val_idx2]
    test_rows  = [rows[i] for i in test_idx]

    # write CSVs
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv",   val_rows)
    write_csv(out_dir / "test.csv",  test_rows)

    # stats
    def count_by_label(rws):
        c = {}
        for _, lb in rws:
            c[lb] = c.get(lb, 0) + 1
        return dict(sorted(c.items()))
    stats = {
        "num_total": len(rows),
        "splits": {
            "train": {"n": len(train_rows), "by_label": count_by_label(train_rows)},
            "val":   {"n": len(val_rows),   "by_label": count_by_label(val_rows)},
            "test":  {"n": len(test_rows),  "by_label": count_by_label(test_rows)},
        },
        "label_map": label_map,
        "raw_dir": str(raw_dir.as_posix())
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()