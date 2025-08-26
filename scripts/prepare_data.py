import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image  # only used when --probe-images true
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Preferred FER7 label order (keeps IDs stable across runs)
PREFERRED_LABEL_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def list_images_in_dir(d: Path) -> List[Path]:
    return [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]


def discover_labels(split_root: Path) -> List[str]:
    labels = sorted([p.name for p in split_root.iterdir() if p.is_dir()])
    return labels


def stable_label_map(all_labels: List[str]) -> Dict[str, int]:
    # Use preferred order first; then append any unexpected labels (alphabetical)
    ordered = [l for l in PREFERRED_LABEL_ORDER if l in all_labels]
    extras = sorted([l for l in all_labels if l not in ordered])
    final = ordered + extras
    return {lbl: idx for idx, lbl in enumerate(final)}


def gather_rows(split_name: str,
                split_dir: Path,
                label_map: Dict[str, int],
                base_dir: Path,
                probe_images: bool) -> Tuple[List[Dict], Dict[str, int]]:
    rows = []
    per_label_counts = {lbl: 0 for lbl in label_map.keys()}
    for lbl in label_map.keys():
        class_dir = split_dir / lbl
        if not class_dir.exists():
            continue
        for img_path in list_images_in_dir(class_dir):
            row = {
                "filepath": str(img_path.relative_to(base_dir).as_posix()),
                "label": lbl,
                "label_id": label_map[lbl],
                "split": split_name,
            }
            if probe_images:
                w = h = None
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                except Exception:
                    pass
                row["width"] = w
                row["height"] = h
            rows.append(row)
            per_label_counts[lbl] += 1
    return rows, per_label_counts


def write_csv(rows: List[Dict], out_csv: Path, include_dims: bool):
    import csv
    cols = ["filepath", "label", "label_id", "split"]
    if include_dims:
        cols += ["width", "height"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, default="raw",
                    help="Root folder that contains train/ validation/ [test/].")
    ap.add_argument("--out-dir", type=str, default="processed/EmoSet_splits",
                    help="Where to write CSVs and JSONs.")
    ap.add_argument("--test-fraction", type=float, default=0.2,
                    help="When raw/test is missing, take this fraction from validation to form test.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    ap.add_argument("--probe-images", type=lambda x: str(x).lower() in {"1","true","yes","y"},
                    default=False, help="Open images to record width/height (slower).")
    args = ap.parse_args()

    random.seed(args.seed)

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    train_dir = raw_dir / "train"
    val_dir = raw_dir / "validation"
    test_dir = raw_dir / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(f"[ERROR] Expected '{train_dir}' and '{val_dir}' to exist.")

    # Discover union of labels from train + validation (+ test if present)
    labels = set()
    for d in [train_dir, val_dir, test_dir] if test_dir.exists() else [train_dir, val_dir]:
        if d.exists():
            labels.update(discover_labels(d))
    labels = sorted(labels)

    label_map = stable_label_map(labels)

    # If test exists → use it. Else, split validation into val/test by fraction.
    have_test = test_dir.exists()

    # Collect rows
    all_rows = []
    stats = {
        "num_total": 0,
        "splits": {}
    }

    # Train
    tr_rows, tr_counts = gather_rows("train", train_dir, label_map, base_dir=raw_dir, probe_images=args.probe_images)
    all_rows.extend(tr_rows)
    stats["splits"]["train"] = {"n": len(tr_rows), "by_label": tr_counts}

    if have_test:
        # Validation rows as-is
        va_rows, va_counts = gather_rows("val", val_dir, label_map, base_dir=raw_dir, probe_images=args.probe_images)
        all_rows.extend(va_rows)
        stats["splits"]["val"] = {"n": len(va_rows), "by_label": va_counts}

        # Test rows as-is
        te_rows, te_counts = gather_rows("test", test_dir, label_map, base_dir=raw_dir, probe_images=args.probe_images)
        all_rows.extend(te_rows)
        stats["splits"]["test"] = {"n": len(te_rows), "by_label": te_counts}

        # Write CSVs
        write_csv(tr_rows, out_dir / "train.csv", include_dims=args.probe_images)
        write_csv(va_rows, out_dir / "val.csv", include_dims=args.probe_images)
        write_csv(te_rows, out_dir / "test.csv", include_dims=args.probe_images)
    else:
        # Need to create test from validation
        # Build per-label lists from validation, then split by fraction
        per_label_paths = {lbl: [] for lbl in label_map.keys()}
        for lbl in label_map.keys():
            class_dir = val_dir / lbl
            if class_dir.exists():
                per_label_paths[lbl] = list_images_in_dir(class_dir)

        val_rows, test_rows = [], []
        va_counts = {lbl: 0 for lbl in label_map.keys()}
        te_counts = {lbl: 0 for lbl in label_map.keys()}

        for lbl, paths in per_label_paths.items():
            if not paths:
                continue
            random.shuffle(paths)
            k = int(round(len(paths) * args.test_fraction))
            test_split = paths[:k]
            val_split = paths[k:]

            # Build rows for each
            for img_path in val_split:
                row = {
                    "filepath": str(img_path.relative_to(raw_dir).as_posix()),
                    "label": lbl,
                    "label_id": label_map[lbl],
                    "split": "val",
                }
                if args.probe_images and PIL_AVAILABLE:
                    try:
                        with Image.open(img_path) as im:
                            row["width"], row["height"] = im.size
                    except Exception:
                        row["width"] = row["height"] = None
                val_rows.append(row)
                va_counts[lbl] += 1

            for img_path in test_split:
                row = {
                    "filepath": str(img_path.relative_to(raw_dir).as_posix()),
                    "label": lbl,
                    "label_id": label_map[lbl],
                    "split": "test",
                }
                if args.probe_images and PIL_AVAILABLE:
                    try:
                        with Image.open(img_path) as im:
                            row["width"], row["height"] = im.size
                    except Exception:
                        row["width"] = row["height"] = None
                test_rows.append(row)
                te_counts[lbl] += 1

        all_rows.extend(val_rows)
        all_rows.extend(test_rows)
        stats["splits"]["val"] = {"n": len(val_rows), "by_label": va_counts}
        stats["splits"]["test"] = {"n": len(test_rows), "by_label": te_counts}

        # Write CSVs
        write_csv(tr_rows, out_dir / "train.csv", include_dims=args.probe_images)
        write_csv(val_rows, out_dir / "val.csv", include_dims=args.probe_images)
        write_csv(test_rows, out_dir / "test.csv", include_dims=args.probe_images)

    # Totals
    stats["num_total"] = sum(s["n"] for s in stats["splits"].values())

    # Save label_map & stats
    with (out_dir / "label_map.json").open("w") as f:
        json.dump(label_map, f, indent=2)
    with (out_dir / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    # Pretty console summary
    print("\n=== Prepare Data: Summary ===")
    print(f"Raw dir: {raw_dir}")
    print(f"Out dir: {out_dir}")
    print(f"Labels (string → id): {label_map}")
    for split_name in ["train", "val", "test"]:
        s = stats["splits"].get(split_name)
        if not s: 
            continue
        print(f"- {split_name}: {s['n']} images")
        counts = {k: v for k, v in s["by_label"].items() if v > 0}
        print(f"  by_label: {counts}")
    print(f"Total images: {stats['num_total']}")
    print("Files written: train.csv, val.csv, test.csv, label_map.json, stats.json")


if __name__ == "__main__":
    main()