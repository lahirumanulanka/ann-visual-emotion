#!/usr/bin/env python3
"""
Convert a dataset of images to grayscale while preserving folder structure.

Usage:
  python grayscale_dataset.py --in_root /path/TO/DATASET --out_root /path/TO/DATASET_gray
  # Save as single-channel (L) or 3-channel gray (RGB) for models expecting 3 channels:
  python grayscale_dataset.py --in_root DATASET --out_root DATASET_gray --mode L
  python grayscale_dataset.py --in_root DATASET --out_root DATASET_gray --mode RGB
"""

import argparse
import concurrent.futures as cf
import os
from pathlib import Path
from typing import Iterable

from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def to_grayscale(in_path: Path, out_path: Path, mode: str = "L", quality: int = 95):
    """
    mode = 'L'  -> single-channel grayscale (1 channel)
    mode = 'RGB'-> 3-channel grayscale (replicated to R,G,B)
    """
    try:
        with Image.open(in_path) as im:
            im = im.convert("L")  # make grayscale first
            if mode.upper() == "RGB":
                im = im.convert("RGB")  # replicate to 3 channels

            ensure_parent(out_path)

            # Keep original format/extension
            save_kwargs = {}
            ext = out_path.suffix.lower()
            if ext in {".jpg", ".jpeg"}:
                save_kwargs.update({"quality": quality, "subsampling": 1, "optimize": True})

            im.save(out_path, **save_kwargs)
        return True, None
    except Exception as e:
        return False, f"{in_path}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=Path, required=True, help="Root folder with class subfolders (e.g., DATASET)")
    ap.add_argument("--out_root", type=Path, required=True, help="Output root (e.g., DATASET_gray)")
    ap.add_argument("--mode", choices=["L", "RGB"], default="RGB",
                    help="L = single-channel; RGB = 3-channel gray (good for models pre-trained on RGB)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite if output exists")
    args = ap.parse_args()

    in_root: Path = args.in_root.resolve()
    out_root: Path = args.out_root.resolve()

    if not in_root.exists():
        raise SystemExit(f"Input root not found: {in_root}")

    imgs = list(iter_images(in_root))
    if not imgs:
        raise SystemExit(f"No images found under {in_root}")

    print(f"Converting {len(imgs)} images from {in_root} -> {out_root} (mode={args.mode})")

    tasks = []
    errors = []

    def process(p: Path):
        rel = p.relative_to(in_root)
        out_path = out_root / rel
        if out_path.exists() and not args.overwrite:
            return True, None
        ok, err = to_grayscale(p, out_path, mode=args.mode)
        return ok, err

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ok, err in ex.map(process, imgs):
            if not ok and err:
                errors.append(err)

    print(f"Done. Wrote to {out_root}. Errors: {len(errors)}")
    if errors:
        print("\nSample errors:")
        for e in errors[:20]:
            print(" -", e)

if __name__ == "__main__":
    main()