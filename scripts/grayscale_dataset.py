"""Convert an image dataset to GRAY or RGB while preserving folder structure.

Usage Examples:
    python scripts/grayscale_dataset.py \
            --in_root data/DATASET \
            --out_root data/DATASET_gray \
            --mode GRAY

    python scripts/grayscale_dataset.py \
            --in_root data/DATASET \
            --out_root data/DATASET_rgb \
            --mode RGB

Flags:
    --overwrite  Overwrite existing outputs
    --limit N    Process only first N images (debug)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image  # pillow

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    p = argparse.ArgumentParser(description="Convert image dataset to GRAY or RGB.")
    p.add_argument('--in_root', required=True, help='Input root directory containing class folders')
    p.add_argument('--out_root', required=True, help='Output root directory')
    p.add_argument('--mode', required=True, choices=['GRAY', 'RGB'], help='Target color mode')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing converted files')
    p.add_argument('--limit', type=int, default=0, help='Limit number of images (0 = all)')
    p.add_argument('--progress-every', type=int, default=200, help='Progress print frequency')
    return p.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def convert_image(src: Path, dst: Path, mode: str, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    try:
        with Image.open(src) as im:
            if mode == 'GRAY':
                im_conv = im.convert('L')
            else:
                im_conv = im.convert('RGB')
            dst.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {}
            if dst.suffix.lower() in {'.jpg', '.jpeg'}:
                save_kwargs['quality'] = 95
            im_conv.save(dst, **save_kwargs)
        return True
    except Exception as e:
        print(f"Error converting {src}: {e}")
        return False


def main():
    args = parse_args()
    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not in_root.exists():
        print(f"Input root does not exist: {in_root}")
        sys.exit(1)

    images = list(iter_images(in_root))
    total = len(images)
    if total == 0:
        print("No images found.")
        return
    limit = args.limit if args.limit > 0 else total
    processed = converted = skipped = 0

    print(f"Converting {limit}/{total} images -> mode {args.mode}")
    print(f"Input: {in_root}\nOutput: {out_root}\nOverwrite: {args.overwrite}")

    for idx, img_path in enumerate(images[:limit], start=1):
        rel = img_path.relative_to(in_root)
        dst = out_root / rel
        changed = convert_image(img_path, dst, args.mode, args.overwrite)
        processed += 1
        if changed:
            converted += 1
        else:
            skipped += 1
        if idx % args.progress_every == 0:
            print(f"Progress {idx}/{limit} converted={converted} skipped={skipped}")

    print("\nDone:")
    print(f"Processed: {processed}")
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    print(f"Output root: {out_root}")


if __name__ == '__main__':
    main()
