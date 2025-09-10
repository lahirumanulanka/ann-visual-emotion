"""Face cropping & (optional) grayscale conversion pipeline.

Processes all images in an input root (recursively or flat) applying:
  1. Face detection (Haar cascade via OpenCV) – picks largest face.
  2. Optional margin expansion around the detected face.
  3. Optional grayscale conversion.
  4. Saves to an output root preserving /<label>/ structure inferred from filename.
  5. Generates metadata CSV + label_map.json + stats.json summarising processing.

Label Inference:
  Attempts to extract the emotion label by searching for known tokens inside the
  filename (case-insensitive). A NORMALIZE_MAP maps variants (e.g. 'scared' -> 'fearful').
  If no label can be inferred the image is skipped.

CSV Schema (if requested):
  path,label

Usage Example:
  python scripts/crop_faces.py \
      --in_root data/newDataset \
      --out_root data/newDataset_cropped_gray \
      --grayscale \
      --write_csv data/newDataset_cropped_gray/metadata.csv

Notes:
  * Requires opencv-python & Pillow.
  * Haar cascade file is resolved from cv2.data.haarcascades.
  * If no face is found you can choose to skip or fallback to full image
    (current default: skip). Adjust with --fallback_full.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
from PIL import Image


EMOTION_TOKENS = [
    "angry",
    "disgusted",
    "happy",
    "sad",
    "scared",
    "fearful",  # direct variant (just in case)
    "surprised",
    "neutral",
]

NORMALIZE_MAP = {
    "scared": "fearful",
    "fear": "fearful",
    "fearful": "fearful",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "disgust": "disgusted",
    "disgusted": "disgusted",
    "surprised": "surprised",
    "neutral": "neutral",
}


@dataclass
class ImageStats:
    total: int = 0
    with_label: int = 0
    no_label: int = 0
    processed: int = 0
    skipped_no_face: int = 0
    skipped_other: int = 0


def infer_label_from_name(name: str) -> Optional[str]:
    lower = name.lower()
    for token in EMOTION_TOKENS:
        if token in lower:
            norm = NORMALIZE_MAP.get(token, token)
            return norm
    # Also try splitting on common separators
    for part in Path(name).stem.lower().replace("-", "_").split("_"):
        if part in NORMALIZE_MAP:
            return NORMALIZE_MAP[part]
    return None


def load_cascade(custom_path: Optional[str] = None):
    if custom_path:
        cascade_path = custom_path
    else:
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load Haar cascade at {cascade_path}")
    return cascade


def detect_face(
    img_bgr,
    cascade,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )
    if len(faces) == 0:
        return None
    # Choose largest (area)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def expand_box(x: int, y: int, w: int, h: int, margin: float, width: int, height: int):
    if margin <= 0:
        return x, y, w, h
    cx = x + w / 2
    cy = y + h / 2
    half_w = w / 2 * (1 + margin)
    half_h = h / 2 * (1 + margin)
    nx1 = max(0, int(cx - half_w))
    ny1 = max(0, int(cy - half_h))
    nx2 = min(width, int(cx + half_w))
    ny2 = min(height, int(cy + half_h))
    return nx1, ny1, nx2 - nx1, ny2 - ny1


def process_image(
    path: Path,
    out_dir: Path,
    cascade,
    args,
    stats: ImageStats,
    per_label_counts: Dict[str, int],
    records: List[Tuple[str, str]],
):
    stats.total += 1
    label = infer_label_from_name(path.name)
    if not label:
        stats.no_label += 1
        return
    stats.with_label += 1

    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            stats.skipped_other += 1
            return
        h, w = img_bgr.shape[:2]
        face_box = detect_face(
            img_bgr,
            cascade,
            args.face_scale_factor,
            args.face_min_neighbors,
            args.min_size,
        )
        if face_box is None:
            if args.fallback_full:
                x, y, fw, fh = 0, 0, w, h
            else:
                stats.skipped_no_face += 1
                return
        else:
            x, y, fw, fh = face_box
        x, y, fw, fh = expand_box(x, y, fw, fh, args.margin, w, h)
        crop = img_bgr[y : y + fh, x : x + fw]
        # Convert to RGB for Pillow
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        if args.grayscale:
            pil_img = pil_img.convert("L")  # single channel

        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        out_path = label_dir / path.name
        # Ensure extension consistency (jpg)
        out_path = out_path.with_suffix(".jpg")
        pil_img.save(out_path, quality=95)

        rel_path = out_path.as_posix()
        records.append((rel_path, label))
        per_label_counts[label] = per_label_counts.get(label, 0) + 1
        stats.processed += 1
    except Exception as e:  # pragma: no cover (defensive)
        stats.skipped_other += 1
        print(f"[WARN] Failed {path}: {e}", file=sys.stderr)


def iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in root.iterdir():  # flat directory based on provided structure
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def write_csv(csv_path: Path, records: List[Tuple[str, str]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for r in records:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(
        description="Crop faces and optionally grayscale images."
    )
    parser.add_argument(
        "--in_root",
        required=True,
        help="Input root directory containing images (flat).",
    )
    parser.add_argument(
        "--out_root",
        required=True,
        help="Output root directory for cropped images.",
    )
    parser.add_argument(
        "--cascade",
        default=None,
        help="Optional custom Haar cascade path.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="Margin expansion ratio around detected face.",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=32,
        help="Minimum face size.",
    )
    parser.add_argument(
        "--face_scale_factor",
        type=float,
        default=1.1,
        help="Haar cascade scale factor.",
    )
    parser.add_argument(
        "--face_min_neighbors",
        type=int,
        default=5,
        help="Haar cascade minNeighbors.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert output crops to grayscale.",
    )
    parser.add_argument(
        "--fallback_full",
        action="store_true",
        help="If no face detected, use full image instead of skipping.",
    )
    parser.add_argument(
        "--write_csv",
        default=None,
        help="Optional path to write metadata CSV.",
    )
    parser.add_argument(
        "--no_stats",
        action="store_true",
        help="Disable writing stats & label map JSONs.",
    )

    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cascade = load_cascade(args.cascade)

    stats = ImageStats()
    per_label_counts: Dict[str, int] = {}
    records: List[Tuple[str, str]] = []

    images = list(iter_images(in_root))
    if args.limit:
        images = images[: args.limit]

    total_images = len(images)
    if total_images == 0:
        print("No images found.")
        return

    print(f"Found {total_images} images. Processing...")
    report_every = max(1, math.ceil(total_images / 20))

    for idx, img_path in enumerate(images, 1):
        process_image(img_path, out_root, cascade, args, stats, per_label_counts, records)
        if idx % report_every == 0:
            print(
                (
                    f"Progress {idx}/{total_images} | processed={stats.processed} "
                    f"no_face={stats.skipped_no_face} no_label={stats.no_label}"
                ),
                flush=True,
            )

    # Write metadata
    if args.write_csv and records:
        write_csv(Path(args.write_csv), records)
        print(f"Wrote CSV: {args.write_csv} ({len(records)} rows)")

    if not args.no_stats:
        label_map = {label: i for i, label in enumerate(sorted({r[1] for r in records}))}
        with (out_root / "label_map.json").open("w") as f:
            json.dump(label_map, f, indent=2)
        with (out_root / "stats.json").open("w") as f:
            json.dump(
                {
                    "global": asdict(stats),
                    "per_label_counts": per_label_counts,
                },
                f,
                indent=2,
            )
        print(f"Wrote stats & label_map in {out_root}")

    print(
        (
            "Done. Summary: "
            f"total={stats.total} with_label={stats.with_label} processed={stats.processed} "
            f"no_label={stats.no_label} no_face={stats.skipped_no_face} "
            f"other={stats.skipped_other}"
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
