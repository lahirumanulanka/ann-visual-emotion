"""Utility script to organize images into class folders based on a CSV or filename heuristics.

Enhancements:
1. If the annotation CSV is missing, you can auto-generate one from filenames (emotion word detection).
2. CLI arguments instead of hard‑coded paths.
3. Optionally skip files whose labels can't be inferred.
4. Generates a summary counts JSON & a cleaned CSV.

Expected CSV columns (case‑insensitive): filename, class
"""

from __future__ import annotations

import os
import re
import json
import shutil
import argparse
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

EMOTION_KEYWORDS = [
    "anger", "angry", "disgust", "disgusted", "fear", "scared", "happy", "happiness",
    "sad", "sadness", "surprise", "surprised", "neutral"
]

EMOTION_NORMALIZE_MAP = {
    "angry": "anger",
    "disgusted": "disgust",
    "scared": "fear",
    "happiness": "happy",
    "sadness": "sad",
    "surprised": "surprise",
}


@dataclass
class Config:
    csv_path: str
    images_dir: str
    output_dir: str
    auto_generate: bool = False
    skip_unknown: bool = True
    generated_csv_name: str = "auto_annotations.csv"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Organize images into class folders using CSV or filename heuristics.")
    p.add_argument("--csv", dest="csv_path", default="../data/newDataSet/1a_afinal_cleaned_annotations.csv", help="Path to annotations CSV.")
    p.add_argument("--images", dest="images_dir", default="../data/newDataSet", help="Directory with source images.")
    p.add_argument("--out", dest="output_dir", default="classified_images", help="Output root directory for class subfolders.")
    p.add_argument("--auto-generate", action="store_true", help="If set, generate CSV from filenames when CSV missing.")
    p.add_argument("--keep-unknown", action="store_true", help="If set, keep images whose label can't be inferred (label = 'unknown').")
    p.add_argument("--generated-csv-name", default="auto_annotations.csv", help="Filename to use when auto-generating CSV.")
    args = p.parse_args()
    return Config(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        auto_generate=args.auto_generate,
        skip_unknown=not args.keep_unknown,
        generated_csv_name=args.generated_csv_name,
    )


def detect_emotion_from_filename(fname: str) -> Optional[str]:
    low = fname.lower()
    for kw in EMOTION_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", low):
            return EMOTION_NORMALIZE_MAP.get(kw, kw)
    # fallback: look for partial tokens like 'image_angry_' style
    for kw in EMOTION_KEYWORDS:
        if kw in low:
            return EMOTION_NORMALIZE_MAP.get(kw, kw)
    return None


def auto_generate_csv(cfg: Config) -> str:
    files = [f for f in os.listdir(cfg.images_dir) if os.path.isfile(os.path.join(cfg.images_dir, f))]
    rows = []
    unknown_count = 0
    for f in files:
        label = detect_emotion_from_filename(f)
        if label is None:
            if cfg.skip_unknown:
                unknown_count += 1
                continue
            label = "unknown"
        rows.append({"filename": f, "class": label})
    if not rows:
        raise RuntimeError("No annotations could be created from filenames.")
    out_csv = os.path.join(os.path.dirname(cfg.csv_path), cfg.generated_csv_name)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✓ Auto-generated CSV: {out_csv} (rows={len(rows)}, skipped unknown={unknown_count})")
    return out_csv


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    if "filename" not in df.columns or all(col not in df.columns for col in ["class", "label", "emotion"]):
        raise ValueError("CSV must contain filename and class/label/emotion column.")
    # unify label column to 'class'
    if "class" not in df.columns:
        for c in ["label", "emotion"]:
            if c in df.columns:
                df["class"] = df[c]
                break
    df["class"] = df["class"].astype(str).str.strip().str.lower()
    df["class"] = df["class"].map(lambda x: EMOTION_NORMALIZE_MAP.get(x, x))
    return df[["filename", "class"]]


def copy_images(df: pd.DataFrame, cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    missing = 0
    for idx, row in df.iterrows():
        filename = row["filename"]
        label = str(row["class"]).strip().lower()
        src_path = os.path.join(cfg.images_dir, filename)
        dst_dir = os.path.join(cfg.output_dir, label)
        dst_path = os.path.join(dst_dir, filename)
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            missing += 1
            if missing <= 10:
                print(f"⚠️ Missing file: {src_path}")
    if missing:
        print(f"⚠️ Total missing files: {missing}")
    print(f"✅ Images have been categorized into folders inside: {cfg.output_dir}")


def write_summary(df: pd.DataFrame, cfg: Config):
    counts = df["class"].value_counts().to_dict()
    summary = {
        "total_images": int(len(df)),
        "classes": counts,
        "num_classes": int(len(counts)),
        "source_images_dir": os.path.abspath(cfg.images_dir),
        "output_dir": os.path.abspath(cfg.output_dir),
        "csv_used": os.path.abspath(cfg.csv_path),
    }
    out_path = os.path.join(cfg.output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Wrote summary: {out_path}")


def main():
    cfg = parse_args()
    if not os.path.exists(cfg.csv_path):
        if cfg.auto_generate:
            cfg.csv_path = auto_generate_csv(cfg)
        else:
            raise FileNotFoundError(f"CSV not found: {cfg.csv_path}. Use --auto-generate to build from filenames.")

    df = pd.read_csv(cfg.csv_path)
    df = normalize_columns(df)

    print(f"Loaded annotations: {cfg.csv_path} (rows={len(df)})")
    copy_images(df, cfg)
    # Write a cleaned copy of the annotations next to output for reproducibility
    cleaned_csv = os.path.join(cfg.output_dir, "annotations_cleaned.csv")
    df.to_csv(cleaned_csv, index=False)
    print(f"✓ Wrote cleaned annotations: {cleaned_csv}")
    write_summary(df, cfg)


if __name__ == "__main__":
    main()