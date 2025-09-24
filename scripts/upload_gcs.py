"""Utility to upload the local data/processed folder to a Google Cloud Storage bucket.

Usage (after installing optional dependency group):
  pip install -e .[gcs]
  python scripts/upload_gcs.py --bucket my-bucket-name \
      --prefix datasets/emotion_project --delete-extra

Authentication:
  1. Easiest: gcloud auth application-default login
  2. Or set GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service account JSON key.

Safety:
  By default we only upload/update changed files. Pass --delete-extra to remove remote
  objects under the chosen prefix that no longer exist locally.
"""
from __future__ import annotations

import argparse
import hashlib
import mimetypes
from pathlib import Path
from typing import Iterable

try:  # Lazy env loading (no new dependency if user doesn't have python-dotenv)
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

from google.cloud import storage

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

if load_dotenv:  # pragma: no cover
    load_dotenv()  # loads .env if present


def iter_files(base: Path) -> Iterable[Path]:
    for p in base.rglob("*"):
        if p.is_file():
            yield p


def md5_hex(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:  # noqa: S324 (not for security, just change detection)
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_directory(
    bucket_name: str,
    prefix: str,
    delete_extra: bool = False,
    dry_run: bool = False,
) -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Local directory not found: {DATA_DIR}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Build local manifest
    local_entries = {}
    for file_path in iter_files(DATA_DIR):
        rel = file_path.relative_to(DATA_DIR)
        blob_name = f"{prefix.rstrip('/')}/{rel.as_posix()}" if prefix else rel.as_posix()
        local_entries[blob_name] = {
            "path": file_path,
            "md5": md5_hex(file_path),
        }

    # Fetch existing remote blobs under prefix
    existing = {b.name: b for b in client.list_blobs(bucket, prefix=prefix or None)}

    uploads = []
    skips = []
    for blob_name, meta in local_entries.items():
        blob = existing.get(blob_name)
        if blob and blob.md5_hash and blob.md5_hash == meta["md5"]:  # already up-to-date
            skips.append(blob_name)
            continue
        uploads.append((blob_name, meta))

    deletions = []
    if delete_extra:
        local_keys = set(local_entries.keys())
        for name in existing:
            if name.startswith(prefix) and name not in local_keys:
                deletions.append(name)

    print(f"Planned uploads: {len(uploads)} (skipping {len(skips)} unchanged)")
    if delete_extra:
        print(f"Planned deletions: {len(deletions)}")

    if dry_run:
        print("Dry run enabled; no changes performed.")
        return

    for blob_name, meta in uploads:
        blob = bucket.blob(blob_name)
        ctype, _ = mimetypes.guess_type(blob_name)
        if ctype:
            blob.content_type = ctype
        print(f"Uploading {meta['path']} -> gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(str(meta["path"]))

    for name in deletions:
        print(f"Deleting gs://{bucket_name}/{name}")
        bucket.blob(name).delete()

    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload processed data directory to GCS")
    p.add_argument("--bucket", required=True, help="GCS bucket name")
    p.add_argument("--prefix", default="", help="Object name prefix inside the bucket")
    p.add_argument(
        "--delete-extra",
        action="store_true",
        help="Delete remote objects not present locally",
    )
    p.add_argument("--dry-run", action="store_true", help="Only show what would change")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    upload_directory(
        bucket_name=args.bucket,
        prefix=args.prefix,
        delete_extra=args.delete_extra,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
