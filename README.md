# ann-visual-emotion

Utilities for training and deploying an emotion recognition model (ANN/CNN) and associated tooling.

## Processed Data Upload to GCS

Upload the `data/processed` directory to the bucket `emotion_face_dataset`.

### Setup
1. Copy environment template:
	```bash
	cp .env.example .env
	```
2. (Optional) Add a service account credentials file path to `.env`:
	```bash
	GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/service-account.json
	```
3. (If not already) install optional dependency group:
	```bash
	pip install -e .[gcs]
	```

### Dry Run
```bash
python scripts/upload_gcs.py --bucket "$GCS_BUCKET" --prefix processed-data --dry-run
```

### Upload
```bash
python scripts/upload_gcs.py --bucket "$GCS_BUCKET" --prefix processed-data
```

Or via VS Code Task (Command Palette > Run Task): `upload-processed-gcs`.

### Delete Remote Extras (Dangerous)
```bash
python scripts/upload_gcs.py --bucket "$GCS_BUCKET" --prefix processed-data --delete-extra
```

Always pair `--delete-extra` with a prior `--dry-run` to confirm impact.

### Notes
- `.env` is git-ignored; keep tokens/keys there.
- Prefer service account JSON over embedding tokens whenever possible.
- MD5 comparison avoids uploading unchanged files.