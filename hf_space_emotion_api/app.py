import io
import json
import logging
import os
from typing import Dict

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)

# Environment variables (configure in Space Settings)
MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")
LABEL_MAP_ENV = os.getenv("LABEL_MAP_JSON")
DEFAULT_LABEL_MAP = {"angry": 0, "fearful": 1, "happy": 2, "neutral": 3, "sad": 4, "surprised": 5}
PROVIDERS = [
    p.strip()
    for p in os.getenv("ORT_PROVIDERS", "CPUExecutionProvider").split(",")
    if p.strip()
]


def _load_label_map(env_value: str | None) -> Dict[str, int]:
    """Load label map from either a JSON string or a file path. Fallback to default."""
    if not env_value:
        logging.info("LABEL_MAP_JSON not set; using default label map")
        return DEFAULT_LABEL_MAP
    # If it's a file path and exists, load from file
    if os.path.isfile(env_value):
        try:
            with open(env_value, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info("Loaded label map from file: %s", env_value)
            return data
        except Exception as exc:
            logging.warning(
                "Failed to load label map from file '%s': %s; falling back to default",
                env_value,
                exc,
            )
            return DEFAULT_LABEL_MAP
    # Otherwise, try to parse as inline JSON
    try:
        data = json.loads(env_value)
        logging.info("Loaded label map from inline JSON")
        return data
    except Exception as exc:
        logging.warning("Failed to parse LABEL_MAP_JSON as JSON: %s; falling back to default", exc)
        return DEFAULT_LABEL_MAP

_label_map: Dict[str, int] = _load_label_map(LABEL_MAP_ENV)
_index_to_label = {v: k for k, v in _label_map.items()}

# Initialize ONNX session
try:
    session = ort.InferenceSession(MODEL_PATH, providers=PROVIDERS)
    first_input = session.get_inputs()[0]
    input_name = first_input.name
    input_shape = first_input.shape  # usually [N, C, H, W]
    logging.info("Model input name=%s shape=%s", input_name, input_shape)
except Exception as exc:
    raise RuntimeError(f"Failed to initialize ONNX Runtime: {exc}") from exc

app = FastAPI(title="Emotion ONNX API")

# Module-level FastAPI param to avoid function call in default signature
FILE_PARAM = File(description="Image file")


def _extract_chw(shape, default_c=3, default_h=224, default_w=224):
    # shape is typically [N, C, H, W] but may contain strings for dynamic dims
    def _safe(v, default):
        if isinstance(v, int):
            return v
        try:
            return int(v)
        except Exception:
            return default

    c = _safe(shape[1] if len(shape) > 1 else None, default_c)
    h = _safe(shape[2] if len(shape) > 2 else None, default_h)
    w = _safe(shape[3] if len(shape) > 3 else None, default_w)
    return c, h, w


MODEL_C, MODEL_H, MODEL_W = _extract_chw(input_shape)
logging.info("Using preprocessor to (C,H,W)=(%s,%s,%s)", MODEL_C, MODEL_H, MODEL_W)

# Preprocess configuration via env vars
CENTER_CROP = os.getenv("CENTER_CROP", "true").lower() in {"1", "true", "yes", "y"}
INPUT_SCALE_MODE = os.getenv("INPUT_SCALE", "0_1").lower()  # '0_1' or '0_255'
CHANNEL_ORDER = os.getenv("CHANNEL_ORDER", "RGB").upper()  # 'RGB' or 'BGR'

def _parse_csv_floats(val: str | None, expected_len: int, defaults: list[float]) -> np.ndarray:
    if not val:
        return np.array(defaults, dtype=np.float32)
    try:
        parts = [float(x.strip()) for x in val.split(",")]
        if len(parts) == 1 and expected_len > 1:
            parts = parts * expected_len
        if len(parts) != expected_len:
            logging.warning(
                "Provided values length (%d) != expected (%d); using defaults",
                len(parts), expected_len,
            )
            return np.array(defaults, dtype=np.float32)
        return np.array(parts, dtype=np.float32)
    except Exception as exc:
        logging.warning("Failed to parse floats '%s': %s; using defaults", val, exc)
        return np.array(defaults, dtype=np.float32)


if MODEL_C == 3:
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
else:
    default_mean = [0.5]
    default_std = [0.5]

NORM_MEAN = _parse_csv_floats(os.getenv("NORM_MEAN"), MODEL_C, default_mean)
NORM_STD = _parse_csv_floats(os.getenv("NORM_STD"), MODEL_C, default_std)
logging.info(
    "Preprocess config: CENTER_CROP=%s INPUT_SCALE=%s CHANNEL_ORDER=%s MEAN=%s STD=%s",
    CENTER_CROP, INPUT_SCALE_MODE, CHANNEL_ORDER, NORM_MEAN.tolist(), NORM_STD.tolist(),
)


def preprocess(img: Image.Image, c: int, h: int, w: int) -> np.ndarray:
    # Convert to correct mode first
    if c == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    # Optional center square crop to focus subject and preserve aspect ratio
    if CENTER_CROP:
        iw, ih = img.size
        side = min(iw, ih)
        left = (iw - side) // 2
        top = (ih - side) // 2
        img = img.crop((left, top, left + side, top + side))

    # Resize to expected (W, H)
    img = img.resize((w, h))

    # To numpy array
    arr = np.array(img, dtype=np.float32)

    # Adjust channel order if requested (only relevant when c==3)
    if c == 3 and CHANNEL_ORDER == "BGR":
        # HWC RGB -> HWC BGR
        arr = arr[:, :, ::-1]

    # Scale to chosen input range
    if INPUT_SCALE_MODE == "0_1":
        arr = arr / 255.0
    elif INPUT_SCALE_MODE == "0_255":
        # keep as 0..255
        pass
    else:
        logging.warning("Unknown INPUT_SCALE='%s'; defaulting to 0_1", INPUT_SCALE_MODE)
        arr = arr / 255.0

    # Normalize per-channel: (x - mean) / std
    if c == 1:
        arr = (arr - NORM_MEAN[0]) / NORM_STD[0]
        arr = arr[None, None, :, :]
    else:
        # arr HWC -> NCHW with normalization
        # Broadcast MEAN/STD over H,W
        arr = ((arr - NORM_MEAN) / NORM_STD).transpose(2, 0, 1)
        arr = arr[None, :, :, :]

    return arr


@app.post("/predict")
async def predict(file: UploadFile = FILE_PARAM):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image upload") from exc

    # Preprocess to match model input shape
    inp = preprocess(image, MODEL_C, MODEL_H, MODEL_W)

    # Inference
    try:
        outputs = session.run(None, {input_name: inp})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {exc}") from exc

    # Assume single output: logits or probabilities shape (1, C)
    out = outputs[0]
    # Convert to numpy array if possible (handle OrtValue, list, etc.)
    try:
        arr = np.asarray(out)
    except Exception:
        # Best effort fallback: if it's a list-like of numbers
        arr = np.array(out)

    if arr.ndim != 2 or arr.shape[0] != 1:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected output shape: {getattr(arr, 'shape', None)}",
        )

    logits = arr[0]
    # Determine if output is already a probability distribution
    # Criteria: all values in [0,1] and sum approximately 1
    if (
        np.all((logits >= 0.0) & (logits <= 1.0))
        and 0.98 <= float(np.sum(logits)) <= 1.02
    ):
        probs = logits.astype(np.float32, copy=False)
    else:
        exp = np.exp(logits - np.max(logits))
        denom = np.sum(exp)
        probs = exp / (denom if denom != 0 else 1.0)

    # Map to labels
    preds: Dict[str, float] = { _index_to_label[i]: float(p) for i, p in enumerate(probs) }
    # Top-1
    top_idx = int(np.argmax(probs))
    top_label = _index_to_label[top_idx]
    top_conf = float(probs[top_idx])

    # Optional debug logging of top-3
    if os.getenv("DEBUG_LOG", "false").lower() in {"1", "true", "yes", "y"}:
        top3_idx = np.argsort(-probs)[:3]
        top3 = [(int(i), _index_to_label[int(i)], float(probs[int(i)])) for i in top3_idx]
        logging.info("Top-3 predictions: %s", top3)

    return JSONResponse({
        "emotion": top_label,
        "confidence": top_conf,
        "all_predictions": preds,
    })

@app.get("/")
async def root():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}
