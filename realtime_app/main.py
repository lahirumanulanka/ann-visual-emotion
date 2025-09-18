from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .face import crop_face, detect_faces
from .model import b64_to_image, get_model
from .schemas import PredictRequest, PredictResponse, Probability

app = FastAPI(title="Realtime Emotion Detection", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        pil_img = b64_to_image(req.image_base64)
        image_rgb = np.array(pil_img)

        face_box = None
        if req.detect_face:
            faces = detect_faces(image_rgb)
            if faces:
                face_box = faces[0]
                face_crop = crop_face(image_rgb, face_box)
            else:
                face_crop = image_rgb
        else:
            face_crop = image_rgb

        # Convert cropped face back to PIL
        pil_face = Image.fromarray(face_crop)
        model = get_model()
        top_label, ranked = model.predict(pil_face)
        probs: List[Probability] = [
            Probability(label=label_name, probability=prob) for label_name, prob in ranked
        ]

        return PredictResponse(
            top_label=top_label,
            probabilities=probs,
            face_box=face_box,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e)) from e

# Serve simple frontend
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("realtime_app/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# Mount static if needed
app.mount("/static", StaticFiles(directory="realtime_app/static"), name="static")
