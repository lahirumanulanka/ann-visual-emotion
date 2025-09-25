# HF Space: Docker container
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for onnxruntime and pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

# Copy model and optional label map into the image
# Ensure these files exist in the build context (the same folder as this Dockerfile)
COPY model.onnx /app/model.onnx
COPY label_map.json /app/label_map.json

# Point the app to the bundled model and label map
ENV MODEL_PATH=/app/model.onnx \
    LABEL_MAP_JSON=/app/label_map.json

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
