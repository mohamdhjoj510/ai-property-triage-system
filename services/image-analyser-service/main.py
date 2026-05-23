"""Image analyser service — real CLIP-based room classification.

Replaces the previous random mock with two real components:

  * Zero-shot room-type classification using OpenAI's CLIP (via Hugging Face
    transformers). Each candidate label is wrapped in a short natural-language
    prompt and CLIP picks the best match.
  * A simple, deterministic condition heuristic based on PIL image statistics
    (brightness, contrast, sharpness). The heuristic is isolated in
    `estimate_condition_score` so a trained PyTorch model can replace its
    body later without touching the HTTP surface.
"""

import io
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

app = FastAPI(title="image-analyser-service")

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Natural-language prompts give CLIP much better zero-shot accuracy than
# raw label strings. Keep the prompt and label aligned via insertion order.
ROOM_LABEL_PROMPTS: dict[str, str] = {
    "kitchen": "a photo of a kitchen",
    "bathroom": "a photo of a bathroom",
    "bedroom": "a photo of a bedroom",
    "living_room": "a photo of a living room",
    "exterior": "a photo of a building exterior",
    "other": "a photo of something other than a room",
}
ROOM_LABELS = list(ROOM_LABEL_PROMPTS.keys())
ROOM_PROMPTS = list(ROOM_LABEL_PROMPTS.values())

MIN_CONDITION_SCORE = 1
MAX_CONDITION_SCORE = 5

# Eager model load at startup so the first request isn't slow.
_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
_model.eval()


def classify_room(image: Image.Image) -> tuple[str, float]:
    """Zero-shot CLIP classification. Returns (label, confidence in [0, 1])."""
    inputs = _processor(
        text=ROOM_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)
    best_idx = int(torch.argmax(probs).item())
    return ROOM_LABELS[best_idx], float(probs[best_idx].item())


def estimate_condition_score(image: Image.Image) -> int:
    """Map image quality signals (well-lit-ness, contrast, sharpness) to 1..5.

    Pure deterministic heuristic — to be replaced by a trained PyTorch
    classifier later. Kept isolated so the swap is mechanical.
    """
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)

    brightness = stats.mean[0] / 255.0
    contrast = min(stats.stddev[0] / 80.0, 1.0)

    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = min(ImageStat.Stat(edges).stddev[0] / 50.0, 1.0)

    # Well-lit-ness peaks at mid-grey (~0.5); over- or under-exposed images
    # score lower even if contrast/sharpness are decent.
    well_lit = 1.0 - min(abs(brightness - 0.5) * 2, 1.0)

    composite = (well_lit + contrast + sharpness) / 3.0
    score = 1 + int(round(composite * 4))
    return max(MIN_CONDITION_SCORE, min(MAX_CONDITION_SCORE, score))


def _fallback_result(filename: str) -> dict:
    return {
        "filename": filename,
        "detected_room_type": "other",
        "condition_score": MIN_CONDITION_SCORE,
        "confidence": 0.0,
    }


async def analyse_upload(file: UploadFile) -> dict:
    filename = file.filename or "unknown"
    raw = await file.read()

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return _fallback_result(filename)

    try:
        room_label, confidence = classify_room(image)
        condition_score = estimate_condition_score(image)
    except Exception:
        return _fallback_result(filename)

    return {
        "filename": filename,
        "detected_room_type": room_label,
        "condition_score": condition_score,
        "confidence": round(confidence, 4),
    }


@app.get("/")
def root():
    return {"service": "image-analyser-service", "status": "running"}


@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        results.append(await analyse_upload(file))
    return {"results": results}
