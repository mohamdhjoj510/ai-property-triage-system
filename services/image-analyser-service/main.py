"""Image analyser service — trained ResNet18 (primary) + CLIP zero-shot (fallback).

If the trained checkpoint exists at `models/real_estate_room_classifier.pth`
and loads cleanly, every request uses the trained head. If the checkpoint is
missing, the load fails, or inference raises, the service transparently falls
back to CLIP zero-shot so the endpoint contract is never broken.

The condition heuristic stays deterministic / PIL-based for now — kept in
its own helper so it can be replaced independently of the room classifier.

The response gains one additional field, `classifier`, which records which
path actually produced the room label: `"trained_resnet18"` or
`"clip_fallback"`. All other fields are unchanged.
"""

import io
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor

app = FastAPI(title="image-analyser-service")

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "real_estate_room_classifier.pth"

# --- Trained-model defaults (used when the checkpoint omits these fields) ---
DEFAULT_IMG_SIZE = 224
DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Trained-class -> stable-API-label mapping ---
# The training dataset uses verbose folder names; the API has used short
# canonical labels since v1. This map keeps the public surface stable while
# letting the training dataset evolve independently.
LABEL_MAPPING = {
    "kitchen_dining": "kitchen",
    "building_exterior": "exterior",
    "living_room": "living_room",
    "not_real_estate": "other",
    "balcony": "balcony",
    "garden": "garden",
    "bathroom": "bathroom",
    "bedroom": "bedroom",
}

# --- CLIP fallback config (unchanged from the previous version) ---
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
ROOM_LABEL_PROMPTS: dict[str, str] = {
    "kitchen": "a photo of a kitchen",
    "bathroom": "a photo of a bathroom",
    "bedroom": "a photo of a bedroom",
    "living_room": "a photo of a living room",
    "exterior": "a photo of a building exterior",
    "other": "a photo of something other than a room",
}
CLIP_ROOM_LABELS = list(ROOM_LABEL_PROMPTS.keys())
CLIP_ROOM_PROMPTS = list(ROOM_LABEL_PROMPTS.values())

MIN_CONDITION_SCORE = 1
MAX_CONDITION_SCORE = 5

CLASSIFIER_TRAINED = "trained_resnet18"
CLASSIFIER_FALLBACK = "clip_fallback"

# --- Image-content guardrail ---
# When the trained classifier emits the `not_real_estate` raw label with high
# confidence, the analyser flags the upload as suspicious. Threshold tuned for
# precision over recall: false positives here block submissions, so we'd
# rather miss a few than block legitimate property photos.
SUSPICIOUS_IMAGE_THRESHOLD = 0.75
NOT_REAL_ESTATE_RAW_LABEL = "not_real_estate"


# --- Trained classifier loading ---


def normalize_class_label(label: str) -> str:
    """Map a trained-model class name to the API's stable room-type label."""
    return LABEL_MAPPING.get(label, label)


def load_trained_classifier() -> dict | None:
    """Load the trained ResNet18 checkpoint. Returns a bundle dict or None.

    The bundle holds the eval-mode model, the inverse class-index mapping,
    and the preprocessing transform sized for the checkpoint.
    """
    if not CHECKPOINT_PATH.exists():
        print(
            f"[image-analyser] No trained checkpoint at {CHECKPOINT_PATH}. "
            "Falling back to CLIP zero-shot for room classification."
        )
        return None

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    except Exception as exc:
        print(
            f"[image-analyser] Failed to load checkpoint {CHECKPOINT_PATH}: {exc}. "
            "Falling back to CLIP zero-shot."
        )
        return None

    try:
        num_classes = int(checkpoint["num_classes"])
        class_to_idx = dict(checkpoint["class_to_idx"])
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except (KeyError, ValueError, RuntimeError, TypeError) as exc:
        print(
            f"[image-analyser] Checkpoint loaded but model build failed: {exc}. "
            "Falling back to CLIP zero-shot."
        )
        return None

    img_size = int(checkpoint.get("img_size", DEFAULT_IMG_SIZE))
    mean = list(checkpoint.get("imagenet_mean", DEFAULT_IMAGENET_MEAN))
    std = list(checkpoint.get("imagenet_std", DEFAULT_IMAGENET_STD))

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print(
        f"[image-analyser] Trained classifier loaded from {CHECKPOINT_PATH} "
        f"({num_classes} classes, {img_size}x{img_size})."
    )
    return {
        "model": model,
        "idx_to_class": idx_to_class,
        "transform": transform,
    }


# --- CLIP fallback (always loaded so the fallback path is ready) ---

print(f"[image-analyser] Loading CLIP fallback ({CLIP_MODEL_ID})...")
_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
_clip_model.eval()
print("[image-analyser] CLIP fallback ready.")

_trained_classifier = load_trained_classifier()


# --- Classifiers ---


def classify_with_trained_model(image: Image.Image) -> tuple[str, str, float]:
    """Run the trained ResNet18 head. Returns (api_label, raw_label, confidence).

    `raw_label` is the unmapped trained-class name (e.g. `not_real_estate`,
    `kitchen_dining`) so callers can apply content guardrails before the
    label is normalised for the public API.
    """
    assert _trained_classifier is not None  # caller checks before invoking
    bundle = _trained_classifier
    tensor = bundle["transform"](image).unsqueeze(0)
    with torch.no_grad():
        logits = bundle["model"](tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    best_idx = int(torch.argmax(probs).item())
    raw_label = bundle["idx_to_class"].get(best_idx, "other")
    return normalize_class_label(raw_label), raw_label, float(probs[best_idx].item())


def classify_with_clip_fallback(image: Image.Image) -> tuple[str, str, float]:
    """Zero-shot CLIP classification. Returns (api_label, raw_label, confidence).

    CLIP labels are already in the API's namespace, so `raw_label == api_label`.
    CLIP cannot emit `not_real_estate`, so its results never trigger the
    image guardrail.
    """
    inputs = _clip_processor(
        text=CLIP_ROOM_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = _clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)
    best_idx = int(torch.argmax(probs).item())
    label = CLIP_ROOM_LABELS[best_idx]
    return label, label, float(probs[best_idx].item())


def _evaluate_image_guardrail(raw_label: str, confidence: float) -> dict:
    """Compute the image-content guardrail flags for a single classification."""
    is_real_estate = raw_label != NOT_REAL_ESTATE_RAW_LABEL
    warning_raised = not is_real_estate
    status = (
        "warning"
        if warning_raised and confidence >= SUSPICIOUS_IMAGE_THRESHOLD
        else "pass"
    )
    return {
        "is_real_estate_image": is_real_estate,
        "image_guardrail_warning": warning_raised,
        "image_guardrail_status": status,
    }


# --- Condition heuristic (unchanged) ---


def estimate_condition_score(image: Image.Image) -> int:
    """Map image quality signals (well-lit-ness, contrast, sharpness) to 1..5."""
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)

    brightness = stats.mean[0] / 255.0
    contrast = min(stats.stddev[0] / 80.0, 1.0)

    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = min(ImageStat.Stat(edges).stddev[0] / 50.0, 1.0)

    well_lit = 1.0 - min(abs(brightness - 0.5) * 2, 1.0)
    composite = (well_lit + contrast + sharpness) / 3.0
    score = 1 + int(round(composite * 4))
    return max(MIN_CONDITION_SCORE, min(MAX_CONDITION_SCORE, score))


# --- Per-upload analysis ---


def _fallback_result(filename: str) -> dict:
    """Returned when the file can't be decoded as an image. Never blocks submission."""
    return {
        "filename": filename,
        "detected_room_type": "other",
        "condition_score": MIN_CONDITION_SCORE,
        "confidence": 0.0,
        "classifier": CLASSIFIER_FALLBACK,
        "raw_label": "other",
        "is_real_estate_image": True,
        "image_guardrail_warning": False,
        "image_guardrail_status": "pass",
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

    label: str | None = None
    raw_label: str = "other"
    confidence: float = 0.0
    classifier_used = CLASSIFIER_FALLBACK

    # Primary path: trained model if available.
    if _trained_classifier is not None:
        try:
            label, raw_label, confidence = classify_with_trained_model(image)
            classifier_used = CLASSIFIER_TRAINED
        except Exception:
            # Swallow and fall through to CLIP — no traceback to client.
            label = None

    # Fallback path: CLIP zero-shot.
    if label is None:
        try:
            label, raw_label, confidence = classify_with_clip_fallback(image)
            classifier_used = CLASSIFIER_FALLBACK
        except Exception:
            return _fallback_result(filename)

    try:
        condition_score = estimate_condition_score(image)
    except Exception:
        condition_score = MIN_CONDITION_SCORE

    guardrail = _evaluate_image_guardrail(raw_label, confidence)

    return {
        "filename": filename,
        "detected_room_type": label,
        "condition_score": condition_score,
        "confidence": round(confidence, 4),
        "classifier": classifier_used,
        "raw_label": raw_label,
        **guardrail,
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
