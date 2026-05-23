"""Image analyser service — minimal FastAPI skeleton with mock predictions.

Returns randomised room-type and condition-score values per uploaded image.
A real CNN-based analyser will replace the mock helpers later — the rest of the
HTTP surface is intended to stay stable.
"""

import random
from typing import List

from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="image-analyser-service")

ROOM_TYPES = (
    "kitchen",
    "bathroom",
    "bedroom",
    "living_room",
    "exterior",
)

MIN_CONDITION_SCORE = 1
MAX_CONDITION_SCORE = 5


def get_mock_room_type() -> str:
    return random.choice(ROOM_TYPES)


def get_mock_condition_score() -> int:
    return random.randint(MIN_CONDITION_SCORE, MAX_CONDITION_SCORE)


@app.get("/")
def root():
    return {"service": "image-analyser-service", "status": "running"}


@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        results.append({
            "filename": file.filename,
            "detected_room_type": get_mock_room_type(),
            "condition_score": get_mock_condition_score()
        })
    return {"results": results}
