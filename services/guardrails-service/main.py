"""Guardrails service — minimal rule-based input validation.

Rejects empty, too-short, spammy, or clearly off-topic text.
NeMo Guardrails and LLM-based checks will be added later.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="guardrails-service")

MIN_TEXT_LENGTH = 15

SPAM_PHRASES = (
    "buy crypto",
    "free money",
    "click here",
)

REAL_ESTATE_KEYWORDS = (
    "apartment",
    "house",
    "kitchen",
    "bedroom",
    "property",
    "balcony",
    "parking",
    "room",
    "flat",
    "villa",
    "studio",
    "garden",
    "bathroom",
    "floor",
)


class InputCheckRequest(BaseModel):
    text: str


def _is_empty(text: str) -> bool:
    return not text.strip()


def _is_too_short(text: str) -> bool:
    return len(text.strip()) < MIN_TEXT_LENGTH


def _contains_spam(text: str) -> str | None:
    lowered = text.lower()
    for phrase in SPAM_PHRASES:
        if phrase in lowered:
            return phrase
    return None


def _mentions_real_estate(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in REAL_ESTATE_KEYWORDS)


def validate_text(text: str) -> tuple[bool, str | None]:
    """Run all guardrail rules. Returns (pass, reason)."""
    if _is_empty(text):
        return False, "Text is empty."

    if _is_too_short(text):
        return False, f"Text is too short (minimum {MIN_TEXT_LENGTH} characters)."

    spam_match = _contains_spam(text)
    if spam_match is not None:
        return False, f"Text contains spam-like phrase: '{spam_match}'."

    if not _mentions_real_estate(text):
        return (
            False,
            "Text appears off-topic — no real-estate-related keywords detected.",
        )

    return True, None


@app.get("/")
def root() -> dict:
    return {"service": "guardrails-service", "status": "running"}


@app.post("/check/input")
def check_input(request: InputCheckRequest) -> dict:
    passed, reason = validate_text(request.text)
    return {"pass": passed, "reason": reason}
