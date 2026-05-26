"""Streamlit Web UI for the AI Property Triage System.

Enterprise-style dark dashboard. All backend calls, the pipeline shape,
and session_state keys are preserved — only presentation and form
lifecycle have changed.
"""

import html
import json
from pathlib import Path
from typing import Any

import requests
import streamlit as st

# --- Constants (unchanged) ---

GUARDRAILS_SERVICE_URL = "http://127.0.0.1:8002/check/input"
AGENT_WITH_IMAGES_URL = "http://127.0.0.1:8004/agent/run-with-images"
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_CHAT_MODEL = "llama3"
SERVICE_TIMEOUT_SECONDS = 30
AGENT_TIMEOUT_SECONDS = 180
CHAT_TIMEOUT_SECONDS = 120
CHAT_UNAVAILABLE_MESSAGE = "Local AI assistant is unavailable."
CHAT_OFF_TOPIC_REPLY = (
    "I can only assist with questions related to the analysed property."
)

ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
PREVIEW_COLUMNS = 3
RESULT_CARD_COLUMNS = 3
DEFAULT_AGENT_NAME = "Mohammad Hajuj"

WEBUI_DIR = Path(__file__).resolve().parent
LISTINGS_PATH = WEBUI_DIR.parent / "data" / "synthetic-listings" / "listings.json"

CAPABILITY_BADGES = [
    "RAG",
    "Vision AI",
    "LangGraph Agent",
    "Ollama",
    "Guardrails",
    "n8n Ready",
]

ROUTE_LABEL = {
    "residential": "Residential",
    "commercial": "Commercial",
    "review_required": "Review Required",
}


# --- HTTP helpers (unchanged behavior) ---


def _post_json(
    url: str,
    payload: dict[str, Any],
    unavailable_message: str,
    timeout: int = SERVICE_TIMEOUT_SECONDS,
    timeout_message: str = "Service request timed out. Please try again.",
) -> dict[str, Any] | None:
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(unavailable_message)
    except requests.exceptions.Timeout:
        st.error(timeout_message)
    except requests.exceptions.HTTPError as exc:
        st.error(f"Service returned an HTTP error: {exc.response.status_code}")
    except ValueError:
        st.error("Service returned an invalid JSON response.")
    return None


def call_guardrails_service(text: str) -> dict[str, Any] | None:
    return _post_json(
        GUARDRAILS_SERVICE_URL,
        {"text": text},
        "Guardrails service is not available on port 8002.",
    )


def call_agent_with_images_service(
    description: str, agent_name: str, uploaded_files
) -> dict[str, Any] | None:
    files = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in (uploaded_files or [])
    ]
    data = {"description": description, "agent_name": agent_name}
    try:
        response = requests.post(
            AGENT_WITH_IMAGES_URL,
            data=data,
            files=files or None,
            timeout=AGENT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Agent service is not available on port 8004.")
    except requests.exceptions.Timeout:
        st.error(
            "Agent service timed out. Ollama may still be generating. "
            "Please try again."
        )
    except requests.exceptions.HTTPError as exc:
        st.error(f"Agent service returned an HTTP error: {exc.response.status_code}")
    except ValueError:
        st.error("Agent service returned an invalid JSON response.")
    return None


# --- Chat helpers (unchanged behavior) ---


def _summarise_rag_for_chat(rag_response: dict[str, Any] | None) -> str:
    if not rag_response:
        return "No RAG results available."
    listings = rag_response.get("similar_listings") or []
    if not listings:
        return "No similar listings retrieved."
    lines = []
    for listing in listings:
        lines.append(
            f"- {listing.get('property_type', '?')} in "
            f"{listing.get('location', '?')}, "
            f"price {listing.get('price', '?')}, "
            f"condition {listing.get('condition', '?')}"
        )
    insight = rag_response.get("insight")
    if insight:
        lines.append(f"RAG insight: {insight}")
    return "\n".join(lines)


def _summarise_images_for_chat(image_response: dict[str, Any] | None) -> str:
    if not image_response:
        return "No image analysis available."
    results = image_response.get("results") or []
    if not results:
        return "No image analysis results."
    lines = []
    for item in results:
        lines.append(
            f"- {item.get('filename', '?')}: "
            f"{item.get('detected_room_type', '?')}, "
            f"condition score {item.get('condition_score', '?')}/5"
        )
    return "\n".join(lines)


def _summarise_agent_for_chat(agent_response: dict[str, Any] | None) -> str:
    if not agent_response:
        return "No agent analysis available."
    lines = [
        f"Suggested route: {agent_response.get('suggested_route', '?')}",
        f"Summary: {agent_response.get('property_summary', '?')}",
    ]
    recommendations = agent_response.get("recommendations") or []
    if recommendations:
        lines.append("Recommendations:")
        for item in recommendations:
            lines.append(f"- {item}")
    renovation_insights = agent_response.get("renovation_insights") or []
    if renovation_insights:
        lines.append("Renovation insights:")
        for item in renovation_insights:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _build_chat_prompt(
    user_message: str,
    history: list[dict[str, str]],
    triage_context: dict[str, Any] | None,
) -> str:
    parts = [
        "You are a helpful real-estate triage assistant. "
        "Answer questions ONLY about the analysed property described below, "
        "or general real-estate questions directly relevant to it. "
        "If the user asks about anything unrelated to this property or real estate, "
        f'respond with EXACTLY: "{CHAT_OFF_TOPIC_REPLY}"',
        "",
        "ANALYSED PROPERTY CONTEXT:",
    ]

    if triage_context:
        parts.append(f"Description: {triage_context.get('description', 'N/A')}")
        parts.append("")
        parts.append("Similar listings from RAG:")
        parts.append(_summarise_rag_for_chat(triage_context.get("rag_response")))
        parts.append("")
        parts.append("Image analysis:")
        parts.append(_summarise_images_for_chat(triage_context.get("image_response")))
        parts.append("")
        parts.append("Agent analysis:")
        parts.append(_summarise_agent_for_chat(triage_context.get("agent_response")))
    else:
        parts.append(
            "No property has been analysed yet in this session. "
            "Ask the user to submit a listing in the Listing Submission tab first."
        )

    if history:
        parts.append("")
        parts.append("CONVERSATION HISTORY:")
        for message in history:
            role_label = "User" if message["role"] == "user" else "Assistant"
            parts.append(f"{role_label}: {message['content']}")

    parts.append("")
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    return "\n".join(parts)


def call_chat_assistant(
    user_message: str,
    history: list[dict[str, str]],
    triage_context: dict[str, Any] | None,
) -> str | None:
    prompt = _build_chat_prompt(user_message, history, triage_context)
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=CHAT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
        ValueError,
    ):
        return None

    reply = (data.get("response") or "").strip()
    return reply or CHAT_OFF_TOPIC_REPLY


# --- Listings dataset loader ---


@st.cache_data(show_spinner=False)
def load_synthetic_listings() -> list[dict] | None:
    """Load the synthetic listings JSON, or None if unavailable / invalid."""
    try:
        with open(LISTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    return data


# --- Theme ---


def apply_custom_theme() -> None:
    """Dark enterprise AI command-center theme with neon-green accents."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #0A0E0C;
            --bg-elev: #121815;
            --bg-elev-2: #1A211D;
            --border: #1F2A23;
            --border-strong: #2A3830;
            --text: #E8EAED;
            --text-muted: #8A9491;
            --text-dim: #5A6562;
            --accent: #86BC25;
            --accent-bright: #A6E22E;
            --accent-dim: #5C9C0E;
            --accent-2: #7B61FF;
            --info: #74A0FF;
            --success: #86BC25;
            --warning: #FFC107;
            --danger: #FF5252;
        }

        /* Base canvas with subtle ambient glow */
        .stApp {
            background: var(--bg);
            color: var(--text);
            background-image:
                radial-gradient(circle at 12% 18%, rgba(134, 188, 37, 0.05) 0%, transparent 35%),
                radial-gradient(circle at 88% 78%, rgba(134, 188, 37, 0.035) 0%, transparent 40%);
            background-attachment: fixed;
        }
        [data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }
        .block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1440px; }

        h1, h2, h3, h4, h5 { color: var(--text); }
        p, label, span, div { color: var(--text); }

        /* ---------- Hero ---------- */
        .apt-hero {
            position: relative;
            padding: 36px 36px 30px 36px;
            border: 1px solid var(--border);
            border-radius: 20px;
            margin: 4px 0 24px 0;
            background:
                radial-gradient(circle at 18% 30%, rgba(134, 188, 37, 0.10) 0%, transparent 50%),
                radial-gradient(circle at 86% 72%, rgba(134, 188, 37, 0.06) 0%, transparent 55%),
                linear-gradient(135deg, #0A0E0C 0%, #121815 100%);
            overflow: hidden;
            box-shadow: 0 0 0 1px rgba(134, 188, 37, 0.05), 0 20px 60px -20px rgba(134, 188, 37, 0.10);
        }
        .apt-hero::before {
            content: "";
            position: absolute;
            top: -40%;
            right: -8%;
            width: 55%;
            height: 220%;
            background: radial-gradient(ellipse, rgba(134, 188, 37, 0.18) 0%, transparent 60%);
            filter: blur(56px);
            pointer-events: none;
        }
        .apt-hero::after {
            content: "";
            position: absolute;
            bottom: -50%;
            left: 8%;
            width: 38%;
            height: 160%;
            background: radial-gradient(ellipse, rgba(166, 226, 46, 0.10) 0%, transparent 55%);
            filter: blur(72px);
            pointer-events: none;
        }
        .apt-hero-eyebrow {
            color: var(--accent-bright);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 3px;
            text-transform: uppercase;
            position: relative;
            z-index: 1;
            text-shadow: 0 0 12px rgba(166, 226, 46, 0.35);
        }
        .apt-hero-title {
            color: #FFFFFF;
            font-size: 44px;
            font-weight: 700;
            margin: 10px 0 12px 0;
            letter-spacing: -0.5px;
            line-height: 1.1;
            position: relative;
            z-index: 1;
            text-shadow: 0 0 40px rgba(134, 188, 37, 0.18);
        }
        .apt-hero-subtitle {
            color: #B8C0BD;
            font-size: 16px;
            line-height: 1.5;
            margin: 0;
            max-width: 760px;
            position: relative;
            z-index: 1;
        }

        /* ---------- Status badges ---------- */
        .apt-status-bar {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: -6px 0 28px 0;
        }
        .apt-status-badge {
            background: rgba(134, 188, 37, 0.07);
            border: 1px solid rgba(134, 188, 37, 0.28);
            color: #C8D4B2;
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 7px;
            transition: all 0.2s;
        }
        .apt-status-badge:hover {
            border-color: rgba(134, 188, 37, 0.5);
            box-shadow: 0 0 14px rgba(134, 188, 37, 0.18);
        }
        .apt-status-badge::before {
            content: "";
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent-bright);
            box-shadow: 0 0 8px rgba(166, 226, 46, 0.8);
        }

        /* ---------- Section header ---------- */
        .apt-section {
            color: #FFFFFF;
            font-size: 16px;
            font-weight: 600;
            margin: 24px 0 14px 0;
            display: flex;
            align-items: center;
            gap: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        .apt-section::before {
            content: "";
            display: inline-block;
            width: 3px;
            height: 16px;
            background: var(--accent-bright);
            border-radius: 2px;
            box-shadow: 0 0 10px rgba(166, 226, 46, 0.6);
        }

        /* ---------- Pipeline / "next" panel ---------- */
        .apt-next-panel {
            background: linear-gradient(135deg, rgba(134, 188, 37, 0.04) 0%, transparent 100%);
            border: 1px solid rgba(134, 188, 37, 0.22);
            border-left: 3px solid var(--accent);
            border-radius: 12px;
            padding: 18px 20px;
            position: relative;
            overflow: hidden;
        }
        .apt-next-panel::after {
            content: "";
            position: absolute;
            top: -50%;
            right: -20%;
            width: 60%;
            height: 200%;
            background: radial-gradient(ellipse, rgba(134, 188, 37, 0.06) 0%, transparent 60%);
            filter: blur(40px);
            pointer-events: none;
        }
        .apt-next-title {
            color: var(--accent-bright);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin: 0 0 12px 0;
            position: relative;
        }
        .apt-next-step { display: flex; align-items: flex-start; gap: 10px; margin: 10px 0; position: relative; }
        .apt-next-step .num { color: var(--accent-bright); font-weight: 700; font-size: 13px; min-width: 18px; }
        .apt-next-step .label { color: var(--text); font-size: 13px; font-weight: 500; }
        .apt-next-step .desc { color: var(--text-muted); font-size: 12px; display: block; margin-top: 2px; }

        /* ---------- Success card ---------- */
        .apt-success-card {
            background:
                radial-gradient(circle at 0% 0%, rgba(134, 188, 37, 0.14) 0%, transparent 55%),
                linear-gradient(135deg, rgba(134, 188, 37, 0.06) 0%, transparent 100%);
            border: 1px solid rgba(134, 188, 37, 0.35);
            border-left: 4px solid var(--accent);
            border-radius: 14px;
            padding: 22px 26px;
            margin: 14px 0 20px 0;
            position: relative;
            overflow: hidden;
            box-shadow: 0 0 28px -8px rgba(134, 188, 37, 0.18);
        }
        .apt-success-card::before {
            content: "";
            position: absolute;
            top: -30%;
            right: -10%;
            width: 40%;
            height: 200%;
            background: radial-gradient(ellipse, rgba(166, 226, 46, 0.10) 0%, transparent 60%);
            filter: blur(50px);
            pointer-events: none;
        }
        .apt-success-title {
            color: var(--accent-bright);
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 14px;
            text-shadow: 0 0 18px rgba(166, 226, 46, 0.3);
            position: relative;
        }
        .apt-success-grid { display: flex; flex-wrap: wrap; gap: 32px; position: relative; }
        .apt-success-grid > div { display: flex; flex-direction: column; min-width: 110px; }
        .apt-success-grid .k { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px; }
        .apt-success-grid .v { color: var(--text); font-size: 16px; font-weight: 600; }

        /* ---------- Executive summary ---------- */
        .apt-exec {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 20px;
            height: 100%;
            transition: all 0.2s ease;
        }
        .apt-exec:hover { border-color: var(--border-strong); }
        .apt-exec-title { color: #FFFFFF; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px; }
        .apt-exec-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed var(--border); }
        .apt-exec-row:last-child { border-bottom: none; }
        .apt-exec-row .k { color: var(--text-muted); font-size: 12px; }
        .apt-exec-row .v { color: var(--text); font-size: 13px; font-weight: 600; text-align: right; }
        .apt-exec-row .v.concern { color: #FFD93D; }
        .apt-exec-row .v.clean { color: var(--accent-bright); }

        /* ---------- Timeline ---------- */
        .apt-timeline {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 20px;
            height: 100%;
            transition: all 0.2s ease;
        }
        .apt-timeline:hover { border-color: var(--border-strong); }
        .apt-timeline-title { color: #FFFFFF; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; }
        .apt-timeline-item { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px dashed var(--border); }
        .apt-timeline-item:last-child { border-bottom: none; }
        .apt-timeline-icon { font-size: 14px; width: 18px; text-align: center; font-weight: 700; }
        .apt-timeline-icon.ok { color: var(--accent-bright); text-shadow: 0 0 8px rgba(166, 226, 46, 0.4); }
        .apt-timeline-icon.skip { color: var(--text-dim); }
        .apt-timeline-icon.warn { color: #FFD93D; }
        .apt-timeline-text { color: var(--text); font-size: 13px; flex: 1; }
        .apt-timeline-meta { color: var(--text-muted); font-size: 11px; }

        /* ---------- Next actions ---------- */
        .apt-next-actions {
            background: linear-gradient(135deg, rgba(123, 97, 255, 0.05) 0%, transparent 100%);
            border: 1px solid rgba(123, 97, 255, 0.22);
            border-left: 3px solid var(--accent-2);
            border-radius: 12px;
            padding: 16px 20px;
            margin: 14px 0 20px 0;
        }
        .apt-next-actions-title { color: #A28CFF; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
        .apt-next-action-item { color: var(--text); font-size: 13px; padding: 5px 0; display: flex; align-items: center; gap: 10px; }
        .apt-next-action-item::before { content: "→"; color: var(--accent-2); font-weight: 700; }

        /* ---------- Listing card ---------- */
        .apt-listing-card {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 10px;
            padding: 14px 16px;
            height: 100%;
            transition: all 0.2s ease;
        }
        .apt-listing-card:hover {
            border-color: rgba(134, 188, 37, 0.45);
            border-left-color: var(--accent-bright);
            box-shadow: 0 6px 24px -8px rgba(134, 188, 37, 0.20);
            transform: translateY(-2px);
        }
        .apt-listing-type { color: var(--accent-bright); font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; }
        .apt-listing-loc { color: #FFFFFF; font-size: 15px; font-weight: 600; margin: 4px 0 2px 0; line-height: 1.3; }
        .apt-listing-price { color: var(--accent-bright); font-size: 17px; font-weight: 700; margin: 2px 0; text-shadow: 0 0 10px rgba(166, 226, 46, 0.18); }
        .apt-listing-cond { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
        .apt-listing-doc { color: #B8BCC4; font-size: 12px; line-height: 1.5; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); }
        .apt-listing-features { color: var(--text-dim); font-size: 11px; margin-top: 8px; font-style: italic; }

        /* ---------- Image card ---------- */
        .apt-image-card {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            transition: all 0.2s ease;
        }
        .apt-image-card:hover {
            border-color: rgba(134, 188, 37, 0.4);
            box-shadow: 0 4px 18px -8px rgba(134, 188, 37, 0.18);
        }
        .apt-image-filename { color: #FFFFFF; font-size: 13px; font-weight: 600; margin-bottom: 8px; word-break: break-all; }
        .apt-image-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 12px; }
        .apt-image-row .k { color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-size: 10px; }
        .apt-image-row .v { color: var(--text); font-weight: 600; }

        /* ---------- Route badge ---------- */
        .apt-route { display: inline-block; padding: 8px 18px; border-radius: 8px; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; }
        .apt-route.residential {
            background: rgba(134, 188, 37, 0.12);
            color: var(--accent-bright);
            border: 1px solid rgba(134, 188, 37, 0.45);
            box-shadow: 0 0 16px -4px rgba(134, 188, 37, 0.30);
        }
        .apt-route.commercial {
            background: rgba(116, 160, 255, 0.10);
            color: var(--info);
            border: 1px solid rgba(116, 160, 255, 0.35);
        }
        .apt-route.review_required {
            background: rgba(255,193,7,0.10);
            color: #FFD93D;
            border: 1px solid rgba(255,193,7,0.35);
        }

        /* ---------- Summary card ---------- */
        .apt-summary {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 20px;
            margin: 12px 0;
            line-height: 1.6;
            color: var(--text);
            font-size: 14px;
        }

        /* ---------- Recommendation / insight bullets ---------- */
        .apt-rec {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent-2);
            border-radius: 8px;
            padding: 10px 14px;
            margin: 8px 0;
            color: var(--text);
            font-size: 13px;
            line-height: 1.5;
            transition: border-color 0.2s;
        }
        .apt-rec:hover { border-left-color: #A28CFF; }
        .apt-insight {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-left: 3px solid var(--warning);
            border-radius: 8px;
            padding: 10px 14px;
            margin: 8px 0;
            color: var(--text);
            font-size: 13px;
            line-height: 1.5;
        }

        /* ---------- Validation tiles ---------- */
        .apt-val-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0 18px 0; }
        .apt-val-tile {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px 16px;
            flex: 1;
            min-width: 180px;
            transition: all 0.2s ease;
        }
        .apt-val-tile:hover { border-color: var(--border-strong); }
        .apt-val-tile .label { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; }
        .apt-val-tile .value { font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 6px; }
        .apt-val-tile.passed { border-left: 3px solid var(--accent); }
        .apt-val-tile.passed .value { color: var(--accent-bright); }
        .apt-val-tile.failed { border-left: 3px solid var(--warning); }
        .apt-val-tile.failed .value { color: #FFD93D; }
        .apt-val-tile.high { border-left: 3px solid var(--accent); }
        .apt-val-tile.high .value { color: var(--accent-bright); }
        .apt-val-tile.medium { border-left: 3px solid var(--info); }
        .apt-val-tile.medium .value { color: var(--info); }
        .apt-val-tile.low { border-left: 3px solid var(--warning); }
        .apt-val-tile.low .value { color: #FFD93D; }
        .apt-val-tile.risk-clean { border-left: 3px solid var(--accent); }
        .apt-val-tile.risk-clean .value { color: var(--accent-bright); }
        .apt-val-tile.risk-detected { border-left: 3px solid var(--danger); }
        .apt-val-tile.risk-detected .value { color: #FF8888; }
        .apt-val-tile.risk-high { border-left: 3px solid var(--danger); }
        .apt-val-tile.risk-high .value { color: #FF8888; }
        .apt-val-tile.risk-low { border-left: 3px solid var(--accent); }
        .apt-val-tile.risk-low .value { color: var(--accent-bright); }

        /* ---------- Empty state ---------- */
        .apt-empty {
            background: var(--bg-elev);
            border: 1px dashed var(--border-strong);
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
        }
        .apt-empty-title { color: #FFFFFF; font-size: 16px; font-weight: 600; margin: 4px 0 6px 0; }
        .apt-empty-text { color: var(--text-muted); font-size: 13px; margin: 0; }

        /* ---------- Rejection card ---------- */
        .apt-rejection {
            background: rgba(255, 82, 82, 0.07);
            border: 1px solid rgba(255, 82, 82, 0.35);
            border-left: 4px solid var(--danger);
            border-radius: 12px;
            padding: 18px 20px;
            margin: 12px 0;
            box-shadow: 0 0 24px -8px rgba(255, 82, 82, 0.18);
        }
        .apt-rejection-title { color: #FF8888; font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
        .apt-rejection-reason { color: var(--text); font-size: 14px; line-height: 1.5; }

        .apt-suggested-label { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; margin: 18px 0 6px 0; }

        /* ---------- Dataset stat tiles ---------- */
        .apt-stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0 18px 0; }
        .apt-stat-tile {
            flex: 1;
            min-width: 160px;
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 10px;
            padding: 16px 18px;
            transition: all 0.2s ease;
        }
        .apt-stat-tile:hover {
            border-color: rgba(134, 188, 37, 0.4);
            box-shadow: 0 4px 18px -8px rgba(134, 188, 37, 0.18);
        }
        .apt-stat-tile .label { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; }
        .apt-stat-tile .value { color: #FFFFFF; font-size: 24px; font-weight: 700; margin-top: 4px; }
        .apt-stat-tile .sub { color: var(--text-muted); font-size: 11px; margin-top: 4px; }

        /* ---------- Breakdown lists ---------- */
        .apt-breakdown { background: var(--bg-elev); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
        .apt-breakdown-title { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
        .apt-breakdown-item { display: flex; justify-content: space-between; padding: 4px 0; color: var(--text); font-size: 13px; }
        .apt-breakdown-item .v { color: var(--accent-bright); font-weight: 600; }

        /* ---------- Streamlit widget tweaks ---------- */
        .stTextInput input, .stTextArea textarea {
            background: #0A0E0C !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(134, 188, 37, 0.18) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--bg-elev) !important;
            border: 1px dashed var(--border-strong) !important;
            border-radius: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(134, 188, 37, 0.4) !important;
        }
        .stButton button {
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--bg-elev);
            color: var(--text);
            font-weight: 500;
            transition: all 0.2s;
        }
        .stButton button:hover {
            border-color: rgba(134, 188, 37, 0.45);
            background: var(--bg-elev-2);
            box-shadow: 0 0 14px -4px rgba(134, 188, 37, 0.20);
        }
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent-bright) 0%, var(--accent-dim) 100%);
            border: none;
            color: #0A0E0C;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-size: 13px;
            box-shadow: 0 0 0 1px rgba(134, 188, 37, 0.3), 0 6px 20px -4px rgba(134, 188, 37, 0.25);
        }
        .stButton button[kind="primary"]:hover {
            background: linear-gradient(135deg, #B6F23E 0%, var(--accent) 100%);
            box-shadow: 0 0 0 1px rgba(166, 226, 46, 0.55), 0 8px 28px -4px rgba(134, 188, 37, 0.45);
            transform: translateY(-1px);
            border: none;
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: var(--text-muted);
            padding: 12px 20px;
            border-radius: 8px 8px 0 0;
            font-weight: 600;
            font-size: 13px;
            letter-spacing: 0.5px;
            transition: color 0.2s;
        }
        .stTabs [data-baseweb="tab"]:hover { color: var(--accent-bright); }
        .stTabs [aria-selected="true"] {
            background: var(--bg-elev);
            color: #FFFFFF;
            border-bottom: 2px solid var(--accent);
            box-shadow: 0 -1px 0 0 var(--border), 0 -8px 24px -8px rgba(134, 188, 37, 0.15);
        }

        /* ---------- Chat messages ---------- */
        [data-testid="stChatMessage"] {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        /* ---------- Expander ---------- */
        [data-testid="stExpander"] { background: var(--bg-elev); border: 1px solid var(--border); border-radius: 10px; }
        [data-testid="stExpander"] summary { color: var(--text); }
        [data-testid="stExpander"] summary:hover { color: var(--accent-bright); }

        /* ---------- JSON viewer ---------- */
        .stJson { background: var(--bg-elev) !important; border-radius: 8px; }

        /* ---------- Streamlit status (st.status) ---------- */
        [data-testid="stStatus"] {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        /* ---------- Native st.info / st.success / st.warning / st.error tweaks ---------- */
        div[data-baseweb="notification"] { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- HTML / utility helpers ---


def _esc(text: Any) -> str:
    return html.escape(str(text) if text is not None else "")


def _section(title: str) -> None:
    st.markdown(f'<div class="apt-section">{_esc(title)}</div>', unsafe_allow_html=True)


def _format_price(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{int(value):,} ILS"
    return _esc(value)


# --- Top-level page chrome ---


def render_header() -> None:
    st.markdown(
        """
        <div class="apt-hero">
          <div class="apt-hero-eyebrow">AI Engineering · Final Project</div>
          <h1 class="apt-hero-title">AI Property Triage System</h1>
          <p class="apt-hero-subtitle">Autonomous real-estate intake, retrieval, vision analysis, and validation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badges() -> None:
    badges = "".join(
        f'<span class="apt-status-badge">{_esc(name)}</span>' for name in CAPABILITY_BADGES
    )
    st.markdown(f'<div class="apt-status-bar">{badges}</div>', unsafe_allow_html=True)


# --- Form (with reset support) ---


def reset_submission_form() -> None:
    """Clear form-field session_state keys so the next render starts blank."""
    st.session_state.description_input = ""
    st.session_state.agent_name_input = DEFAULT_AGENT_NAME
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


def _handle_form_reset() -> None:
    """Honor a pending reset flag BEFORE widgets are created on this run."""
    if st.session_state.get("reset_form_flag"):
        reset_submission_form()
        st.session_state.reset_form_flag = False
        st.session_state.form_just_reset = True


def _ensure_form_state() -> None:
    if "description_input" not in st.session_state:
        st.session_state.description_input = ""
    if "agent_name_input" not in st.session_state:
        st.session_state.agent_name_input = DEFAULT_AGENT_NAME
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def render_image_previews(uploaded_files) -> None:
    if not uploaded_files:
        return
    st.caption(f"{len(uploaded_files)} image(s) ready to submit")
    for row_start in range(0, len(uploaded_files), PREVIEW_COLUMNS):
        row_files = uploaded_files[row_start : row_start + PREVIEW_COLUMNS]
        cols = st.columns(PREVIEW_COLUMNS)
        for col, image_file in zip(cols, row_files):
            with col:
                st.image(image_file, caption=image_file.name, use_container_width=True)


def render_what_happens_next() -> None:
    steps = [
        ("Input guardrails", "Rule-based checks on the description."),
        ("Agent tool orchestration", "LangGraph planner picks which tools to call."),
        ("RAG retrieval", "ChromaDB search over the listings corpus."),
        ("Vision analysis", "CLIP zero-shot room classification."),
        ("Output validation", "Unsupported claims, risky phrases, confidence."),
    ]
    items_html = "".join(
        f'<div class="apt-next-step">'
        f'<span class="num">{idx + 1}</span>'
        f'<div><span class="label">{_esc(label)}</span>'
        f'<span class="desc">{_esc(desc)}</span></div></div>'
        for idx, (label, desc) in enumerate(steps)
    )
    st.markdown(
        f'<div class="apt-next-panel">'
        f'<div class="apt-next-title">What happens next</div>{items_html}</div>',
        unsafe_allow_html=True,
    )


def render_submission_form():
    """Render the intake form. Returns (submitted, description, agent_name, uploaded_files)."""
    _ensure_form_state()

    with st.container(border=True):
        st.markdown("**Property description**")
        description = st.text_area(
            "Property description",
            label_visibility="collapsed",
            height=180,
            placeholder=(
                "Describe the property in 3-6 sentences. Include type "
                "(apartment / house / office / villa / retail / industrial), "
                "location, key features (balcony, parking, sea view), and "
                "condition (new, renovated, needs renovation, etc)."
            ),
            key="description_input",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Listing agent name**")
            agent_name = st.text_input(
                "Listing agent name",
                label_visibility="collapsed",
                key="agent_name_input",
            )
        with c2:
            st.markdown("**Reference / source**")
            st.text_input(
                "Reference",
                label_visibility="collapsed",
                placeholder="Optional — e.g. CRM ID, URL",
                disabled=True,
                help="Wired in a future iteration.",
            )

        st.markdown("**Property images**")
        uploaded_files = st.file_uploader(
            "Upload one or more property images",
            label_visibility="collapsed",
            type=ALLOWED_IMAGE_TYPES,
            accept_multiple_files=True,
            help="PNG / JPG / JPEG. Multiple files supported.",
            key=f"property_images_{st.session_state.uploader_key}",
        )
        if uploaded_files:
            render_image_previews(uploaded_files)
        else:
            st.caption("No images uploaded yet — vision analysis will be skipped.")

        st.markdown("")
        submitted = st.button("Run Autonomous Triage", type="primary", use_container_width=True)

    return submitted, description, agent_name, uploaded_files


def render_rejection_card(reason: str) -> None:
    st.markdown(
        f'<div class="apt-rejection">'
        f'<div class="apt-rejection-title">Submission rejected by input guardrails</div>'
        f'<div class="apt-rejection-reason">{_esc(reason)}</div></div>',
        unsafe_allow_html=True,
    )


# --- Demo / presentation cards ---


def _compute_primary_concern(triage: dict[str, Any]) -> str:
    agent = triage.get("agent_response") or {}
    validation = agent.get("validation") or {}
    renovation = agent.get("renovation_insights") or []
    if validation.get("risky_claims_detected"):
        return "Risky language detected in agent output"
    if validation.get("unsupported_claims"):
        return "Output flags unsupported claims"
    if renovation:
        return "Renovation work likely required"
    if agent.get("suggested_route") == "review_required":
        return "Property type unclear — human review"
    return "None flagged"


def _renovation_risk(triage: dict[str, Any]) -> str:
    agent = triage.get("agent_response") or {}
    return "high" if (agent.get("renovation_insights") or []) else "low"


def _validation_risk(triage: dict[str, Any]) -> str:
    agent = triage.get("agent_response") or {}
    validation = agent.get("validation") or {}
    if validation.get("risky_claims_detected") or not validation.get("validation_passed", True):
        return "high"
    return "low"


def _evidence_confidence(triage: dict[str, Any]) -> str:
    agent = triage.get("agent_response") or {}
    validation = agent.get("validation") or {}
    level = validation.get("confidence_level", "unknown")
    return level if level in {"high", "medium", "low"} else "medium"


def render_success_summary_card(triage: dict[str, Any]) -> None:
    agent = triage.get("agent_response") or {}
    rag = triage.get("rag_response") or {}
    image = triage.get("image_response") or {}
    validation = agent.get("validation") or {}

    route = agent.get("suggested_route", "unknown")
    route_label = ROUTE_LABEL.get(route, route.replace("_", " ").title())
    confidence = (validation.get("confidence_level") or "unknown").title()
    rag_count = len(rag.get("similar_listings") or [])
    image_count = len(image.get("results") or [])
    tools = agent.get("tools_used") or []
    tools_str = ", ".join(tools) if tools else "—"

    st.markdown(
        f"""
        <div class="apt-success-card">
          <div class="apt-success-title">✓ Triage completed successfully</div>
          <div class="apt-success-grid">
            <div><span class="k">Suggested route</span><span class="v">{_esc(route_label)}</span></div>
            <div><span class="k">Confidence</span><span class="v">{_esc(confidence)}</span></div>
            <div><span class="k">Comparable listings</span><span class="v">{rag_count}</span></div>
            <div><span class="k">Images analysed</span><span class="v">{image_count}</span></div>
            <div><span class="k">Tools used</span><span class="v">{_esc(tools_str)}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_summary(triage: dict[str, Any]) -> None:
    agent = triage.get("agent_response") or {}
    validation = agent.get("validation") or {}
    rag = triage.get("rag_response") or {}
    image = triage.get("image_response") or {}

    route = agent.get("suggested_route", "unknown")
    route_label = ROUTE_LABEL.get(route, route.replace("_", " ").title())
    confidence = (validation.get("confidence_level") or "unknown").title()
    primary_concern = _compute_primary_concern(triage)
    concern_class = "clean" if primary_concern == "None flagged" else "concern"

    st.markdown(
        f"""
        <div class="apt-exec">
          <div class="apt-exec-title">Executive Summary</div>
          <div class="apt-exec-row"><span class="k">Suggested route</span><span class="v">{_esc(route_label)}</span></div>
          <div class="apt-exec-row"><span class="k">Confidence</span><span class="v">{_esc(confidence)}</span></div>
          <div class="apt-exec-row"><span class="k">Primary concern</span><span class="v {concern_class}">{_esc(primary_concern)}</span></div>
          <div class="apt-exec-row"><span class="k">Comparable listings</span><span class="v">{len(rag.get("similar_listings") or [])}</span></div>
          <div class="apt-exec-row"><span class="k">Images analysed</span><span class="v">{len(image.get("results") or [])}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reasoning_timeline(triage: dict[str, Any]) -> None:
    agent = triage.get("agent_response") or {}
    validation = agent.get("validation") or {}
    rag = triage.get("rag_response") or {}
    image = triage.get("image_response") or {}
    image_count = triage.get("image_count", 0)

    rows: list[tuple[str, str, str]] = []
    rows.append(("ok", "Input guardrails passed", ""))

    if rag:
        rag_count = len(rag.get("similar_listings") or [])
        rows.append(("ok", "RAG retrieval used", f"{rag_count} comparable(s)"))
    else:
        rows.append(("skip", "RAG retrieval skipped", ""))

    if image:
        img_count = len(image.get("results") or [])
        rows.append(("ok", "Image analysis used", f"{img_count} file(s)"))
    elif image_count == 0:
        rows.append(("skip", "Image analysis skipped", "no images uploaded"))
    else:
        rows.append(("skip", "Image analysis did not contribute", ""))

    if validation.get("validation_passed", False):
        rows.append(("ok", "Output validation passed", ""))
    else:
        rows.append(("warn", "Output validation: needs review", ""))

    items_html = "".join(
        f'<div class="apt-timeline-item">'
        f'<span class="apt-timeline-icon {cls}">{"✓" if cls == "ok" else ("⊘" if cls == "skip" else "⚠")}</span>'
        f'<span class="apt-timeline-text">{_esc(text)}</span>'
        f'<span class="apt-timeline-meta">{_esc(meta)}</span>'
        f'</div>'
        for cls, text, meta in rows
    )
    st.markdown(
        f'<div class="apt-timeline">'
        f'<div class="apt-timeline-title">Agent Reasoning Timeline</div>{items_html}</div>',
        unsafe_allow_html=True,
    )


def render_risk_indicators(triage: dict[str, Any]) -> None:
    reno = _renovation_risk(triage)
    val_risk = _validation_risk(triage)
    evidence = _evidence_confidence(triage)

    reno_class = "risk-high" if reno == "high" else "risk-low"
    val_class = "risk-high" if val_risk == "high" else "risk-low"
    evidence_class = evidence  # high|medium|low maps to existing tile classes

    st.markdown(
        f"""
        <div class="apt-val-row">
          <div class="apt-val-tile {reno_class}">
            <div class="label">Renovation risk</div>
            <div class="value">{reno.title()}</div>
          </div>
          <div class="apt-val-tile {val_class}">
            <div class="label">Validation risk</div>
            <div class="value">{val_risk.title()}</div>
          </div>
          <div class="apt-val-tile {evidence_class}">
            <div class="label">Evidence confidence</div>
            <div class="value">{evidence.title()}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_next_actions() -> None:
    items = [
        "Review comparable listings",
        "Check renovation insights",
        "Ask follow-up questions in the assistant tab",
        "Export / report (planned)",
    ]
    items_html = "".join(
        f'<div class="apt-next-action-item">{_esc(item)}</div>' for item in items
    )
    st.markdown(
        f'<div class="apt-next-actions">'
        f'<div class="apt-next-actions-title">Next recommended actions</div>{items_html}</div>',
        unsafe_allow_html=True,
    )


# --- Existing result renderers ---


def render_submitted_summary(payload: dict[str, Any], image_names: list[str]) -> None:
    _section("Submitted Listing")
    summary_col, images_col = st.columns([2, 1])
    with summary_col:
        st.json(payload)
    with images_col:
        st.markdown("**Image inventory**")
        if image_names:
            for name in image_names:
                st.write(f"• {name}")
        else:
            st.caption("No images attached.")


def render_rag_cards(rag_response: dict[str, Any] | None) -> None:
    _section("Comparable Listings (RAG)")
    if not rag_response:
        st.info("RAG did not contribute to this analysis.")
        return

    listings = rag_response.get("similar_listings") or []
    insight = rag_response.get("insight")

    if insight:
        st.markdown(
            f'<div class="apt-summary"><strong style="color:#74A0FF;">RAG insight</strong><br/>{_esc(insight)}</div>',
            unsafe_allow_html=True,
        )

    if not listings:
        st.caption("No similar listings returned.")
        return

    for row_start in range(0, len(listings), RESULT_CARD_COLUMNS):
        row = listings[row_start : row_start + RESULT_CARD_COLUMNS]
        cols = st.columns(RESULT_CARD_COLUMNS)
        for col, listing in zip(cols, row):
            ptype = _esc(listing.get("property_type", "unknown"))
            location = _esc(listing.get("location", "unknown"))
            price_str = _format_price(listing.get("price"))
            condition = _esc(listing.get("condition", "unknown"))
            doc = (listing.get("document") or "").strip()
            if len(doc) > 220:
                doc = doc[:217] + "..."
            doc_html = _esc(doc).replace("\n", "<br/>")
            with col:
                st.markdown(
                    f"""
                    <div class="apt-listing-card">
                      <div class="apt-listing-type">{ptype}</div>
                      <div class="apt-listing-loc">{location}</div>
                      <div class="apt-listing-price">{price_str}</div>
                      <div class="apt-listing-cond">condition · {condition}</div>
                      <div class="apt-listing-doc">{doc_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_image_analysis_cards(
    image_response: dict[str, Any] | None, image_count: int
) -> None:
    _section("Image Analysis")
    if not image_response:
        if image_count == 0:
            st.info("No images uploaded, image analysis skipped.")
        else:
            st.info("Image analysis did not contribute to this analysis.")
        return

    results = image_response.get("results") or []
    if not results:
        st.caption("Image analyser returned no results.")
        return

    for row_start in range(0, len(results), RESULT_CARD_COLUMNS):
        row = results[row_start : row_start + RESULT_CARD_COLUMNS]
        cols = st.columns(RESULT_CARD_COLUMNS)
        for col, item in zip(cols, row):
            filename = _esc(item.get("filename", "unknown"))
            room_type = _esc(item.get("detected_room_type", "unknown"))
            condition_score = item.get("condition_score")
            confidence = item.get("confidence")
            confidence_str = (
                f"{float(confidence) * 100:.1f}%"
                if isinstance(confidence, (int, float))
                else _esc(confidence)
            )
            score_str = (
                f"{condition_score}/5"
                if isinstance(condition_score, (int, float))
                else _esc(condition_score)
            )
            with col:
                st.markdown(
                    f"""
                    <div class="apt-image-card">
                      <div class="apt-image-filename">{filename}</div>
                      <div class="apt-image-row"><span class="k">Room</span><span class="v">{room_type}</span></div>
                      <div class="apt-image-row"><span class="k">Condition</span><span class="v">{score_str}</span></div>
                      <div class="apt-image-row"><span class="k">Confidence</span><span class="v">{confidence_str}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_agent_analysis(agent_response: dict[str, Any]) -> None:
    _section("Agent Analysis")

    suggested_route = agent_response.get("suggested_route", "unknown")
    route_label = ROUTE_LABEL.get(suggested_route, suggested_route.replace("_", " ").title())
    route_class = suggested_route if suggested_route in ROUTE_LABEL else "review_required"
    st.markdown(
        f'<div class="apt-route {route_class}">Route · {_esc(route_label)}</div>',
        unsafe_allow_html=True,
    )

    summary = agent_response.get("property_summary")
    if summary:
        st.markdown(
            f'<div class="apt-summary">{_esc(summary).replace(chr(10), "<br/>")}</div>',
            unsafe_allow_html=True,
        )

    rec_col, ins_col = st.columns(2)

    with rec_col:
        st.markdown("**Recommendations**")
        recommendations = agent_response.get("recommendations") or []
        if recommendations:
            for item in recommendations:
                st.markdown(
                    f'<div class="apt-rec">{_esc(item)}</div>', unsafe_allow_html=True
                )
        else:
            st.caption("No recommendations returned.")

    with ins_col:
        st.markdown("**Renovation insights**")
        insights = agent_response.get("renovation_insights") or []
        if insights:
            for item in insights:
                st.markdown(
                    f'<div class="apt-insight">{_esc(item)}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No renovation insights flagged.")

    tools_used = agent_response.get("tools_used") or []
    if tools_used:
        chips = "".join(
            f'<span class="apt-status-badge">{_esc(t)}</span>' for t in tools_used
        )
        st.markdown(
            f'<div style="margin-top:14px;"><span class="apt-suggested-label">Tools used</span>'
            f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{chips}</div></div>',
            unsafe_allow_html=True,
        )


def render_output_validation(validation: dict[str, Any]) -> None:
    _section("Output Validation")

    if not validation:
        st.info("No validation data was returned by the agent.")
        return

    passed = bool(validation.get("validation_passed", False))
    confidence = validation.get("confidence_level", "unknown")
    risky = bool(validation.get("risky_claims_detected"))
    unsupported = validation.get("unsupported_claims") or []

    status_class = "passed" if passed else "failed"
    status_value = "✓ Passed" if passed else "⚠ Needs review"
    confidence_class = confidence if confidence in {"high", "medium", "low"} else "medium"
    confidence_value = (confidence or "unknown").title()
    risky_class = "risk-detected" if risky else "risk-clean"
    risky_value = "Risky claims detected" if risky else "No risky claims detected"

    st.markdown(
        f"""
        <div class="apt-val-row">
          <div class="apt-val-tile {status_class}">
            <div class="label">Status</div>
            <div class="value">{status_value}</div>
          </div>
          <div class="apt-val-tile {confidence_class}">
            <div class="label">Confidence</div>
            <div class="value">{_esc(confidence_value)}</div>
          </div>
          <div class="apt-val-tile {risky_class}">
            <div class="label">Risky claims</div>
            <div class="value">{risky_value}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Unsupported claims**")
    if unsupported:
        for claim in unsupported:
            st.markdown(
                f'<div class="apt-insight">{_esc(claim)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No unsupported claims detected.")


# --- Pipeline ---


def run_triage_pipeline(description: str, agent_name: str, uploaded_files) -> bool:
    """Guardrails → autonomous Agent. Persists results to session_state on success.

    Returns True on a successful triage (agent responded), False otherwise.
    """
    agent_response: dict[str, Any] | None = None

    with st.status("Triage in progress...", expanded=True) as status:
        st.write("Validating listing input...")
        guardrails_result = call_guardrails_service(description)
        if guardrails_result is None:
            status.update(label="Guardrails service unavailable", state="error")
            return False

        if not guardrails_result.get("pass"):
            reason = guardrails_result.get("reason") or "no reason provided"
            status.update(label="Listing rejected", state="error")
            render_rejection_card(reason)
            return False

        st.write("Input approved by guardrails.")
        st.write("Running autonomous LangGraph agent...")
        st.write("Retrieving comparable properties...")
        if uploaded_files:
            st.write("Analysing uploaded images...")
        st.write("Generating AI property brief...")
        st.write("Running output validation...")

        agent_response = call_agent_with_images_service(
            description, agent_name, uploaded_files
        )
        if agent_response is None:
            status.update(label="Agent service unavailable", state="error")
            return False

        st.write("Triage completed successfully.")
        status.update(label="Triage completed successfully.", state="complete")

    image_names = [f.name for f in (uploaded_files or [])]
    st.session_state.last_triage = {
        "description": description,
        "agent_name": agent_name,
        "rag_response": agent_response.get("rag_result"),
        "image_response": agent_response.get("image_analysis"),
        "agent_response": agent_response,
        "image_names": image_names,
        "image_count": len(image_names),
    }
    return True


def render_full_results_from_state() -> None:
    """Render the full results UI from session_state.last_triage."""
    triage = st.session_state.get("last_triage")
    if not triage:
        return

    description = triage["description"]
    agent_name = triage["agent_name"]
    rag_response = triage["rag_response"]
    image_response = triage["image_response"]
    agent_response = triage["agent_response"]
    image_names = triage["image_names"]
    image_count = triage["image_count"]

    _section("Triage Result")
    render_success_summary_card(triage)

    exec_col, timeline_col = st.columns(2)
    with exec_col:
        render_executive_summary(triage)
    with timeline_col:
        render_reasoning_timeline(triage)

    render_risk_indicators(triage)
    render_next_actions()

    submitted_payload = {
        "description": description,
        "agent_name": agent_name,
        "image_count": len(image_names),
        "image_names": image_names,
    }
    render_submitted_summary(submitted_payload, image_names)
    render_rag_cards(rag_response)
    render_image_analysis_cards(image_response, image_count)
    render_agent_analysis(agent_response)
    render_output_validation(agent_response.get("validation") or {})

    _section("Raw Response")
    with st.expander("Raw agent response"):
        st.json(agent_response)


# --- Submission tab ---


def render_submission_tab() -> None:
    # MUST run before any widget is created so reset is honored.
    _handle_form_reset()

    if st.session_state.get("form_just_reset"):
        st.info(
            "Form cleared. Latest analysis remains available below and in the assistant."
        )
        st.session_state.form_just_reset = False

    form_col, info_col = st.columns([2, 1])

    with form_col:
        _section("New Listing Intake")
        submitted, description, agent_name, uploaded_files = render_submission_form()

    with info_col:
        _section("Pipeline")
        render_what_happens_next()

    if submitted:
        errors = []
        if not description.strip():
            errors.append("Property description is required.")
        if not agent_name.strip():
            errors.append("Listing agent name is required.")

        if errors:
            for error in errors:
                st.warning(error)
            # Fall through to render any previous results
        else:
            success = run_triage_pipeline(
                description.strip(), agent_name.strip(), uploaded_files or []
            )
            if success:
                st.session_state.reset_form_flag = True
                st.rerun()
            # If failed, fall through to render previous results

    if st.session_state.get("last_triage"):
        render_full_results_from_state()


# --- Assistant tab ---


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="apt-empty">
          <div class="apt-empty-title">No property analysed yet</div>
          <p class="apt-empty-text">Submit a listing first to unlock grounded AI assistant answers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_suggested_questions(triage: dict[str, Any]) -> list[str]:
    """Build a context-aware list of suggested questions for the chat tab."""
    questions: list[str] = []
    agent = triage.get("agent_response") or {}
    route = agent.get("suggested_route")
    renovation_insights = agent.get("renovation_insights") or []

    if renovation_insights:
        questions.append("What should be renovated first?")
    if route == "commercial":
        questions.append("What should a business tenant verify?")
    if route == "residential":
        questions.append("Would this appeal to residential buyers?")
    questions.append("What are the main risks?")
    return questions


def render_suggested_questions(triage: dict[str, Any]) -> None:
    questions = _build_suggested_questions(triage)
    st.markdown(
        '<div class="apt-suggested-label">Suggested questions</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(questions))
    for col, question in zip(cols, questions):
        if col.button(question, key=f"sq__{question}", use_container_width=True):
            st.session_state.pending_chat_input = question


def render_assistant_tab() -> None:
    _section("Property AI Assistant")
    st.markdown(
        '<p style="color:#9AA0A6;margin:-6px 0 14px 0;font-size:13px;">'
        "Ask follow-up questions grounded in the latest analysed property. "
        "The assistant runs locally on Ollama (llama3)."
        "</p>",
        unsafe_allow_html=True,
    )

    triage_context = st.session_state.get("last_triage")
    if not triage_context:
        render_empty_state()
    else:
        st.success("Latest triage loaded into assistant context.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if triage_context:
        render_suggested_questions(triage_context)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask about the analysed property...")
    if not user_input and st.session_state.get("pending_chat_input"):
        user_input = st.session_state.pop("pending_chat_input")

    if not user_input:
        return

    history_snapshot = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = call_chat_assistant(user_input, history_snapshot, triage_context)
        if reply is None:
            reply = CHAT_UNAVAILABLE_MESSAGE
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})


# --- Listings dataset tab ---


def _compute_listings_stats(listings: list[dict]) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    condition_counts: dict[str, int] = {}
    prices: list[float] = []
    cities: set[str] = set()

    for listing in listings:
        ptype = listing.get("property_type") or "unknown"
        cond = listing.get("condition") or "unknown"
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
        condition_counts[cond] = condition_counts.get(cond, 0) + 1
        price = listing.get("price")
        if isinstance(price, (int, float)):
            prices.append(float(price))
        loc = listing.get("location")
        if isinstance(loc, str) and loc:
            cities.add(loc.split(",")[0].strip())

    return {
        "total": len(listings),
        "type_counts": type_counts,
        "condition_counts": condition_counts,
        "min_price": min(prices) if prices else None,
        "max_price": max(prices) if prices else None,
        "avg_price": (sum(prices) / len(prices)) if prices else None,
        "cities": sorted(cities),
    }


def _render_dataset_stats(stats: dict[str, Any]) -> None:
    if stats["min_price"] is not None:
        price_label = (
            f"{int(stats['min_price']):,}"
            f" – {int(stats['max_price']):,} ILS"
        )
        price_sub = f"avg {int(stats['avg_price']):,} ILS"
    else:
        price_label = "—"
        price_sub = ""

    tiles_html = f"""
    <div class="apt-stat-row">
      <div class="apt-stat-tile">
        <div class="label">Total listings</div>
        <div class="value">{stats['total']}</div>
        <div class="sub">in corpus</div>
      </div>
      <div class="apt-stat-tile">
        <div class="label">Property types</div>
        <div class="value">{len(stats['type_counts'])}</div>
        <div class="sub">distinct</div>
      </div>
      <div class="apt-stat-tile">
        <div class="label">Price range</div>
        <div class="value">{_esc(price_label)}</div>
        <div class="sub">{_esc(price_sub)}</div>
      </div>
      <div class="apt-stat-tile">
        <div class="label">Cities</div>
        <div class="value">{len(stats['cities'])}</div>
        <div class="sub">unique</div>
      </div>
    </div>
    """
    st.markdown(tiles_html, unsafe_allow_html=True)

    bd_col, cond_col = st.columns(2)
    with bd_col:
        items = "".join(
            f'<div class="apt-breakdown-item"><span>{_esc(k)}</span><span class="v">{v}</span></div>'
            for k, v in sorted(stats["type_counts"].items(), key=lambda x: (-x[1], x[0]))
        )
        st.markdown(
            f'<div class="apt-breakdown">'
            f'<div class="apt-breakdown-title">By property type</div>{items}</div>',
            unsafe_allow_html=True,
        )
    with cond_col:
        items = "".join(
            f'<div class="apt-breakdown-item"><span>{_esc(k)}</span><span class="v">{v}</span></div>'
            for k, v in sorted(stats["condition_counts"].items(), key=lambda x: (-x[1], x[0]))
        )
        st.markdown(
            f'<div class="apt-breakdown">'
            f'<div class="apt-breakdown-title">By condition</div>{items}</div>',
            unsafe_allow_html=True,
        )


def _render_dataset_filters(listings: list[dict]) -> tuple[list[str], list[str], str]:
    types = sorted({l.get("property_type") for l in listings if l.get("property_type")})
    conditions = sorted({l.get("condition") for l in listings if l.get("condition")})

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        type_filter = st.multiselect(
            "Property type",
            types,
            default=[],
            placeholder="All types",
            key="dataset_filter_type",
        )
    with c2:
        condition_filter = st.multiselect(
            "Condition",
            conditions,
            default=[],
            placeholder="All conditions",
            key="dataset_filter_condition",
        )
    with c3:
        location_filter = st.text_input(
            "Location contains",
            placeholder="e.g. Tel Aviv, Haifa...",
            key="dataset_filter_location",
        )
    return type_filter, condition_filter, location_filter


def build_dataset_listing_description(listing: dict) -> str:
    """Compose a single readable description string from a dataset listing."""
    title = listing.get("title", "Untitled listing")
    ptype = listing.get("property_type", "unknown")
    location = listing.get("location", "unknown")
    price = listing.get("price")
    price_str = (
        f"{int(price):,} ILS" if isinstance(price, (int, float)) else str(price)
    )
    rooms = listing.get("rooms", "?")
    features_list = listing.get("features") or []
    features = ", ".join(features_list) if features_list else "none listed"
    condition = listing.get("condition", "unknown")
    description = (listing.get("description") or "").strip() or "no description provided"

    return (
        f"{title}. Type: {ptype}. Location: {location}. Price: {price_str}. "
        f"Rooms: {rooms}. Features: {features}. Condition: {condition}. "
        f"Description: {description}"
    )


def render_dataset_listing_card(listing: dict) -> None:
    """Render one dataset listing card and its 'Analyse this listing' action."""
    ptype = _esc(listing.get("property_type", "?"))
    rooms = listing.get("rooms", "?")
    title = _esc(listing.get("title", ""))
    location = _esc(listing.get("location", ""))
    price_str = _format_price(listing.get("price"))
    condition = _esc(listing.get("condition", "?"))
    description = (listing.get("description") or "").strip()
    if len(description) > 200:
        description = description[:197] + "..."
    features = ", ".join(listing.get("features") or [])
    features_html = (
        f'<div class="apt-listing-features">features: {_esc(features)}</div>'
        if features
        else ""
    )

    st.markdown(
        f"""
        <div class="apt-listing-card">
          <div class="apt-listing-type">{ptype} · {_esc(rooms)} rooms</div>
          <div class="apt-listing-loc">{title}</div>
          <div style="color: var(--text-muted); font-size: 12px;">{location}</div>
          <div class="apt-listing-price">{price_str}</div>
          <div class="apt-listing-cond">condition · {condition}</div>
          <div class="apt-listing-doc">{_esc(description)}</div>
          {features_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    listing_id = listing.get("id", "unknown")
    if st.button(
        "Analyse this listing",
        key=f"analyse_listing_{listing_id}",
        use_container_width=True,
    ):
        st.session_state.pending_dataset_analysis = build_dataset_listing_description(
            listing
        )
        st.rerun()


def _render_dataset_grid(listings: list[dict]) -> None:
    if not listings:
        st.info("No listings match the current filters.")
        return

    for row_start in range(0, len(listings), 3):
        row = listings[row_start : row_start + 3]
        cols = st.columns(3)
        for col, listing in zip(cols, row):
            with col:
                render_dataset_listing_card(listing)


def _render_dataset_demo_intro() -> None:
    st.markdown(
        """
        <div class="apt-next-panel" style="margin-bottom: 18px;">
          <div class="apt-next-title">Dataset Demo Mode</div>
          <p style="color: var(--text); font-size: 13px; margin: 0; line-height: 1.5;">
            Select any synthetic listing and run the full autonomous triage pipeline
            without manually typing a description. The selected listing's metadata
            is composed into a description and sent through the same Guardrails →
            Agent flow as a manual submission.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_listings_dataset_tab() -> None:
    _section("Listings Dataset")
    st.markdown(
        '<p style="color:#9AA0A6;margin:-6px 0 14px 0;font-size:13px;">'
        "Visual browser over the synthetic listings corpus that the RAG service "
        "is grounded on. Not connected to ChromaDB — reads the JSON directly."
        "</p>",
        unsafe_allow_html=True,
    )

    _render_dataset_demo_intro()

    # Handle a pending dataset-triggered analysis. Runs the same pipeline the
    # submission tab uses; persists results to session_state.last_triage.
    pending_analysis = st.session_state.pop("pending_dataset_analysis", None)
    if pending_analysis:
        st.info("Running analysis for selected dataset listing...")
        run_triage_pipeline(pending_analysis, DEFAULT_AGENT_NAME, [])
        # Intentionally no st.rerun() here — let results render below in the
        # same script run so the user sees them immediately.

    # Render the latest triage at the top of the tab (whether it was just
    # produced by a dataset click or came from a previous submission).
    if st.session_state.get("last_triage"):
        render_full_results_from_state()
        st.divider()

    listings = load_synthetic_listings()
    if not listings:
        st.markdown(
            f"""
            <div class="apt-empty">
              <div class="apt-empty-title">Listings dataset unavailable</div>
              <p class="apt-empty-text">
                Could not load <code>{_esc(LISTINGS_PATH.as_posix())}</code>.
                Make sure the file exists and is valid JSON.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    stats = _compute_listings_stats(listings)
    _render_dataset_stats(stats)

    _section("Browse")
    type_filter, condition_filter, location_filter = _render_dataset_filters(listings)

    filtered = [
        l for l in listings
        if (not type_filter or l.get("property_type") in type_filter)
        and (not condition_filter or l.get("condition") in condition_filter)
        and (
            not location_filter
            or location_filter.lower() in str(l.get("location", "")).lower()
        )
    ]

    st.caption(f"Showing {len(filtered)} of {len(listings)} listings")
    _render_dataset_grid(filtered)


# --- App entry ---


st.set_page_config(
    page_title="AI Property Triage System",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_custom_theme()
render_header()
render_status_badges()

submission_tab, assistant_tab, dataset_tab = st.tabs(
    ["Listing Submission", "Property AI Assistant", "Listings Dataset"]
)

with submission_tab:
    render_submission_tab()

with assistant_tab:
    render_assistant_tab()

with dataset_tab:
    render_listings_dataset_tab()
