"""Streamlit Web UI for the AI Property Triage System.

Enterprise-style dark dashboard. All backend calls, the pipeline shape,
and session_state keys are preserved — only presentation has changed.
"""

import html
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

CAPABILITY_BADGES = [
    "RAG",
    "Vision AI",
    "LangGraph Agent",
    "Ollama",
    "Guardrails",
    "n8n Ready",
]

SUGGESTED_QUESTIONS = [
    "What are the main risks?",
    "What should the agent verify?",
    "Is this property attractive for investors?",
    "What should be renovated first?",
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


# --- Theme ---


def apply_custom_theme() -> None:
    """Inject dark enterprise dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #0F1419;
            --bg-elev: #1A1F2E;
            --bg-elev-2: #232938;
            --border: #2A3142;
            --text: #E8EAED;
            --text-muted: #9AA0A6;
            --text-dim: #6B7280;
            --accent: #4B7BEC;
            --accent-2: #7B61FF;
            --success: #28A745;
            --warning: #FFC107;
            --danger: #DC3545;
        }

        .stApp { background: var(--bg); color: var(--text); }
        [data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }
        .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px; }

        h1, h2, h3, h4, h5 { color: var(--text); }
        p, label, span, div { color: var(--text); }
        .stMarkdown small { color: var(--text-muted); }

        /* Hero */
        .apt-hero {
            padding: 4px 0 18px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 18px;
        }
        .apt-hero-eyebrow {
            color: var(--accent-2);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 2.5px;
            text-transform: uppercase;
        }
        .apt-hero-title {
            color: #FFFFFF;
            font-size: 38px;
            font-weight: 700;
            margin: 6px 0 8px 0;
            letter-spacing: -0.5px;
        }
        .apt-hero-subtitle {
            color: var(--text-muted);
            font-size: 15px;
            margin: 0;
        }

        /* Status badges row */
        .apt-status-bar {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 14px 0 28px 0;
        }
        .apt-status-badge {
            background: rgba(75, 123, 234, 0.08);
            border: 1px solid rgba(75, 123, 234, 0.25);
            color: #B8C4E0;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .apt-status-badge::before {
            content: "";
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 6px rgba(40, 167, 69, 0.6);
        }

        /* Section header */
        .apt-section {
            color: #FFFFFF;
            font-size: 16px;
            font-weight: 600;
            margin: 22px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 10px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        .apt-section::before {
            content: "";
            display: inline-block;
            width: 3px;
            height: 14px;
            background: var(--accent);
            border-radius: 2px;
        }

        /* What happens next */
        .apt-next-panel {
            background: rgba(75, 123, 234, 0.04);
            border: 1px solid rgba(75, 123, 234, 0.18);
            border-radius: 12px;
            padding: 18px 20px;
        }
        .apt-next-title {
            color: var(--accent);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin: 0 0 12px 0;
        }
        .apt-next-step {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin: 10px 0;
        }
        .apt-next-step .num {
            color: var(--accent);
            font-weight: 700;
            font-size: 13px;
            min-width: 18px;
        }
        .apt-next-step .label {
            color: var(--text);
            font-size: 13px;
            font-weight: 500;
        }
        .apt-next-step .desc {
            color: var(--text-muted);
            font-size: 12px;
            display: block;
            margin-top: 2px;
        }

        /* Listing card (RAG) */
        .apt-listing-card {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 10px;
            padding: 14px 16px;
            height: 100%;
        }
        .apt-listing-type {
            color: var(--accent);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
        .apt-listing-loc {
            color: #FFFFFF;
            font-size: 15px;
            font-weight: 600;
            margin: 4px 0 2px 0;
        }
        .apt-listing-price {
            color: var(--success);
            font-size: 17px;
            font-weight: 700;
            margin: 2px 0;
        }
        .apt-listing-cond {
            color: var(--text-muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .apt-listing-doc {
            color: #B8BCC4;
            font-size: 12px;
            line-height: 1.5;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border);
        }

        /* Image card */
        .apt-image-card {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
        }
        .apt-image-filename {
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 8px;
            word-break: break-all;
        }
        .apt-image-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 12px;
        }
        .apt-image-row .k { color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-size: 10px; }
        .apt-image-row .v { color: var(--text); font-weight: 600; }

        /* Route badge */
        .apt-route {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        .apt-route.residential { background: rgba(40,167,69,0.12); color: #4ADE80; border: 1px solid rgba(40,167,69,0.35); }
        .apt-route.commercial { background: rgba(75,123,234,0.12); color: #74A0FF; border: 1px solid rgba(75,123,234,0.35); }
        .apt-route.review_required { background: rgba(255,193,7,0.12); color: #FFD93D; border: 1px solid rgba(255,193,7,0.35); }

        /* Summary card */
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

        /* Recommendation bullet */
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
        }
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

        /* Validation badges */
        .apt-val-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0 18px 0; }
        .apt-val-tile {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 14px;
            flex: 1;
            min-width: 180px;
        }
        .apt-val-tile .label {
            color: var(--text-muted);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 6px;
        }
        .apt-val-tile .value { font-size: 15px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
        .apt-val-tile.passed { border-left: 3px solid var(--success); }
        .apt-val-tile.passed .value { color: #4ADE80; }
        .apt-val-tile.failed { border-left: 3px solid var(--warning); }
        .apt-val-tile.failed .value { color: #FFD93D; }
        .apt-val-tile.high { border-left: 3px solid var(--success); }
        .apt-val-tile.high .value { color: #4ADE80; }
        .apt-val-tile.medium { border-left: 3px solid var(--accent); }
        .apt-val-tile.medium .value { color: #74A0FF; }
        .apt-val-tile.low { border-left: 3px solid var(--warning); }
        .apt-val-tile.low .value { color: #FFD93D; }
        .apt-val-tile.risk-clean { border-left: 3px solid var(--success); }
        .apt-val-tile.risk-clean .value { color: #4ADE80; }
        .apt-val-tile.risk-detected { border-left: 3px solid var(--danger); }
        .apt-val-tile.risk-detected .value { color: #FF6B7A; }

        /* Empty state */
        .apt-empty {
            background: var(--bg-elev);
            border: 1px dashed var(--border);
            border-radius: 12px;
            padding: 36px 20px;
            text-align: center;
        }
        .apt-empty-title {
            color: #FFFFFF;
            font-size: 16px;
            font-weight: 600;
            margin: 4px 0 6px 0;
        }
        .apt-empty-text { color: var(--text-muted); font-size: 13px; margin: 0; }

        /* Rejection card */
        .apt-rejection {
            background: rgba(220, 53, 69, 0.08);
            border: 1px solid rgba(220, 53, 69, 0.35);
            border-left: 4px solid var(--danger);
            border-radius: 12px;
            padding: 18px 20px;
            margin: 12px 0;
        }
        .apt-rejection-title {
            color: #FF6B7A;
            font-weight: 700;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
        }
        .apt-rejection-reason { color: var(--text); font-size: 14px; line-height: 1.5; }

        /* Suggested-question chip helper text */
        .apt-suggested-label {
            color: var(--text-muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin: 18px 0 6px 0;
        }

        /* Streamlit widget tweaks */
        .stTextInput input, .stTextArea textarea {
            background: #0F1419 !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(75,123,234,0.15) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--bg-elev) !important;
            border: 1px dashed var(--border) !important;
            border-radius: 10px !important;
        }
        .stButton button {
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--bg-elev);
            color: var(--text);
            font-weight: 500;
            transition: all 0.15s;
        }
        .stButton button:hover {
            border-color: var(--accent);
            background: var(--bg-elev-2);
        }
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #4B7BEC 0%, #7B61FF 100%);
            border: none;
            color: white;
            font-weight: 600;
        }
        .stButton button[kind="primary"]:hover {
            filter: brightness(1.1);
            border: none;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: var(--text-muted);
            padding: 10px 18px;
            border-radius: 8px 8px 0 0;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: var(--bg-elev);
            color: #FFFFFF;
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        /* Containers */
        [data-testid="stExpander"] {
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        [data-testid="stExpander"] summary { color: var(--text); }

        /* JSON viewer dark tweak */
        .stJson { background: var(--bg-elev) !important; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- HTML helpers ---


def _esc(text: Any) -> str:
    return html.escape(str(text) if text is not None else "")


def _section(title: str) -> None:
    st.markdown(f'<div class="apt-section">{_esc(title)}</div>', unsafe_allow_html=True)


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


# --- Submission tab ---


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
        f"""
        <div class="apt-next-panel">
          <div class="apt-next-title">What happens next</div>
          {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def render_submission_form():
    """Returns (submitted, description, agent_name, uploaded_files)."""
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
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Listing agent name**")
            agent_name = st.text_input(
                "Listing agent name",
                label_visibility="collapsed",
                placeholder="e.g. Dana Levi",
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
        f"""
        <div class="apt-rejection">
          <div class="apt-rejection-title">Submission rejected by input guardrails</div>
          <div class="apt-rejection-reason">{_esc(reason)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            price = listing.get("price")
            price_str = (
                f"{int(price):,} ILS" if isinstance(price, (int, float)) else _esc(price)
            )
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
    image_response: dict[str, Any] | None, uploaded_files
) -> None:
    _section("Image Analysis")
    if image_response is None and not uploaded_files:
        st.info("No images uploaded, image analysis skipped.")
        return
    if not image_response:
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


def run_triage_pipeline(description: str, agent_name: str, uploaded_files) -> None:
    """Guardrails → autonomous Agent. Status messages worded as an enterprise console."""
    agent_response: dict[str, Any] | None = None

    with st.status("Triage in progress...", expanded=True) as status:
        st.write("Validating listing input...")
        guardrails_result = call_guardrails_service(description)
        if guardrails_result is None:
            status.update(label="Guardrails service unavailable", state="error")
            return

        if not guardrails_result.get("pass"):
            reason = guardrails_result.get("reason") or "no reason provided"
            status.update(label="Listing rejected", state="error")
            render_rejection_card(reason)
            return

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
            return

        st.write("Triage completed successfully.")
        status.update(label="Triage completed successfully.", state="complete")

    rag_response = agent_response.get("rag_result")
    image_response = agent_response.get("image_analysis")

    st.session_state.last_triage = {
        "description": description,
        "rag_response": rag_response,
        "image_response": image_response,
        "agent_response": agent_response,
    }

    image_names = [f.name for f in (uploaded_files or [])]
    submitted_payload = {
        "description": description,
        "agent_name": agent_name,
        "image_count": len(image_names),
        "image_names": image_names,
    }

    render_submitted_summary(submitted_payload, image_names)
    render_rag_cards(rag_response)
    render_image_analysis_cards(image_response, uploaded_files)
    render_agent_analysis(agent_response)
    render_output_validation(agent_response.get("validation") or {})

    _section("Raw Response")
    with st.expander("Raw agent response"):
        st.json(agent_response)


def render_submission_tab() -> None:
    form_col, info_col = st.columns([2, 1])

    with form_col:
        _section("New Listing Intake")
        submitted, description, agent_name, uploaded_files = render_submission_form()

    with info_col:
        _section("Pipeline")
        render_what_happens_next()

    if not submitted:
        return

    errors = []
    if not description.strip():
        errors.append("Property description is required.")
    if not agent_name.strip():
        errors.append("Listing agent name is required.")

    if errors:
        for error in errors:
            st.warning(error)
        return

    run_triage_pipeline(description.strip(), agent_name.strip(), uploaded_files or [])


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


def render_suggested_questions() -> None:
    st.markdown(
        '<div class="apt-suggested-label">Suggested questions</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, question in zip(cols, SUGGESTED_QUESTIONS):
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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if triage_context:
        render_suggested_questions()

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


# --- App entry ---


st.set_page_config(
    page_title="AI Property Triage System",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_custom_theme()
render_header()
render_status_badges()

submission_tab, assistant_tab = st.tabs(["Listing Submission", "Property AI Assistant"])

with submission_tab:
    render_submission_tab()

with assistant_tab:
    render_assistant_tab()
