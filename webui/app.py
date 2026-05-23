"""Streamlit Web UI for the AI Property Triage System."""

from typing import Any

import requests
import streamlit as st

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

ROUTE_BADGE_RENDERER = {
    "residential": lambda label: st.success(f"Suggested route: {label}"),
    "commercial": lambda label: st.info(f"Suggested route: {label}"),
    "review_required": lambda label: st.warning(f"Suggested route: {label}"),
}

CONFIDENCE_BADGE_RENDERER = {
    "high": lambda label: st.success(f"Confidence: {label}"),
    "medium": lambda label: st.info(f"Confidence: {label}"),
    "low": lambda label: st.warning(f"Confidence: {label}"),
}
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
PREVIEW_COLUMNS = 3

def _post_json(
    url: str,
    payload: dict[str, Any],
    unavailable_message: str,
    timeout: int = SERVICE_TIMEOUT_SECONDS,
    timeout_message: str = "Service request timed out. Please try again.",
) -> dict[str, Any] | None:
    """POST JSON to a backend service. Returns parsed JSON or None on failure."""
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
    description: str,
    agent_name: str,
    uploaded_files,
) -> dict[str, Any] | None:
    """POST multipart (description + agent_name + images) to the autonomous agent."""
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
        parts.append(
            f"Description: {triage_context.get('description', 'N/A')}"
        )
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
    """Call Ollama llama3 for a single chat turn. Returns reply text, or None if unavailable."""
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


def render_assistant_tab() -> None:
    st.subheader("Property AI Assistant")

    triage_context = st.session_state.get("last_triage")
    if triage_context:
        st.caption(
            "Ask questions about the most recently analysed property. "
            "The assistant runs locally on Ollama (llama3)."
        )
    else:
        st.info(
            "No property has been analysed yet — submit a listing in the "
            "**Listing Submission** tab first, then come back here to ask about it."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask about the analysed property...")
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


def render_submitted_summary(payload: dict[str, Any], image_names: list[str]) -> None:
    st.markdown("### Submitted listing")
    summary_col, images_col = st.columns([2, 1])
    with summary_col:
        st.json(payload)
    with images_col:
        st.markdown("**Uploaded image names**")
        if image_names:
            for name in image_names:
                st.write(f"- {name}")
        else:
            st.caption("No images attached.")


def render_output_validation(validation: dict[str, Any]) -> None:
    """Pretty-print the validator's findings as a dedicated section."""
    st.markdown("### Output Validation")

    if not validation:
        st.info("No validation data was returned by the agent.")
        return

    validation_passed = validation.get("validation_passed", False)
    confidence_level = validation.get("confidence_level", "unknown")
    risky_detected = bool(validation.get("risky_claims_detected"))
    unsupported_claims = validation.get("unsupported_claims") or []

    status_col, confidence_col, risky_col = st.columns(3)

    with status_col:
        st.markdown("**Status**")
        if validation_passed:
            st.success("✅ Passed")
        else:
            st.warning("⚠️ Needs review")

    with confidence_col:
        st.markdown("**Confidence**")
        label = confidence_level.title() if isinstance(confidence_level, str) else "Unknown"
        renderer = CONFIDENCE_BADGE_RENDERER.get(confidence_level)
        if renderer is not None:
            renderer(label)
        else:
            st.info(f"Confidence: {label}")

    with risky_col:
        st.markdown("**Risky claims**")
        if risky_detected:
            st.warning("Risky claims detected")
        else:
            st.success("No risky claims detected")

    st.markdown("**Unsupported claims**")
    if unsupported_claims:
        for claim in unsupported_claims:
            st.write(f"- {claim}")
    else:
        st.caption("No unsupported claims detected.")


def render_agent_analysis(agent_response: dict[str, Any]) -> None:
    """Pretty-print the structured agent response."""
    st.markdown("### Agent Analysis")

    suggested_route = agent_response.get("suggested_route", "unknown")
    route_label = suggested_route.replace("_", " ").title()
    renderer = ROUTE_BADGE_RENDERER.get(suggested_route)
    if renderer is not None:
        renderer(route_label)
    else:
        st.info(f"Suggested route: {route_label}")

    summary = agent_response.get("property_summary")
    if summary:
        st.markdown("**Property summary**")
        st.write(summary)

    recommendations = agent_response.get("recommendations") or []
    st.markdown("**Recommendations**")
    if recommendations:
        for item in recommendations:
            st.write(f"- {item}")
    else:
        st.caption("No recommendations returned.")

    renovation_insights = agent_response.get("renovation_insights") or []
    st.markdown("**Renovation insights**")
    if renovation_insights:
        for item in renovation_insights:
            st.write(f"- {item}")
    else:
        st.caption("No renovation insights flagged.")

    tools_used = agent_response.get("tools_used") or []
    st.markdown("**Tools used**")
    if tools_used:
        st.write(", ".join(tools_used))
    else:
        st.caption("No tools reported.")

    render_output_validation(agent_response.get("validation") or {})

    with st.expander("Raw agent response"):
        st.json(agent_response)


def run_triage_pipeline(description: str, agent_name: str, uploaded_files) -> None:
    """Guardrails → autonomous Agent (which decides which tools to call internally)."""
    agent_response: dict[str, Any] | None = None

    with st.status("Processing listing...", expanded=True) as status:
        st.write("Checking listing safety...")
        guardrails_result = call_guardrails_service(description)
        if guardrails_result is None:
            status.update(label="Guardrails service unavailable", state="error")
            return

        if not guardrails_result.get("pass"):
            reason = guardrails_result.get("reason") or "no reason provided"
            st.error(f"Listing rejected by guardrails: {reason}")
            status.update(label="Listing rejected", state="error")
            return

        st.write("Listing approved")

        st.write("Running autonomous AI agent...")
        agent_response = call_agent_with_images_service(
            description, agent_name, uploaded_files
        )
        if agent_response is None:
            status.update(label="Agent service unavailable", state="error")
            return
        st.write("Agent completed analysis")

        status.update(label="Analysis complete", state="complete")

    rag_response = agent_response.get("rag_result")
    image_response = agent_response.get("image_analysis")

    st.session_state.last_triage = {
        "description": description,
        "rag_response": rag_response,
        "image_response": image_response,
        "agent_response": agent_response,
    }

    st.markdown("### RAG service response (via agent)")
    if rag_response:
        st.json(rag_response)
    else:
        st.info("RAG did not contribute to this analysis.")

    st.markdown("### Image Analyzer response (via agent)")
    if image_response:
        st.json(image_response)
    elif not uploaded_files:
        st.info("No images uploaded, image analysis skipped.")
    else:
        st.info("Image analysis did not contribute to this analysis.")

    render_agent_analysis(agent_response)


def render_submission_tab() -> None:
    st.subheader("Listing Submission")
    st.write(
        "Submit a property listing. Guardrails check it for safety, then an "
        "autonomous LangGraph agent decides which tools to call (RAG, image "
        "analyser) and produces a triaged recommendation."
    )

    with st.container(border=True):
        st.markdown("**Listing details**")
        description = st.text_area(
            "Property description",
            height=180,
            placeholder="e.g. Renovated 3-room apartment near the beach...",
        )
        agent_name = st.text_input(
            "Listing agent name", placeholder="e.g. Dana Levi"
        )

    with st.container(border=True):
        st.markdown("**Property images**")
        uploaded_files = st.file_uploader(
            "Upload one or more property images",
            type=ALLOWED_IMAGE_TYPES,
            accept_multiple_files=True,
        )
        if uploaded_files:
            render_image_previews(uploaded_files)
        else:
            st.caption("No images uploaded yet.")

    st.divider()
    submitted = st.button("Submit listing", type="primary")

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

    image_names = [f.name for f in (uploaded_files or [])]
    submitted_payload = {
        "description": description.strip(),
        "agent_name": agent_name.strip(),
        "image_count": len(image_names),
        "image_names": image_names,
    }

    st.divider()
    render_submitted_summary(submitted_payload, image_names)
    run_triage_pipeline(
        description.strip(), agent_name.strip(), uploaded_files or []
    )


st.set_page_config(page_title="AI Property Triage System", layout="wide")
st.title("AI Property Triage System")
st.caption(
    "Analyse real estate listings end-to-end: retrieve comparable properties, "
    "inspect images, and produce a triaged recommendation."
)
st.divider()

assistant_tab, submission_tab = st.tabs(
    ["Conversational Assistant", "Listing Submission"]
)

with assistant_tab:
    render_assistant_tab()

with submission_tab:
    render_submission_tab()
