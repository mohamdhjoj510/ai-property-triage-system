"""Streamlit Web UI for the AI Property Triage System."""

from typing import Any

import requests
import streamlit as st

GUARDRAILS_SERVICE_URL = "http://127.0.0.1:8002/check/input"
RAG_SERVICE_URL = "http://127.0.0.1:8001/query"
IMAGE_ANALYZER_SERVICE_URL = "http://127.0.0.1:8003/analyze"
AGENT_SERVICE_URL = "http://127.0.0.1:8004/agent/run"
SERVICE_TIMEOUT_SECONDS = 30
AGENT_TIMEOUT_SECONDS = 120

ROUTE_BADGE_RENDERER = {
    "residential": lambda label: st.success(f"Suggested route: {label}"),
    "commercial": lambda label: st.info(f"Suggested route: {label}"),
    "review_required": lambda label: st.warning(f"Suggested route: {label}"),
}
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
PREVIEW_COLUMNS = 3

ASSISTANT_PLACEHOLDER_REPLY = (
    "This is a placeholder response from the local real estate assistant."
)


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


def _post_files(
    url: str, files: list[tuple], unavailable_message: str
) -> dict[str, Any] | None:
    """POST multipart files to a backend service. Returns parsed JSON or None on failure."""
    try:
        response = requests.post(url, files=files, timeout=SERVICE_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(unavailable_message)
    except requests.exceptions.Timeout:
        st.error("Service request timed out. Please try again.")
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


def call_rag_service(description: str) -> dict[str, Any] | None:
    return _post_json(
        RAG_SERVICE_URL,
        {"description": description},
        "RAG service is not available on port 8001.",
    )


def call_image_analyzer_service(uploaded_files) -> dict[str, Any] | None:
    files = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in uploaded_files
    ]
    return _post_files(
        IMAGE_ANALYZER_SERVICE_URL,
        files,
        "Image Analyzer service is not available on port 8003.",
    )


def call_agent_service(
    description: str,
    rag_result: dict[str, Any] | None,
    image_analysis: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return _post_json(
        AGENT_SERVICE_URL,
        {
            "description": description,
            "rag_result": rag_result or {},
            "image_analysis": image_analysis or {},
        },
        "Agent service is not available on port 8004.",
        timeout=AGENT_TIMEOUT_SECONDS,
        timeout_message=(
            "Agent service timed out. Ollama may still be generating. "
            "Please try again."
        ),
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


def render_assistant_tab() -> None:
    st.subheader("Conversational Assistant")
    st.info(
        "Local LLM (Ollama) integration will be added later. "
        "For now this tab returns a placeholder response."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask the real estate assistant...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append(
            {"role": "assistant", "content": ASSISTANT_PLACEHOLDER_REPLY}
        )
        with st.chat_message("assistant"):
            st.markdown(ASSISTANT_PLACEHOLDER_REPLY)


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

    with st.expander("Raw agent response"):
        st.json(agent_response)


def run_triage_pipeline(description: str, uploaded_files) -> None:
    """Guardrails → RAG → Image Analyzer → Agent, with step-by-step status feedback."""
    rag_response: dict[str, Any] | None = None
    image_response: dict[str, Any] | None = None
    agent_response: dict[str, Any] | None = None
    images_skipped = False

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

        st.write("Querying RAG service...")
        rag_response = call_rag_service(description)
        if rag_response is None:
            status.update(label="RAG service unavailable", state="error")
            return
        st.write("RAG analysis complete")

        if not uploaded_files:
            st.write("No images uploaded — skipping image analysis")
            images_skipped = True
        else:
            st.write(f"Analysing {len(uploaded_files)} image(s)...")
            image_response = call_image_analyzer_service(uploaded_files)
            if image_response is None:
                status.update(label="Image Analyzer service unavailable", state="error")
                return
            st.write("Image analysis complete")

        st.write("Running agent analysis...")
        agent_response = call_agent_service(description, rag_response, image_response)
        if agent_response is None:
            status.update(label="Agent service unavailable", state="error")
            return
        st.write("Agent analysis complete")

        status.update(label="Analysis complete", state="complete")

    st.markdown("### RAG service response")
    st.json(rag_response)

    st.markdown("### Image Analyzer response")
    if images_skipped:
        st.info("No images uploaded, skipping image analysis.")
    else:
        st.json(image_response)

    render_agent_analysis(agent_response)


def render_submission_tab() -> None:
    st.subheader("Listing Submission")
    st.write(
        "Submit a property listing to run it through the triage pipeline: "
        "guardrails → RAG comparable-listing analysis → image analysis → agent reasoning."
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
    run_triage_pipeline(description.strip(), uploaded_files or [])


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
