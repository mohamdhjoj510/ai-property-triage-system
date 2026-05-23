"""Streamlit Web UI for the AI Property Triage System."""

from typing import Any

import requests
import streamlit as st

RAG_SERVICE_URL = "http://127.0.0.1:8001/query"
RAG_TIMEOUT_SECONDS = 10
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
PREVIEW_COLUMNS = 3

ASSISTANT_PLACEHOLDER_REPLY = (
    "This is a placeholder response from the local real estate assistant."
)


def call_rag_service(description: str) -> dict[str, Any] | None:
    """POST the description to the RAG service. Returns parsed JSON or None on failure."""
    try:
        response = requests.post(
            RAG_SERVICE_URL,
            json={"description": description},
            timeout=RAG_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("RAG service is not available. Please start it on port 8001.")
    except requests.exceptions.Timeout:
        st.error("RAG service timed out. Please try again.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"RAG service returned an error: {exc.response.status_code}")
    except ValueError:
        st.error("RAG service returned an invalid JSON response.")
    return None


def render_image_previews(uploaded_files) -> None:
    """Render uploaded image previews in a responsive grid."""
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


def render_submission_tab() -> None:
    st.subheader("Listing Submission")
    st.write(
        "Submit a property listing to run it through the triage pipeline. "
        "Only the description is sent to the RAG service today; image analysis "
        "and downstream automation will be wired in next."
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
    st.markdown("### Submitted listing")
    summary_col, images_col = st.columns([2, 1])
    with summary_col:
        st.json(submitted_payload)
    with images_col:
        st.markdown("**Uploaded image names**")
        if image_names:
            for name in image_names:
                st.write(f"- {name}")
        else:
            st.caption("No images attached.")

    st.markdown("### RAG service response")
    with st.spinner("Querying RAG service..."):
        rag_response = call_rag_service(description.strip())
    if rag_response is not None:
        st.json(rag_response)


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
