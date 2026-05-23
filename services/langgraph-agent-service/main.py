"""LangGraph agent service — StateGraph + local LLM (Ollama llama3).

The graph runs:  planner → synthesizer

`planner_node` decides the routing label (rule-based) and records which
upstream tools contributed input. `synthesizer_node` calls a local llama3
model through Ollama to generate the property summary, recommendations,
and renovation insights — if the model is unreachable or returns invalid
JSON, the node falls back to the original rule-based helpers so the
endpoint contract is preserved.
"""

import json
import re
from typing import TypedDict

from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

app = FastAPI(title="langgraph-agent-service")

RESIDENTIAL_KEYWORDS = ("apartment", "house", "villa")
COMMERCIAL_KEYWORDS = ("office", "shop", "retail", "industrial")
LOW_CONDITION_THRESHOLD = 2

RENOVATION_IMAGE_INSIGHT = (
    "Some uploaded images indicate areas that may need improvement."
)
RENOVATION_TEXT_INSIGHT = (
    "The listing text or retrieved similar listings suggest renovation work "
    "may be required."
)

RENOVATION_DESCRIPTION_KEYWORDS = (
    "needs renovation",
    "requires renovation",
    "old",
    "dated",
)
RENOVATION_RAG_CONDITIONS = {"needs renovation", "fair"}

SUMMARY_SNIPPET_MAX_LEN = 120

OLLAMA_MODEL = "llama3"
OLLAMA_TEMPERATURE = 0.2


class AgentRunRequest(BaseModel):
    description: str
    rag_result: dict = Field(default_factory=dict)
    image_analysis: dict = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    # Inputs
    description: str
    rag_result: dict
    image_analysis: dict

    # Planner outputs
    suggested_route: str
    tools_used: list[str]

    # Synthesizer outputs
    property_summary: str
    recommendations: list[str]
    renovation_insights: list[str]


# --- Pure rule helpers (used by planner and as synthesizer fallback) ---


def determine_route(description: str) -> str:
    lowered = description.lower()
    if any(keyword in lowered for keyword in RESIDENTIAL_KEYWORDS):
        return "residential"
    if any(keyword in lowered for keyword in COMMERCIAL_KEYWORDS):
        return "commercial"
    return "review_required"


def extract_condition_scores(image_analysis: dict) -> list[int]:
    scores: list[int] = []
    for item in image_analysis.get("results", []) or []:
        score = item.get("condition_score")
        if isinstance(score, int):
            scores.append(score)
    return scores


def _description_suggests_renovation(description: str) -> bool:
    lowered = description.lower()
    for keyword in RENOVATION_DESCRIPTION_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return True
    return False


def _rag_suggests_renovation(rag_result: dict) -> bool:
    for listing in rag_result.get("similar_listings", []) or []:
        if listing.get("condition") in RENOVATION_RAG_CONDITIONS:
            return True
    return False


def build_renovation_insights(
    description: str, rag_result: dict, image_analysis: dict
) -> list[str]:
    insights: list[str] = []
    scores = extract_condition_scores(image_analysis)
    if any(score <= LOW_CONDITION_THRESHOLD for score in scores):
        insights.append(RENOVATION_IMAGE_INSIGHT)
    if _description_suggests_renovation(description) or _rag_suggests_renovation(
        rag_result
    ):
        insights.append(RENOVATION_TEXT_INSIGHT)
    return insights


def build_property_summary(description: str, route: str) -> str:
    snippet = description.strip()
    if len(snippet) > SUMMARY_SNIPPET_MAX_LEN:
        snippet = snippet[: SUMMARY_SNIPPET_MAX_LEN - 3] + "..."
    route_label = route.replace("_", " ").capitalize()
    return f"{route_label} listing — {snippet}"


def build_recommendations(route: str) -> list[str]:
    if route == "residential":
        return [
            "Compare against recently sold residential listings in the same area.",
            "Confirm furnishing and finish condition with the listing agent.",
        ]
    if route == "commercial":
        return [
            "Verify zoning and permitted uses for the address.",
            "Request occupancy and lease history from the agent.",
        ]
    return [
        "Route to a human reviewer — property type could not be inferred from the description.",
    ]


# --- LLM (Ollama llama3) ---


_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    format="json",
)


def _summarise_rag(rag_result: dict) -> str:
    listings = rag_result.get("similar_listings") or []
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
    insight = rag_result.get("insight")
    if insight:
        lines.append(f"RAG insight: {insight}")
    return "\n".join(lines)


def _summarise_images(image_analysis: dict) -> str:
    results = image_analysis.get("results") or []
    if not results:
        return "No image analysis available."
    lines = []
    for item in results:
        lines.append(
            f"- {item.get('filename', '?')}: "
            f"{item.get('detected_room_type', '?')}, "
            f"condition score {item.get('condition_score', '?')}/5"
        )
    return "\n".join(lines)


def _build_llm_prompt(
    description: str,
    route: str,
    rag_result: dict,
    image_analysis: dict,
) -> str:
    return f"""You are a real estate triage assistant. Analyse the property and \
produce a triage report.

PROPERTY DESCRIPTION:
{description}

SUGGESTED ROUTE: {route}

SIMILAR LISTINGS FROM RAG:
{_summarise_rag(rag_result)}

IMAGE ANALYSIS:
{_summarise_images(image_analysis)}

Return ONLY a JSON object with this exact schema and no other text:
{{
  "property_summary": "one or two sentences summarising the property and how it compares to similar listings",
  "recommendations": ["short actionable item", "..."],
  "renovation_insights": ["specific renovation concern", "..."]
}}

Rules:
- recommendations: 2 to 4 short, actionable items for a real-estate agent reviewing this listing.
- renovation_insights: empty list if nothing in the description, RAG results, or images suggests renovation; otherwise list specific concerns.
- Output valid JSON only. No markdown fences, no commentary."""


def _parse_llm_json(content: str) -> dict | None:
    """Parse the LLM's response into the synthesizer output dict, or None on failure."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    summary = data.get("property_summary")
    recommendations = data.get("recommendations") or []
    insights = data.get("renovation_insights") or []

    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(recommendations, list) or not isinstance(insights, list):
        return None

    return {
        "property_summary": summary.strip(),
        "recommendations": [str(item) for item in recommendations],
        "renovation_insights": [str(item) for item in insights],
    }


def _generate_with_llm(
    description: str,
    route: str,
    rag_result: dict,
    image_analysis: dict,
) -> dict | None:
    """Call Ollama; return the parsed synthesizer dict, or None on any failure."""
    try:
        prompt = _build_llm_prompt(description, route, rag_result, image_analysis)
        response = _llm.invoke(prompt)
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            return None
        return _parse_llm_json(content)
    except Exception:
        return None


# --- LangGraph nodes ---


def planner_node(state: AgentState) -> dict:
    """Pick a routing label and record which upstream tools contributed input."""
    route = determine_route(state.get("description", ""))
    tools_used: list[str] = []
    if state.get("rag_result"):
        tools_used.append("rag_service")
    if state.get("image_analysis"):
        tools_used.append("image_analyser_service")
    return {"suggested_route": route, "tools_used": tools_used}


def synthesizer_node(state: AgentState) -> dict:
    """Generate the human-readable response fields, preferring the LLM over rules."""
    description = state.get("description", "")
    route = state.get("suggested_route", "review_required")
    rag_result = state.get("rag_result", {})
    image_analysis = state.get("image_analysis", {})

    llm_output = _generate_with_llm(description, route, rag_result, image_analysis)
    if llm_output is not None:
        return llm_output

    return {
        "property_summary": build_property_summary(description, route),
        "recommendations": build_recommendations(route),
        "renovation_insights": build_renovation_insights(
            description, rag_result, image_analysis
        ),
    }


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


_compiled_graph = _build_graph()


# --- HTTP surface (contract unchanged) ---


@app.get("/")
def root():
    return {"service": "langgraph-agent-service", "status": "running"}


@app.post("/agent/run")
def run_agent(request: AgentRunRequest):
    initial_state: AgentState = {
        "description": request.description,
        "rag_result": request.rag_result,
        "image_analysis": request.image_analysis,
    }
    final_state = _compiled_graph.invoke(initial_state)
    return {
        "property_summary": final_state["property_summary"],
        "recommendations": final_state["recommendations"],
        "renovation_insights": final_state["renovation_insights"],
        "suggested_route": final_state["suggested_route"],
        "tools_used": final_state["tools_used"],
    }
