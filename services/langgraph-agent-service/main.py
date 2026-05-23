"""LangGraph agent service — minimal FastAPI skeleton with mock reasoning.

Today this service runs simple keyword-based rules to produce an agent-style
analysis. The real LangGraph orchestration (tool calls, multi-step reasoning,
state) will replace the body of `run_agent` later — input and output shapes
are intended to stay stable so downstream consumers don't have to change.
"""

import re

from fastapi import FastAPI
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


class AgentRunRequest(BaseModel):
    description: str
    rag_result: dict = Field(default_factory=dict)
    image_analysis: dict = Field(default_factory=dict)


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
        # Word-boundary match so short keywords like "old" don't fire on
        # "gold", "told", "household", etc.
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


@app.get("/")
def root():
    return {"service": "langgraph-agent-service", "status": "running"}


@app.post("/agent/run")
def run_agent(request: AgentRunRequest):
    route = determine_route(request.description)
    return {
        "property_summary": build_property_summary(request.description, route),
        "recommendations": build_recommendations(route),
        "renovation_insights": build_renovation_insights(
            request.description, request.rag_result, request.image_analysis
        ),
        "suggested_route": route,
        "tools_used": ["rag_service", "image_analyser_service"],
    }
