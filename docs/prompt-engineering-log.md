# Prompt Engineering Log

Chronological record of prompt iterations, agent behavior changes, and
orchestration decisions across the AI Property Triage System. Each entry
follows the same shape: **Initial approach → Problem → Example failure →
Change → Result.** The intent is to capture not just *what* the prompts
and helpers became but *why* — so future contributors can judge whether
a change is still load-bearing.

## Initial architecture assumptions

When the project started, the working assumptions were:

- **Local-first** — every component (LLM, vector store, vision model,
  workflow runner) had to run on the developer machine. No OpenAI, no
  hosted vector DB, no cloud orchestration. This drove the choice of
  Ollama + ChromaDB + CLIP-from-HF + n8n.
- **Microservices over a monolith** — each capability (Guardrails, RAG,
  Image Analyzer, Agent) is its own FastAPI service. Pays off when one
  component needs to scale or be replaced without disturbing the others;
  hurts during early dev with multiple ports to keep alive.
- **Autonomous agent at the centre** — the WebUI should not orchestrate
  tool calls itself. The Agent service owns the decision of which tool
  to invoke. The WebUI just submits a description (and optionally
  images) and renders whatever comes back.
- **Deterministic shape, probabilistic content** — every response field
  must always be present with a known type, even when the LLM fails.
  Fallback rule-based generators exist for every LLM call.

These assumptions shaped most of the iterations below.

## RAG prompt evolution

### Iteration 1 — Mock dummy response

**Initial approach.** The first `/query` endpoint returned two hardcoded
listings and a fixed insight, so the WebUI could be wired up before any
real retrieval existed.

**Problem.** Useful only as a stub. Any query returned the same two
listings regardless of input.

**Change.** Implemented `populate_chroma.py` to load the 22 synthetic
listings into ChromaDB with `sentence-transformers/all-MiniLM-L6-v2`
embeddings; replaced the mock with a real `collection.query(...)`.

**Result.** Queries now actually retrieve semantically nearby listings.

### Iteration 2 — Flat document text vs labeled fields

**Initial approach.** Tried passing the description text alone as the
document body in ChromaDB.

**Problem.** Semantically, the descriptions are similar across listing
types ("modern apartment with sea view" vs "modern office with sea view").
Embeddings collapsed both into the same neighborhood.

**Change.** `build_document(listing)` now produces a labeled block:

```
Title: ...
Type: apartment
Location: ...
Price: 1850000
Rooms: 3
Features: balcony, elevator, parking, ...
Condition: new
Description: ...
```

The labels embed real signal into the vector — `Type: office` shifts the
embedding away from apartments.

**Result.** Better separation between property types in the vector space,
which helped iteration 3 work.

### Iteration 3 — Type-filtering fix (the "office query returned apartments" bug)

**Initial approach.** Pure semantic top-K with no metadata filter.

**Problem.** Query: `"Modern office space in Tel Aviv with open floor
plan and meeting rooms."` Top result: *modern 3-room apartment with sea
view, Carmel*. The embedding was dominated by "modern" + "Tel Aviv"; the
office signal lost out.

**Change.** Added `detect_property_type(description)` — word-boundary
regex over `("office",)`, `("retail", "shop", "storefront")`,
`("industrial", "warehouse")`, `("villa",)`, `("house",)`, `("apartment",)`.
When a type is detected, the query passes `where={"property_type": ...}`
to ChromaDB. If the filtered query returns nothing, fall back to
unfiltered.

```python
detected_type = detect_property_type(description)
result = _run_query(description, detected_type)
if detected_type and not result["ids"][0]:
    result = _run_query(description, None)
```

Word boundaries matter — naive substring matching would catch `house` in
`warehouse`, routing warehouses to the "house" filter.

**Result.** Office queries now return office listings. Order of
`TYPE_DETECTION` puts more specific keywords first
(`office, retail, industrial` before `villa, house, apartment`) so
ambiguous queries land on the more specific intent.

### Iteration 4 — Insight phrasing for "needs renovation"

**Initial approach.** Single template:
`"This listing is similar to {condition} {property_type_plural} in {city}."`

**Problem.** With `condition == "needs renovation"`, that produced
`"...similar to needs renovation apartments in Haifa."` — ungrammatical.

**Change.** Special-cased that condition value:

```python
if condition == "needs renovation":
    return f"This listing is similar to {plural} in {city} that need renovation."
```

Also added `PROPERTY_TYPE_PLURALS` to handle `retail → "retail spaces"`
and `industrial → "industrial properties"` instead of `retails` /
`industrials`.

**Result.** Natural-sounding insights regardless of the condition value.

## Image analysis prompt evolution

### Iteration 1 — Random mock

**Initial approach.** `random.choice(ROOM_TYPES)` + `random.randint(1, 5)`
per uploaded file. The endpoint contract was real; the body was fake.

**Problem.** Anything you uploaded got a coin-flip room type and a
random condition. The agent's renovation-from-image logic was therefore
firing on noise.

**Change.** Replaced with CLIP zero-shot classification using
`openai/clip-vit-base-patch32` from `transformers`.

**Result.** Real grounding from pixels.

### Iteration 2 — Raw labels vs natural-language prompts

**Initial approach.** Used the bare labels as CLIP text input:
`["kitchen", "bathroom", "bedroom", "living_room", "exterior", "other"]`.

**Problem.** CLIP's text encoder was trained on captions, not labels.
Accuracy was meaningfully worse with one-word prompts.

**Change.** Wrap each label in a short caption:

```python
ROOM_LABEL_PROMPTS = {
    "kitchen": "a photo of a kitchen",
    "bathroom": "a photo of a bathroom",
    "bedroom": "a photo of a bedroom",
    "living_room": "a photo of a living room",
    "exterior": "a photo of a building exterior",
    "other": "a photo of something other than a room",
}
```

Labels and prompts stay aligned by insertion order so the argmax index
maps back to the human-readable name.

**Result.** Higher and more stable confidences from CLIP; the `confidence`
field returned to the agent is now actionable signal, which became the
basis for the validation layer's confidence heuristic later.

### Iteration 3 — Condition heuristic without random

**Initial approach.** Even after swapping in CLIP, `condition_score` was
still a random integer.

**Problem.** Inconsistent with the rest of the response — a real model
output paired with a random score.

**Change.** Deterministic PIL-based heuristic isolated in
`estimate_condition_score(image)`:

- Brightness mapped to "well-lit-ness" that peaks at mid-grey (0.5).
  Over- or under-exposed images score lower even if everything else is
  good.
- Contrast = stddev of the greyscale image, normalised.
- Sharpness = stddev of `ImageFilter.FIND_EDGES` output, normalised.
  Avoided a `cv2` dependency just for variance-of-Laplacian.
- Composite = simple average → 1..5 integer.

**Result.** Repeatable scores driven by real image properties. The
helper is one function, so a future PyTorch model can drop in without
touching the HTTP surface.

## Agent synthesis prompt evolution

### Iteration 1 — Rule-based summaries

**Initial approach.** `build_property_summary(description, route)`
produced `"{route_label} listing — {first 120 chars of description}…"`.
Recommendations were hard-coded per route.

**Problem.** Outputs were correct but generic. The system *had* RAG and
image data but the summary couldn't reflect them.

**Change.** Added `synthesizer_node` calling Ollama llama3 with a
structured prompt that includes the description, route, summarised RAG
listings, and summarised image analysis.

**Result.** Summaries became contextual: "Three-room residential listing
in Bat Yam, comparable to similar renovated apartments in northern
Carmel currently priced ~10% higher."

### Iteration 2 — Ollama JSON mode and the parse problem

**Initial approach.** Just asked the LLM to "return JSON" in plain text.

**Problem.** llama3 reliably wrapped the JSON in markdown fences:

```
Here is the JSON you requested:
```json
{ "property_summary": "..." }
```
```

That broke `json.loads`.

**Change.** Two layers:

1. `ChatOllama(model="llama3", temperature=0.2, format="json")` —
   the `format="json"` passes through to Ollama's native JSON mode and
   strongly suppresses prose.
2. Defensive parser:

   ```python
   try:
       data = json.loads(content)
   except json.JSONDecodeError:
       match = re.search(r"\{.*\}", content, re.DOTALL)
       if match: data = json.loads(match.group(0))
       else:     return None
   ```

   Strict first, regex-extract fallback second.

**Result.** Parse-failure rate dropped to effectively zero across
hundreds of test invocations. The few remaining failures route to the
rule-based fallback so the WebUI still gets a valid response.

### Iteration 3 — Compact summaries inside the prompt

**Initial approach.** Dumped raw `rag_result` and `image_analysis` JSON
into the prompt.

**Problem.** Token waste, and the LLM tended to repeat field names
verbatim ("the rag_result.similar_listings[0] shows...") instead of
reasoning over the content.

**Change.** Two formatter helpers turn the raw payloads into bullet
lines before the prompt is built:

```
SIMILAR LISTINGS FROM RAG:
- apartment in Haifa, Carmel, price 1850000, condition new
- apartment in Bat Yam, Promenade, price 1420000, condition renovated
RAG insight: This listing is similar to renovated apartments in Bat Yam.

IMAGE ANALYSIS:
- kitchen.jpg: kitchen, condition score 4/5
- bath.jpg: bathroom, condition score 3/5
```

**Result.** Shorter prompts, more grounded summaries.

### Iteration 4 — Truthful `tools_used`

**Initial approach.** `tools_used` was hard-coded to
`["rag_service", "image_analyser_service"]` regardless of what actually
ran.

**Problem.** Lied to the consumer. Made the validation layer's
confidence heuristic harder to trust ("both tools ran" was a constant,
not a signal).

**Change.** Compute `tools_used` from the post-tool state at the end of
the synthesizer node:

```python
tools_used = []
if state.get("rag_result"):
    tools_used.append("rag_service")
if state.get("image_analysis"):
    tools_used.append("image_analyser_service")
```

So if RAG failed or images weren't provided, `tools_used` correctly omits
that entry — and downstream, the validation node's confidence drops.

**Result.** `tools_used` became a real signal, used by both the WebUI
and the output validator.

## Conversational assistant prompt evolution

### Iteration 1 — Static placeholder

**Initial approach.** Hard-coded reply:
`"This is a placeholder response from the local real estate assistant."`

**Problem.** Useful as scaffolding only. Not a chat.

**Change.** Direct call to Ollama via `POST /api/generate` (no
LangChain wrapper for this path) with `stream: false` so a single JSON
response comes back.

### Iteration 2 — Grounded in the last triage

**Initial approach.** Started as a generic chat with no context.

**Problem.** Users asked "what's the price?" and the assistant had no
idea what listing they meant.

**Change.** Pipeline saves the last successful triage to
`st.session_state.last_triage`:

```python
st.session_state.last_triage = {
    "description": description,
    "rag_response": rag_response,
    "image_response": image_response,
    "agent_response": agent_response,
}
```

The chat tab reads this and bakes it into the prompt as
`ANALYSED PROPERTY CONTEXT:` — flattened versions of description, RAG
listings, image results, and the agent's summary/recommendations/
insights.

**Result.** "what's the price?" now returns the price from the
description; "are similar listings cheaper?" reasons over the RAG
results.

### Iteration 3 — Off-topic rejection

**Initial approach.** Relied on the model to decline off-topic questions
on its own.

**Problem.** llama3 was happy to answer anything — recipes, history
questions, code generation.

**Change.** Pinned the off-topic response in the system rules:

```
If the user asks about anything unrelated to this property or real estate,
respond with EXACTLY: "I can only assist with questions related to the analysed property."
```

The exact-string instruction makes it both detectable (we can search the
UI for that line) and easy to extend later (one source of truth for the
phrase).

**Result.** Off-topic queries get the exact rejection sentence. On-topic
queries pass through unaffected.

## Guardrails evolution

### Iteration 1 — No guardrails

**Initial approach.** Submit form sent the description straight to RAG.

**Problem.** Sending `""`, `"asdf"`, or `"buy crypto now"` would still
exercise the LLM (slow, wasteful, embarrassing).

**Change.** New `guardrails-service` FastAPI app, `POST /check/input`,
rule-based:

- empty / whitespace → reject
- `< 15 chars` → reject
- contains `"buy crypto" / "free money" / "click here"` → reject
- no real-estate keyword found → reject as off-topic

### Iteration 2 — Real-estate keyword vocabulary

**Initial approach.** Three keywords:
`("apartment", "house", "property")`.

**Problem.** "office space" got rejected as off-topic because none of
the three appeared.

**Change.** Expanded `REAL_ESTATE_KEYWORDS` to cover the common
residential and commercial vocabulary:
`apartment, house, kitchen, bedroom, property, balcony, parking,
room, flat, villa, studio, garden, bathroom, floor`. Still a finite
list — and that's deliberate, because the whole point of guardrails is
predictable behavior over an enumerable vocabulary.

**Result.** Legitimate office and commercial submissions stop getting
rejected as off-topic.

### Iteration 3 — Word-boundary matching

**Initial approach.** Substring matching for everything.

**Problem.** The agent's renovation-detection helper started catching
`old` in words like `gold`, `told`, `household`, flagging unrelated
listings as needing renovation.

**Change.** Switched to word-boundary regex everywhere short keywords
are involved:

```python
if re.search(rf"\b{re.escape(keyword)}\b", lowered):
    return True
```

Applied in both the guardrails service and the agent's
`_description_suggests_renovation`. The same pattern protected the RAG
type-detection from the `warehouse → house` collision.

**Result.** False positives on short keywords disappeared.

## Output validation evolution

The validation layer was added late, after observing that the LLM
occasionally drifted into confident-but-ungrounded claims. It is
deterministic and rule-based — explicitly **not** another LLM call,
to keep the cost and latency profile flat.

### Iteration 1 — No validation

**Problem.** Sample failure: a description of a basic apartment came
back with `"recommendations": ["Highlight the rooftop swimming pool and
gym in marketing materials."]`. There was no pool or gym in the
description, RAG results, or image analysis.

**Change.** Added `validation_node` to the graph:
`planner → tool → synthesizer → validation → END`.

### Iteration 2 — Unsupported claims detection

**Approach.** Curated feature vocabulary `FEATURE_KEYWORDS`
(swimming pool, elevator, sea view, parking, gym, cinema room, …).
`extract_known_features` flattens description + RAG + image analysis
into one haystack and returns the subset of `FEATURE_KEYWORDS` actually
present. `find_unsupported_claims` returns features the LLM output
mentions that aren't in that grounded set.

**Result.** The "rooftop swimming pool" hallucination now lands in
`validation.unsupported_claims: ["swimming pool"]` and the response gets
`validation_passed: false`. Recommendation text is preserved (so the
user sees what was claimed) but the badge clearly says "needs review".

### Iteration 3 — Risky phrases + sanitization

**Approach.** Curated `RISKY_PHRASES` (`guaranteed investment`,
`guaranteed profit`, `no risk`, `perfect investment`,
`always increases in value`, …). When detected:

1. Drop any recommendation lines containing a risky phrase.
2. Append the disclaimer:
   `"AI-generated recommendations should be verified by a human agent."`
3. Set `risky_claims_detected: true`.

Sanitization is intentionally only applied to recommendations — the
summary is kept verbatim so the user can see what was said.

**Result.** Risky lines don't leak into the recommendations card; the
disclaimer is always appended when the model strayed.

### Iteration 4 — Confidence-level heuristic

**Approach.** `compute_confidence_level` produces one of `low | medium |
high`:

| Has RAG | Has images | Avg img conf | Unsupported / Risky | Result |
|---------|------------|--------------|---------------------|--------|
| ✓ | ✓ | ≥ 0.8 | none | `high` |
| ✓ | ✓ | 0.5–0.8 | none | `medium` |
| ✓ | ✓ | < 0.5 | none | `low` |
| ✓ | ✗ | — | none | `medium` |
| ✗ | ✓ | — | none | `medium` |
| ✗ | ✗ | — | none | `low` |
| any | any | any | yes | `low` |

The threshold values (`0.8`, `0.5`) are constants at the top of the
module so they can be retuned without code spelunking. Risky/unsupported
claims override everything else — even if both tools ran perfectly, a
single hallucinated feature drops the confidence to `low`.

**Result.** Confidence is now a real, surfaced signal in the UI rather
than just LLM vibes.

## n8n orchestration decisions

### Flow 01 — text-only webhook

**Decision.** First flow intentionally bypasses guardrails to keep the
hop count minimal and prove the path end-to-end:
`Webhook → HTTP Request → Agent → Respond to Webhook`.

**Reasoning.** Wiring multiple n8n nodes plus per-node auth and timeout
options gets complex quickly. Land the simplest flow first; layer
defensive checks later as separate flows.

### Flow 02 — multipart with binary forwarding

**Decision.** Separate flow for image uploads rather than extending
Flow 01.

**Reasoning.** n8n handles binary differently from JSON — text fields
land under `$json.body.<name>` and files under a Binary Property. Mixing
both in one HTTP Request node only works when the upstream caller
always sends multipart, which Flow 01 callers don't. Two flows are
cleaner than one branched flow.

The agent endpoint `/agent/run-with-images` accepts the multipart with
field name `files` (matching FastAPI's `List[UploadFile]` parameter).
The n8n HTTP Request node uses *Body Parameters → n8n Binary File →
Binary Property: `data`* — that `data` key is the form-field name from
the upstream caller, not a fixed convention.

### Flow 03 — guardrails-prefilter

**Decision.** Cheap gate in front of the expensive path:
`Webhook → Guardrails → IF → (Agent | Reject)`.

**Reasoning.** Guardrails returns in ~10 ms; the agent path (RAG +
optional CLIP + Ollama) can take 30–180 s. Rejecting obvious bad input
at the boundary saves both compute and the n8n execution log from being
clogged with spam.

The IF node uses `{{$json.pass}}` (not `approved`) because the
guardrails service returns `pass` as the boolean field — the Python
keyword would have been illegal as a Pydantic attribute name, so the
service uses it as a JSON field instead. Documented inline so future
flow-builders don't get tripped up.

## Lessons learned

- **Prompt vs code split matters.** Off-topic rejection is enforced by
  the chat prompt; spam rejection is enforced by code. Both are
  necessary. Prompts are good for stylistic and content rules; code is
  better for hard guarantees (length limits, allow-lists, parsing).
- **Always have a deterministic fallback.** Every Ollama call has a
  rule-based fallback that produces the same response shape. The
  contract — what callers see — is invariant; the *quality* degrades
  gracefully when the LLM is unavailable.
- **Tools used must tell the truth.** Anything you report back gets
  trusted by downstream code. Hard-coding `tools_used` once led the
  validator to think we had image grounding when we didn't.
- **Curated vocabularies beat free-form NLP at this scale.** The
  feature, spam, risky, and renovation keyword lists are small and
  hand-curated. That's fine — the system is auditable, predictable,
  and easy to extend.
- **Eager model loading is the right default for a single-tenant
  service.** Loading CLIP / sentence-transformers / the Ollama model on
  first request makes that request painfully slow; loading at startup
  makes the dev loop better and the user-facing latency stable.
- **Word boundaries everywhere.** The same `\b` pattern saved us in
  guardrails, RAG type detection, and renovation insight detection.
  Substring matching seems easier until `house` matches `warehouse`.

## Future prompt-engineering improvements

- **Memory.** Add a persistent conversation memory beyond
  `st.session_state.messages` — e.g. SQLite or Redis — so the chat tab
  remembers prior triaged listings across sessions and can be queried
  ("how does this compare to the Caesarea villa from last week?").
- **Better grounding.** Augment the synthesizer prompt with explicit
  per-claim citations (e.g. `[RAG listing 3]`, `[image kitchen.jpg]`)
  and validate that every assertion in the summary has a citation
  pointing back at the inputs.
- **Hybrid retrieval.** Combine dense embeddings (current) with a
  sparse BM25 retriever over the listings corpus. Specific tokens like
  prices or street names suffer in pure-dense retrieval; BM25 catches
  them.
- **NeMo Guardrails.** Replace the hand-rolled rule-based guardrails
  service with NVIDIA's NeMo Guardrails. The current rule list scales
  poorly; a colang-defined policy is easier to audit and version.
- **Tool-selection reasoning.** Today the planner is a thin rule:
  "RAG if no rag_result; image analyser if uploaded_images." A small
  LLM call inside the planner could decide *which* RAG query to run
  (e.g. reformulate "old beach apartment" into a stricter type query),
  or whether image analysis is worth the latency for this submission.
- **Multi-agent specialization.** Split the synthesizer into
  domain-specialised sub-agents (residential vs commercial vs
  industrial), each with its own prompt and its own RAG slice, and let
  the planner route. The current single-agent prompt is biased toward
  generic real-estate advice; a commercial specialist would catch
  zoning, lease, and occupancy concerns the generic prompt sometimes
  misses.
