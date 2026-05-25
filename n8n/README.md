# n8n Workflows

Workflow automation around the AI Property Triage System. n8n runs alongside
the four FastAPI services and the Streamlit WebUI — it does not replace any
of them. The first workflow exposes the agent through a public webhook so
external systems (forms, CRMs, Zapier, scripts) can submit listings without
going through the WebUI.

## Prerequisites

Before importing or running the workflow, make sure the following are up:

| Service | URL | Notes |
|---------|-----|-------|
| LangGraph Agent | `http://127.0.0.1:8004` | Required — n8n calls `POST /agent/run`. |
| RAG service | `http://127.0.0.1:8001` | Called internally by the agent. |
| Image Analyzer | `http://127.0.0.1:8003` | Not used by this first flow (text-only). |
| Guardrails service | `http://127.0.0.1:8002` | Used by Flow 03 to prefilter input before the agent runs. |
| Ollama (llama3) | `http://127.0.0.1:11434` | Used by the agent for synthesis. |

> The first webhook intentionally **bypasses the input Guardrails service**
> so it stays minimal. Wire it in once the basic path is verified
> (see "Next iterations" below).

## Flow 01 — `agent-webhook`

```
[ Webhook Trigger ]  →  [ HTTP Request → Agent ]  →  [ Respond to Webhook ]
```

### 1. Webhook Trigger

Receives the incoming HTTP request that kicks the flow off.

| Setting | Value |
|---------|-------|
| Node type | **Webhook** |
| HTTP Method | `POST` |
| Path | `agent-webhook` |
| Authentication | `None` (add Basic / Header auth before exposing publicly) |
| Response Mode | `Using 'Respond to Webhook' Node` |

**Expected request body:**

```json
{
  "description": "Renovated 3-room apartment near the beach in Bat Yam, balcony, parking.",
  "agent_name": "Dana Levi"
}
```

n8n exposes this body as `{{ $json.description }}` and `{{ $json.agent_name }}`
inside downstream nodes.

### 2. HTTP Request → Agent

Forwards the description to the LangGraph Agent service, which autonomously
calls RAG and synthesizes the analysis via Ollama llama3.

| Setting | Value |
|---------|-------|
| Node type | **HTTP Request** |
| Method | `POST` |
| URL | `http://127.0.0.1:8004/agent/run` |
| Authentication | `None` |
| Send Body | `On` |
| Body Content Type | `JSON` |
| Specify Body | `Using JSON` |
| JSON Body | see below |
| Response → Response Format | `JSON` |
| Options → Timeout | `180000` (ms) — Ollama can be slow on the first call |

**JSON Body** (paste into the *JSON* field; n8n evaluates the expressions):

```json
{
  "description": "{{ $json.description }}",
  "rag_result": {},
  "image_analysis": {}
}
```

Sending `rag_result: {}` and `image_analysis: {}` tells the agent to fetch
RAG itself and to skip image analysis (no image bytes are flowing through
this first webhook).

### 3. Respond to Webhook

Returns the agent's JSON response to whoever called the webhook.

| Setting | Value |
|---------|-------|
| Node type | **Respond to Webhook** |
| Respond With | `JSON` |
| Response Body | `{{ $json }}` |
| Response Code | `200` |

`$json` here is whatever the HTTP Request node returned, i.e. the full
agent response:

```json
{
  "property_summary": "...",
  "recommendations": ["..."],
  "renovation_insights": ["..."],
  "suggested_route": "residential",
  "tools_used": ["rag_service"],
  "validation": {
    "unsupported_claims": [],
    "risky_claims_detected": false,
    "confidence_level": "medium",
    "validation_passed": true
  },
  "rag_result": { "similar_listings": [...], "insight": "..." }
}
```

## Flow 02 — `agent-webhook-with-images`

```
[ Webhook Trigger ]  →  [ HTTP Request → Agent /agent/run-with-images ]  →  [ Respond to Webhook ]
```

Variant of Flow 01 that accepts **multipart/form-data** so external callers
can submit property photos alongside the description. The agent's
`/agent/run-with-images` endpoint then handles RAG retrieval **and** real
image analysis (CLIP-based) end-to-end.

### 1. Webhook Trigger

| Setting | Value |
|---------|-------|
| Node type | **Webhook** |
| HTTP Method | `POST` |
| Path | `agent-webhook-with-images` |
| Authentication | `None` (add Basic / Header auth before exposing publicly) |
| Response Mode | `Using 'Respond to Webhook' Node` |

Callers send a **`multipart/form-data`** request — text fields plus one or
more files. n8n exposes:

- Text fields under `{{ $json.body.<field> }}` — e.g. `{{ $json.body.description }}`.
- Each uploaded file under the **Binary** section of the node output,
  keyed by the form-field name used in the request (e.g. `data`).

### 2. HTTP Request → Agent `/agent/run-with-images`

Forwards the description, agent name, and binary image(s) to the agent
service as multipart.

| Setting | Value |
|---------|-------|
| Node type | **HTTP Request** |
| Method | `POST` |
| URL | `http://127.0.0.1:8004/agent/run-with-images` |
| Authentication | `None` |
| Send Body | `On` |
| Body Content Type | `Form-Data` |
| Response → Response Format | `JSON` |
| Options → Timeout | `180000` (ms) — covers RAG + CLIP + Ollama in one round trip |

**Body Parameters** — add three rows in the *Body Parameters* table:

| Name | Parameter Type | Value |
|------|----------------|-------|
| `description` | *String* | `{{ $json.body.description }}` |
| `agent_name`  | *String* | `{{ $json.body.agent_name }}` |
| `files`       | *n8n Binary File* | Binary Property: `data` |

The agent expects the file field to be named **`files`** (matching its
FastAPI `files: List[UploadFile]` parameter). The *Binary Property* value
`data` refers to the property name n8n created on the webhook output when
the upstream caller sent the file under that form-field name (see the
curl example below).

> Sending multiple images: have the caller upload them under the same
> form-field name (`data`, `data1`, …) and add a row per binary property,
> all named `files` on the agent side. n8n's HTTP Request node sends
> repeated form fields when the same name is reused.

### 3. Respond to Webhook

| Setting | Value |
|---------|-------|
| Node type | **Respond to Webhook** |
| Respond With | `JSON` |
| Response Body | `{{ $json }}` |
| Response Code | `200` |

`$json` is whatever the HTTP Request node returned — the full agent
response, now richer because both tools ran:

- `tools_used` includes both `"rag_service"` and `"image_analyser_service"`.
- `rag_result` is present (similar listings + insight).
- `image_analysis` is present (per-file `detected_room_type`,
  `condition_score`, `confidence`).
- `validation` is present (`unsupported_claims`, `risky_claims_detected`,
  `confidence_level`, `validation_passed`).

## Testing Flow 02

The path differs from Flow 01 (`agent-webhook-with-images` vs
`agent-webhook`); everything else mirrors n8n's usual test/production URL
rules.

### Windows CMD

```cmd
curl -X POST http://127.0.0.1:5678/webhook/agent-webhook-with-images ^
  -F "description=Luxury villa in Caesarea with swimming pool and smart-home system" ^
  -F "agent_name=Mohammad Hajuj" ^
  -F "data=@C:\Users\moham\OneDrive\Documents\rooms\501LesesneExterior-KeenEyeMarketing1-1920px.jpg"
```

Note the form-field name `data=@...` — that's what becomes the binary
property name on the webhook output, which the HTTP Request node then
references via *Binary Property: `data`*. If you change the field name
in the request, update the HTTP Request node accordingly.

### Expected response shape

```json
{
  "property_summary": "...",
  "recommendations": ["..."],
  "renovation_insights": ["..."],
  "suggested_route": "residential",
  "tools_used": ["rag_service", "image_analyser_service"],
  "rag_result": { "similar_listings": [...], "insight": "..." },
  "image_analysis": {
    "results": [
      {
        "filename": "501LesesneExterior-KeenEyeMarketing1-1920px.jpg",
        "detected_room_type": "exterior",
        "condition_score": 4,
        "confidence": 0.83
      }
    ]
  },
  "validation": {
    "unsupported_claims": [],
    "risky_claims_detected": false,
    "confidence_level": "high",
    "validation_passed": true
  }
}
```

## Testing Flow 01

Once the workflow is activated, n8n shows the production URL — something
like `http://127.0.0.1:5678/webhook/agent-webhook`. The test URL (with
`/webhook-test/` instead of `/webhook/`) only fires once after you click
*Execute Workflow* in the editor.

### curl

```bash
curl -X POST http://127.0.0.1:5678/webhook/agent-webhook \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Renovated 3-room apartment near the beach in Bat Yam, balcony, parking.",
    "agent_name": "Dana Levi"
  }'
```

### PowerShell

```powershell
$body = @{
  description = "Renovated 3-room apartment near the beach in Bat Yam, balcony, parking."
  agent_name  = "Dana Levi"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:5678/webhook/agent-webhook `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

### Postman

1. New `POST` request to `http://127.0.0.1:5678/webhook/agent-webhook`.
2. **Body** → *raw* → *JSON*, paste the sample body above.
3. Send. The agent's JSON response is the body of the reply.

## Flow 03 — `guardrails-prefilter`

> **Status: Implemented, published, and tested in n8n.**

```
[ Webhook ]
   └→ [ HTTP Request → Guardrails ]
         └→ [ IF: approved? ]
               ├── true  → [ HTTP Request → Agent ] → [ Respond to Webhook ]
               └── false → [ Respond to Webhook (rejected) ]
```

Wraps the agent in a cheap, rule-based safety check. The Guardrails
service is a few milliseconds of regex/keyword matching; the Agent is
RAG retrieval, CLIP inference (if images), and an Ollama LLM call. Running
guardrails *before* the agent means obviously off-topic, spammy, or
malformed input never burns the expensive path.

### Why prefilter before the agent?

- **Cost / latency.** A single agent run can take 30–180 s on CPU
  (model load + LLM generation). Guardrails returns in tens of
  milliseconds. Rejecting `"buy crypto now"` should not require waking
  llama3.
- **Abuse surface.** A public webhook is a free LLM endpoint to anyone
  who finds the URL. Guardrails caps that exposure with a clear pre-LLM
  gate that's easy to audit and extend.
- **Quality signal in the output.** Output validation (`validation`
  block) catches risky/unsupported claims the LLM might emit; **input**
  guardrails stop bad input from reaching the LLM in the first place.
  The two layers are independent and complement each other.
- **Clear blame in logs.** Failures are tagged at the boundary: a
  rejected payload never appears in the agent service logs at all,
  making post-incident triage simpler.

### 1. Webhook Trigger

| Setting | Value |
|---------|-------|
| Node type | **Webhook** |
| HTTP Method | `POST` |
| Path | `guardrails-prefilter` |
| Authentication | `None` (add Basic / Header auth before exposing publicly) |
| Response Mode | `Using 'Respond to Webhook' Node` |

**Expected request body:**

```json
{
  "description": "Renovated 3-room apartment near the beach in Bat Yam, balcony, parking.",
  "agent_name": "Dana Levi"
}
```

### 2. HTTP Request → Guardrails

| Setting | Value |
|---------|-------|
| Node type | **HTTP Request** |
| Method | `POST` |
| URL | `http://127.0.0.1:8002/check/input` |
| Authentication | `None` |
| Send Body | `On` |
| Body Content Type | `JSON` |
| Specify Body | `Using JSON` |
| JSON Body | see below |
| Response → Response Format | `JSON` |
| Options → Timeout | `5000` (ms) — guardrails is local + rule-based |

**JSON Body:**

```json
{
  "text": "{{ $json.description }}"
}
```

> The guardrails service field is `text` (not `description`). Make sure
> the JSON body uses that key, otherwise the request will fail validation.

**Service response shape:**

```json
{ "pass": true,  "reason": null }
```

or

```json
{ "pass": false, "reason": "Text contains spam-like phrase: 'buy crypto'." }
```

> The boolean field is named **`pass`** (a Python-reserved-word
> workaround in the service). Use `{{$json.pass}}` in the IF node
> below — not `approved`.

### 3. IF — approval gate

| Setting | Value |
|---------|-------|
| Node type | **IF** |
| Conditions | one boolean condition |
| Condition value 1 | `{{ $json.pass }}` |
| Operator | `Equal` |
| Condition value 2 | `true` (Boolean) |

The IF node exposes two output branches: **true** (approved) and **false**
(rejected).

### 4a. Approved branch — HTTP Request → Agent

Connect the IF node's **true** output to a new HTTP Request node
identical to Flow 01's agent call:

| Setting | Value |
|---------|-------|
| Node type | **HTTP Request** |
| Method | `POST` |
| URL | `http://127.0.0.1:8004/agent/run` |
| Body Content Type | `JSON` |
| JSON Body | see below |
| Options → Timeout | `180000` (ms) |

**JSON Body:**

```json
{
  "description": "{{ $('Webhook').item.json.description }}",
  "rag_result": {},
  "image_analysis": {}
}
```

The `$('Webhook')` expression pulls `description` from the *original*
webhook payload rather than from the Guardrails response (which only
contains `pass` / `reason`).

### 4b. Approved branch — Respond to Webhook

| Setting | Value |
|---------|-------|
| Node type | **Respond to Webhook** |
| Respond With | `JSON` |
| Response Body | `{{ $json }}` |
| Response Code | `200` |

Returns the full agent response (`property_summary`, `recommendations`,
`tools_used`, `validation`, etc.) — the same shape Flow 01 returns.

### 5. Rejected branch — Respond to Webhook

Connect the IF node's **false** output to a second Respond to Webhook node:

| Setting | Value |
|---------|-------|
| Node type | **Respond to Webhook** |
| Respond With | `JSON` |
| Response Body | see below |
| Response Code | `400` |

**Response Body:**

```json
{
  "status": "rejected",
  "reason": "{{ $json.reason }}"
}
```

`$json.reason` here is the human-readable string the guardrails service
returns (e.g. `"Text is too short (minimum 15 characters)."`,
`"Text contains spam-like phrase: 'buy crypto'."`,
`"Text appears off-topic — no real-estate-related keywords detected."`).

## Testing Flow 03

The webhook URL is `http://127.0.0.1:5678/webhook/guardrails-prefilter`.

### Approved input (passes guardrails)

```bash
curl -X POST http://127.0.0.1:5678/webhook/guardrails-prefilter \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Renovated 3-room apartment near the beach in Bat Yam, balcony, parking.",
    "agent_name": "Dana Levi"
  }'
```

Expected response — the full agent payload, same shape as Flow 01:

```json
{
  "property_summary": "...",
  "recommendations": ["..."],
  "renovation_insights": ["..."],
  "suggested_route": "residential",
  "tools_used": ["rag_service"],
  "rag_result": { "similar_listings": [...], "insight": "..." },
  "validation": {
    "unsupported_claims": [],
    "risky_claims_detected": false,
    "confidence_level": "medium",
    "validation_passed": true
  }
}
```

### Rejected input — spam phrase

```bash
curl -X POST http://127.0.0.1:5678/webhook/guardrails-prefilter \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Buy crypto now and get a free apartment in Tel Aviv!",
    "agent_name": "Spammer Inc."
  }'
```

Expected response (HTTP 400):

```json
{
  "status": "rejected",
  "reason": "Text contains spam-like phrase: 'buy crypto'."
}
```

### Rejected input — too short

```bash
curl -X POST http://127.0.0.1:5678/webhook/guardrails-prefilter \
  -H "Content-Type: application/json" \
  -d '{ "description": "nice flat", "agent_name": "X" }'
```

Expected response (HTTP 400):

```json
{
  "status": "rejected",
  "reason": "Text is too short (minimum 15 characters)."
}
```

### Rejected input — off-topic

```bash
curl -X POST http://127.0.0.1:5678/webhook/guardrails-prefilter \
  -H "Content-Type: application/json" \
  -d '{
    "description": "I really enjoyed the latest Marvel movie, the visual effects were stunning.",
    "agent_name": "Cinephile"
  }'
```

Expected response (HTTP 400):

```json
{
  "status": "rejected",
  "reason": "Text appears off-topic — no real-estate-related keywords detected."
}
```

The key observation: in all three rejection cases, the agent service is
**never called** — n8n short-circuits at the IF node and the expensive
LLM path is never woken up.

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `ECONNREFUSED 127.0.0.1:8004` | Agent service isn't running. Start `uvicorn main:app --port 8004` in `services/langgraph-agent-service/`. |
| Request times out at 180s | Ollama is loading `llama3` for the first time — retry, or pre-warm with `ollama run llama3` in another terminal. |
| Webhook returns 404 in n8n | Workflow isn't activated, or you're hitting the test URL after it already fired. Activate the workflow and use the production URL. |
| Empty `rag_result` in the response | RAG service is down or hasn't been populated. Run `python services/rag-service/populate_chroma.py`, then restart the RAG service on port 8001. |

## Next iterations

Flows 01–03 are all built, published, and tested in n8n today. Planned
follow-ups (separate workflows; do not modify the existing ones):

1. **`guardrails-prefilter-with-images`** — same gate as Flow 03 but in
   front of `/agent/run-with-images` so image submissions also get a
   cheap text pre-check before CLIP and Ollama run.
2. **`triage-notify`** — fan-out node that posts the agent response to
   Slack / email / a CRM webhook based on `suggested_route`.

> No application code changes are required for any of the n8n work — n8n
> is purely orchestration on top of the existing HTTP surface.
