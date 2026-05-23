# Architecture Notes

Working notes on the system architecture for the AI Property Triage System.

## Components

- **Web UI** — user-facing entrypoint.
- **LangGraph Agent Service** — orchestrator.
- **RAG Service** — grounded text understanding.
- **Image Analyser Service** — vision signals from property photos.
- **Guardrails Service** — output validation.
- **n8n** — workflow automation and human-in-the-loop.

## Open Questions

- Synchronous vs. asynchronous orchestration between the agent and downstream services.
- Storage choice for the listings vector index.
- Model selection per service (hosted vs. self-hosted).
- Evaluation harness and golden dataset format.

_(notes to be expanded as design decisions are made)_
