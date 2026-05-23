# AI Property Triage System

An AI Engineering final project that automatically triages real estate property listings using a combination of LLM agents, retrieval-augmented generation (RAG), computer vision, and workflow automation.

## Goal

Given a raw property listing (text description, photos, and metadata), the system:

1. **Understands** the listing text using a RAG service grounded on a corpus of reference listings and market data.
2. **Analyses the images** to detect property condition, room types, and visible issues.
3. **Validates** outputs through a guardrails service to filter unsafe, biased, or low-quality responses.
4. **Orchestrates** the full pipeline using a LangGraph agent that decides which tools to call and in what order.
5. **Automates** ingestion, notification, and human-in-the-loop steps via n8n workflows.
6. **Presents** the triaged result (score, summary, flags, recommended action) through a simple web UI.

## Repository Layout

- `webui/` — Frontend / lightweight web app for interacting with the triage system.
- `services/rag-service/` — Retrieval-augmented generation microservice over the listings corpus.
- `services/image-analyser-service/` — Vision microservice for analysing property photos.
- `services/guardrails-service/` — Safety, quality, and policy checks on model outputs.
- `services/langgraph-agent-service/` — LangGraph orchestration agent that ties the services together.
- `n8n/` — n8n workflow exports and documentation.
- `docs/` — Architecture notes, prompt engineering log, and other project documentation.
- `data/synthetic-listings/` — Synthetic property listing text data used for development and evaluation.
- `data/image-dataset/` — Property image dataset used by the image analyser.

## Status

This repository currently contains only the initial scaffolding. Business logic, models, and evaluation harnesses will be added in subsequent iterations.
