# Changelog

All notable changes to pagent are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-04-26

### Added

#### Phase 17 — GA Hardening
- **Security:** JWT expiry validation enforced via `options={"verify_exp": True}` in token decode
- **Security:** Configurable CORS origins (`cors_allowed_origins` setting) replacing wildcard `*`
- **Security:** Tool permission enforcement in `ToolRegistry.get_langchain_tools()` and `execute()` — agents only access permitted tools
- **Security:** New settings: `jwt_algorithm`, `jwt_access_token_expire_minutes`, `cors_allowed_origins`
- **docker-compose.prod.yml:** Resource limits (CPU/memory) for all services; JSON logging with rotation
- **Prompt versioning:** `prompts/PROMPT_CHANGELOG.md` tracking all prompt versions and changes
- **DeepEval CI/CD:** `tests/eval/` with golden datasets (4 teams × 3 cases) and eval harnesses
- **GitHub Actions:** `ci.yml` (unit + integration tests, coverage) and `eval.yml` (DeepEval on main)
- **Load test:** `tests/load/chat_stream.js` — k6 script targeting 50 concurrent SSE streams
- **README.md:** Quick start, architecture overview, API reference, contribution guide, config table
- **CHANGELOG.md:** This file

#### Phase 16 — Quota & Rate Limiting
- Token quota tracking: daily/weekly/monthly per user per model
- Sliding window rate limiter (Redis-based, 60 req/min per user)
- `GET /quota` endpoint with usage breakdown
- `QuotaExceededError` raised and mapped to HTTP 429

#### Phase 15 — Background Scheduler
- APScheduler integrated into FastAPI lifespan hooks
- `reminder.py` job: push pending reminders via webhook or log
- `quota_reset.py` job: reset daily tokens at midnight, weekly on Monday

#### Phase 14 — Observability
- Arize Phoenix tracing: auto-instrument all LangGraph nodes via OpenTelemetry
- Prometheus metrics: `pagent_token_count_total`, `pagent_agent_latency_seconds`, `pagent_tool_calls_total`, `pagent_hitl_total`
- DeepEval harness with `AnswerRelevancyMetric` and `FaithfulnessMetric`
- `GET /metrics` Prometheus scrape endpoint

#### Phase 13 — Team 4: Knowledge & Learning
- `KnowledgeScout`: doc/link → Technical Markdown Summary
- `ExperienceArchivist`: Phoenix trace analysis → lessons learned patterns
- `YamlArchitect`: Synthesize findings → YAML guidelines + git commit to `prompts/`

#### Phase 12 — Team 3: Technical Execution
- `VisualInterpreter`: Multimodal Vision — image/Figma → design component spec
- `CodeArchitect`: Code generation following spec and design patterns
- `QualityGatekeeper`: Execute + review → PASS/FAIL verdict

#### Phase 11 — Team 2: Research & Advisory
- `InternalBrain`: Corrective RAG loop on Personal Knowledge Base (pgvector)
- `MarketScout`: Web search + LATS tree expansion for market intelligence
- `StrategicAdvisor`: SWOT synthesis from internal + market research

#### Phase 10 — Team 1: Documentation & Ops
- `DataProcessor`: Multi-modal input (audio/video) → transcript with timestamps
- `Summarizer`: Script → Meeting Minutes + Executive Summary
- `TaskArchitect`: Discussion → Action Items JSON (name, deadline, owner)
- `SyncManager`: Push tasks to Google Calendar / Notion / Jira

#### Phase 9 — Manager Orchestrator
- Top-level `StateGraph`: guardrail → intent_classifier → planner → router → [team] → validator
- LLM-based Dynamic Router: intent → team name
- Plan-and-Execute planner: step-by-step task decomposition
- HITL `interrupt()` node with resume via `POST /chat/{thread_id}/resume`
- Final Validation node: scores output quality before responding

#### Phase 8 — REST API + SSE Streaming
- `POST /chat` → `StreamingResponse` (SSE): per-node ChatEvent delivery
- `POST /chat/{thread_id}/resume` → resume from HITL checkpoint
- Session CRUD: `GET/DELETE /sessions`, `GET /sessions/{id}/checkpoints`
- `GET /quota`, `GET /models`, `GET /health`
- Global error handlers (AgentError → 429/500, ValidationError → 422)
- Redis sliding-window rate limiter middleware

#### Phase 7 — Persistence + Checkpointing
- SQLAlchemy async models: `User`, `Thread`, `TokenUsage`, `Reminder`
- `PostgresSaver` wrapper for LangGraph state checkpointing
- Time-travel: `list_checkpoints()`, `get_checkpoint()`
- Alembic migrations for MySQL (threads, token_usage, users, reminders) and PostgreSQL (documents)

#### Phase 6 — Base Agent + LangGraph Integration
- `BaseAgent(ABC)` with `build_graph()`, `run()`, `stream()`
- `AgentRegistry` auto-discovers agents from `agents/teams/`
- `LLMCache` (Redis semantic cache) integrated into `BaseLLMAdapter`

#### Phase 5 — Reusable Nodes
- `summarization_node`: message condensation
- `few_shot_node`: vector-store example retrieval → prompt injection
- `meta_prompt_node`: LLM-based system prompt improvement
- `self_discovery_node`: capability listing → plan generation

#### Phase 4 — Memory System
- `ShortTermMemory` (Redis): recent messages with TTL
- `LongTermMemory` (pgvector): embed/store/retrieve with Corrective RAG loop
- `EntityMemory` (Redis Hash): persistent fact store per user
- `MemoryCondenser`: automatic conversation condensation

#### Phase 3 — Tool Registry
- 3-layer `ToolRegistry`: register → `get_langchain_tools()` → `execute()`
- MCP Python client bridge for external tool discovery
- Builtin tools: `web_search` (Tavily), `file_reader`, `code_runner` (sandbox), `calendar`, `guidelines`

#### Phase 2 — LLM Adapters
- `GeminiAdapter`: google-generativeai SDK
- `BeeknoeeAdapter`: custom HTTP API with streaming
- `MockAdapter`: seeded deterministic responses, configurable latency/errors, full tool calling
- `LLMRegistry`: register adapters, fallback chain, select by name
- `TokenCounter`: per-provider token counting

#### Phase 1 — Core Primitives
- `AgentState` TypedDict + 4 Team State variants
- `ApiResponse[T]`, `PageResponse[T]` Pydantic generics
- `GuardrailNode`: safety validation before graph entry
- `LLMCache`: Redis semantic cache interface
- `ConflictResolver`: voting, weighted merge, LLM synthesis strategies
- Exception hierarchy: `AgentError`, `ToolError`, `QuotaExceededError`, `HitlRequiredException`, `LLMProviderError`

#### Phase 0 — Project Scaffold
- FastAPI app factory with lifespan hooks
- Pydantic `BaseSettings` from `.env`
- Docker Compose: MySQL (3316), PostgreSQL+pgvector (5442), Redis (6389), Phoenix (6016), Prometheus (9290)
- Multi-database Alembic setup (MySQL + PostgreSQL)
- `GET /health` endpoint
- VSCode debug configuration

---

## Version Tags

| Version | Milestone |
|---------|-----------|
| v0.1.0 | Phase 0–1: Foundation |
| v0.2.0 | Phase 2–3: LLM + Tools |
| v0.3.0 | Phase 4–5: Memory + Nodes |
| v0.4.0 | Phase 6–7: Agent Core |
| v0.5.0 | Phase 8: API Layer |
| v0.6.0 | Phase 9: Manager |
| v0.7.0 | Phase 10–11: Teams 1–2 |
| v0.8.0 | Phase 12–13: Teams 3–4 |
| v0.9.0 | Phase 14–16: Production Infra |
| v1.0.0 | Phase 17: GA |
