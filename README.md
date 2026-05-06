# pagent

Personal AI Agent — a multi-agent orchestration system powered by LangGraph, FastAPI, and Google Gemini. Four specialized teams of agents handle documentation, research, technical execution, and knowledge management through a Manager orchestrator.

## Architecture

```
User Request
    └── Manager Orchestrator (intent → plan → route → validate)
            ├── Team 1: Documentation & Ops
            │   └── DataProcessor → Summarizer → TaskArchitect → SyncManager
            ├── Team 2: Research & Advisory
            │   └── InternalBrain (RAG) → MarketScout (Web+LATS) → StrategicAdvisor
            ├── Team 3: Technical Execution
            │   └── VisualInterpreter → CodeArchitect → QualityGatekeeper
            └── Team 4: Knowledge & Learning
                └── KnowledgeScout → ExperienceArchivist → YamlArchitect
```

**Tech stack:**

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI 0.115 + uvicorn |
| Orchestration | LangGraph 0.2 + LangChain 0.3 |
| LLM | Google Gemini (primary), Beeknoee (fallback), Mock (test) |
| Memory | Redis (short-term) + pgvector (long-term RAG) |
| Persistence | MySQL (sessions, quota) + PostgreSQL (checkpoints, documents) |
| Observability | Arize Phoenix (tracing) + Prometheus + DeepEval |
| Scheduler | APScheduler (reminders, quota reset) |

## Quick Start

### 1. Clone and configure environment

```bash
git clone <repo-url>
cd pagent
cp .env.example .env
# Edit .env: set JWT_SECRET, GEMINI_API_KEY, and other required values
```

### 2. Start infrastructure services

```bash
docker-compose up -d
# Services: MySQL (3316), PostgreSQL+pgvector (5442), Redis (6389), Phoenix (6016), Prometheus (9290)
```

### 3. Install dependencies

```bash
pip install uv
uv sync
```

### 4. Run database migrations

```bash
uv run alembic -c alembic/mysql/alembic.ini upgrade head
uv run alembic -c alembic/postgres/alembic.ini upgrade head
```

### 5. Start the app

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8200 --reload
```

### 6. Verify

```bash
curl http://localhost:8200/health
# {"status": "ok"}
```

Open Swagger UI: http://localhost:8200/docs

## API Reference

### Public endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

### Protected endpoints (requires `Authorization: Bearer <token>`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send message → SSE stream of agent events |
| POST | `/chat/{thread_id}/resume` | Resume after HITL pause |
| GET | `/sessions` | List chat sessions |
| GET | `/sessions/{id}/checkpoints` | Time-travel checkpoints |
| GET | `/quota` | Token usage breakdown |
| GET | `/models` | Available LLM models |

### SSE event format
```json
{"type": "token", "content": "Hello"}
{"type": "node_complete", "node": "summarizer", "output": "..."}
{"type": "awaiting_approval", "reason": "...", "thread_id": "..."}
{"type": "done", "final": "..."}
{"type": "error", "message": "..."}
```

## Development

### Run tests

```bash
uv run pytest tests/unit/ -v           # Unit tests (fast, no infra)
uv run pytest tests/integration/ -v   # Integration tests (requires Redis)
uv run pytest tests/eval/ -v          # DeepEval harness (requires GEMINI_API_KEY)
```

### Debug in VSCode

Press **F5** — the `.vscode/launch.json` is pre-configured for uvicorn with breakpoint support.

### Project structure

```
src/
├── agents/       # LangGraph orchestration (manager + 4 teams × 13 agents)
├── api/          # REST API (routers, DTOs, middleware, deps)
├── core/         # Domain types (state, messages, exceptions, guardrails)
├── llm/          # LLM adapters (Gemini, Beeknoee, Mock)
├── memory/       # Short-term (Redis), long-term (pgvector RAG), entity
├── nodes/        # Reusable LangGraph nodes
├── persistence/  # SQLAlchemy models, repositories, checkpointer
├── tools/        # Tool registry + builtins (web search, code runner, etc.)
├── scheduler/    # APScheduler jobs (reminders, quota reset)
└── observability/ # Phoenix tracing, Prometheus metrics, DeepEval
```

## Configuration

All settings are read from `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | _(required)_ | ≥32 char secret for JWT signing |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Access token TTL |
| `CORS_ALLOWED_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins (JSON array) |
| `MYSQL_URL` | `mysql+aiomysql://pagent:pagent@localhost:3316/pagent` | Sessions + quota DB |
| `POSTGRES_URL` | `postgresql+asyncpg://pagent:pagent@localhost:5442/pagent` | Checkpoints + documents |
| `REDIS_URL` | `redis://localhost:6389/0` | Cache + rate limiting |
| `GEMINI_API_KEY` | _(required for prod)_ | Google Gemini API key |
| `QUOTA_DAILY_TOKEN_LIMIT` | `100000` | Max tokens per user per day |
| `OTEL_ENABLED` | `true` | Enable Phoenix tracing |

## Production deployment

```bash
# Copy and configure production env
cp .env.example .env
# Set: JWT_SECRET (strong random), DB passwords, GEMINI_API_KEY, CORS origins

docker-compose -f docker-compose.prod.yml up -d

# Verify
curl http://localhost:8200/health
```

Resource limits are pre-configured: app (2 CPU / 2GB RAM), MySQL (1GB), PostgreSQL (1GB), Redis (256MB).

## Contribution Guide

### Branch naming
- `feat/description` — new feature
- `fix/description` — bug fix
- `chore/description` — maintenance, deps, tooling

### PR process
1. Branch off `main`
2. Write tests for new code
3. Run `pytest tests/unit/ tests/integration/` — all must pass
4. PR title: `feat: ...` / `fix: ...` following Conventional Commits
5. At least one reviewer approval required

### Commit conventions
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add streaming support for knowledge team
fix: correct JWT expiry validation
chore: bump langchain to 0.3.5
```

## Observability

- **Traces:** http://localhost:6016 — Arize Phoenix UI
- **Metrics:** http://localhost:9290 — Prometheus
- **API docs:** http://localhost:8200/docs — Swagger UI
- **Metrics endpoint:** http://localhost:8200/metrics — Prometheus scrape target
