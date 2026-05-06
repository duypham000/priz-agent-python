# AI_ARCH.MD — pagent Architecture Reference

> Tài liệu này là nguồn sự thật duy nhất về kiến trúc của `pagent`.
> Mọi AI assistant (Claude, Aider, Copilot...) và developer đều phải đọc file này trước khi đóng góp code.

---

## 1. Tổng quan project

`pagent` là Python AI Agent Orchestration Service, thay thế hoàn toàn service `agent/` (Spring Boot). Được thiết kế từ đầu theo mô hình của các hệ thống AI agent lớn (CrewAI, Claude Code, LangGraph Supervisor, Agno).

**Chức năng cốt lõi:**
- Tiếp nhận yêu cầu từ người dùng qua REST API (SSE streaming)
- Phân tích intent → lập kế hoạch → phân phối cho các agent team
- 13 specialized agents tổ chức thành 4 teams dưới 1 Manager
- Human-in-the-Loop (HITL): dừng chờ phê duyệt rồi tiếp tục
- Quản lý memory (short-term, long-term vector, entity)
- Theo dõi token quota theo model/user/ngày/tuần/tháng

---

## 2. Stack kỹ thuật

| Concern | Technology | Ghi chú |
|---------|-----------|---------|
| API Framework | FastAPI (async) | SSE streaming, JWT auth |
| Agent Orchestration | LangGraph StateGraph + Supervisor | Core execution engine |
| LLM Abstraction | LangChain `BaseChatModel` | Unified interface cho mọi provider |
| LLM Providers | Gemini, Beeknoee, **Mock** | Anthropic/OpenAI optional |
| State Persistence | LangGraph `PostgresSaver` | Checkpointing + time-travel |
| Vector Memory | pgvector (PostgreSQL extension) | Long-term RAG |
| Short-term Memory | Redis | Conversation buffer |
| Entity Memory | Redis Hash | Structured facts về entities |
| App DB | MySQL | Sessions, quota, users |
| LLM Caching | LangChain Redis semantic cache | Tránh duplicate LLM calls |
| Observability | Arize Phoenix + OTEL | Distributed tracing |
| Evaluation | DeepEval | LLM-as-judge |
| Metrics | Prometheus | Token usage, latency |
| Prompt Management | YAML files + Git | Versioned, versionable |
| Tool Protocol | MCP Python client | Model Context Protocol |
| Background Jobs | APScheduler | Reminders, quota reset |
| Package Manager | uv | Fast Python dep management |
| DB Migrations | Alembic (multi-DB) | MySQL + Postgres riêng nhau |
| Testing | pytest + pytest-asyncio | Unit + integration |

---

## 3. Hexagonal Architecture

`pagent` áp dụng Hexagonal Architecture (Ports & Adapters) theo style Python AI ecosystem — không rigid như Java nhưng giữ nguyên triết lý phân tách trách nhiệm:

```
┌─────────────────────────────────────────────────────────┐
│                    INBOUND ADAPTERS                      │
│              api/ (FastAPI routers, SSE)                 │
├─────────────────────────────────────────────────────────┤
│                    APPLICATION CORE                      │
│         agents/ (LangGraph graphs, orchestration)        │
│         nodes/ (reusable node functions)                 │
├──────────────────────┬──────────────────────────────────┤
│      DOMAIN CORE     │     CROSS-CUTTING CONCERNS       │
│   core/ (state,      │  observability/ (tracing, eval)  │
│   messages,          │  scheduler/ (background jobs)    │
│   exceptions)        │                                  │
├──────────────────────┴──────────────────────────────────┤
│                   OUTBOUND ADAPTERS                      │
│  llm/          tools/       memory/      persistence/   │
│  (providers)   (registry)   (Redis+PG)   (SQLAlchemy)   │
└─────────────────────────────────────────────────────────┘
```

| Hex Layer | Module | Nội dung |
|-----------|--------|---------|
| **Domain** | `src/core/` | AgentState, message types, exceptions, response wrappers |
| **Application** | `src/agents/` | LangGraph graphs, agent logic, business orchestration |
| **Inbound Adapters** | `src/api/` | FastAPI routers, DTOs, middleware, auth dependencies |
| **Outbound Adapters** | `src/llm/`, `src/tools/`, `src/memory/`, `src/persistence/` | LLM calls, tool execution, DB access |
| **Cross-cutting** | `src/observability/`, `src/nodes/`, `src/scheduler/` | Tracing, reusable nodes, background jobs |

---

## 4. Cấu trúc thư mục

```
pagent/
├── pyproject.toml                   # uv deps: langgraph, fastapi, langchain-google-genai, ...
├── Dockerfile                       # Python 3.12, uv install, src/ entrypoint
├── docker-compose.yml               # Infrastructure only (không có app) — dùng cho local dev
├── docker-compose.prod.yml          # Production: includes app service
├── .vscode/
│   ├── launch.json                  # Debug: uvicorn src.main:app --reload --port 8200
│   └── settings.json                # Python interpreter: .venv/Scripts/python
├── docker/
│   ├── mysql/init.sql               # MySQL schema init
│   ├── postgres/init.sql            # pgvector extension + schema
│   └── phoenix/config.yaml          # Arize Phoenix server config
├── alembic/
│   ├── mysql/                       # MySQL migrations (sessions, quota, users)
│   └── postgres/                    # Postgres migrations (pgvector, checkpointer tables)
├── .env.example                     # All configurable values với non-default ports
├── tests/
│   ├── conftest.py                  # pytest fixtures, mock LLM factory
│   ├── unit/
│   │   ├── test_llm_mock.py
│   │   ├── test_tool_registry.py
│   │   ├── test_memory.py
│   │   └── test_nodes.py
│   └── integration/
│       ├── test_manager_graph.py    # Full graph với mock LLM
│       └── test_api_chat.py         # FastAPI TestClient SSE
├── config/
│   ├── agents.yaml                  # Agent definitions (xem mục 9)
│   └── tasks.yaml                   # Task templates
├── prompts/                         # YAML prompt files, versioned bằng Git
│   ├── manager/system.yaml
│   └── teams/{docs,research,technical,knowledge}/
│
└── src/
    ├── main.py                      # FastAPI app factory, lifespan hooks
    ├── settings.py                  # Pydantic BaseSettings (env-driven)
    │
    ├── core/                        # DOMAIN CORE
    │   ├── state.py                 # AgentState (global), Team States (per team)
    │   ├── messages.py              # HumanMessage, AIMessage, ToolMessage wrappers
    │   ├── exceptions.py            # AgentError, ToolError, QuotaExceededError, ...
    │   ├── response.py              # ApiResponse[T], PageResponse[T]
    │   ├── guardrails.py            # Node Guardrail — inject safety check vào graph
    │   ├── caching.py               # LLM semantic cache (Redis) + exact cache
    │   └── conflict.py              # Conflict Resolution: voting, weighting, LLM synthesis
    │
    ├── llm/                         # OUTBOUND: LLM ADAPTER
    │   ├── base.py                  # BaseLLMAdapter(ABC): get_model(), count_tokens()
    │   ├── registry.py              # LLMRegistry: register → select by name → fallback chain
    │   ├── token_counter.py         # Token counting per provider (dùng cho quota)
    │   ├── gemini.py                # GeminiAdapter (google-generativeai)
    │   ├── beeknoee.py              # BeeknoeeAdapter (custom API)
    │   └── mock.py                  # MockAdapter: GenericFakeChatModel, deterministic, latency
    │
    ├── tools/                       # OUTBOUND: TOOL REGISTRY
    │   ├── registry.py              # ToolRegistry: register → represent → execute (3-layer)
    │   ├── mcp/
    │   │   └── client.py            # MCP Python client bridge
    │   └── builtin/
    │       ├── guidelines.py        # Đọc YAML từ prompts/ → trả về string
    │       ├── web_search.py        # Tavily / Perplexity search
    │       ├── file_reader.py       # Đọc file/URL → text
    │       ├── calendar.py          # Google Calendar / Notion / Jira API
    │       └── code_runner.py       # Chạy code trong sandbox (Team 3)
    │
    ├── memory/                      # OUTBOUND: MEMORY SYSTEM
    │   ├── short_term.py            # Redis-based: lưu N messages gần nhất
    │   ├── long_term.py             # pgvector RAG: embed → store → retrieve → grade → rewrite
    │   ├── entity_memory.py         # Redis Hash: facts về entities (người, project, ...)
    │   └── condensation.py          # Memory Pruning: tóm tắt lịch sử cũ khi vượt ngưỡng
    │
    ├── nodes/                       # CROSS-CUTTING: REUSABLE NODES
    │   ├── summarization.py         # Summarization Node (dùng lại trong nhiều teams)
    │   ├── few_shot.py              # Dynamic Few-Shot: lấy examples từ vector store
    │   ├── meta_prompt.py           # Meta-Prompting: tự sinh/cải thiện prompt
    │   └── self_discovery.py        # Self-Discovery: agent reflect on capabilities → plan
    │
    ├── agents/                      # APPLICATION: AGENT GRAPHS
    │   ├── base.py                  # BaseAgent: build_graph() → CompiledGraph
    │   ├── registry.py              # AgentRegistry: auto-discover tất cả agents
    │   ├── manager/
    │   │   ├── state.py             # ManagerState TypedDict
    │   │   ├── graph.py             # Top-level StateGraph + HITL interrupt()
    │   │   ├── router.py            # Dynamic Router: intent → team name
    │   │   ├── planner.py           # Plan-and-Execute: tạo step-by-step plan
    │   │   └── validator.py         # Final Validation: quality gate trước khi respond
    │   └── teams/
    │       ├── docs/                # Team 1: Documentation & Ops
    │       │   ├── state.py         # DocsTeamState
    │       │   ├── supervisor.py    # LangGraph Supervisor subgraph
    │       │   ├── data_processor.py
    │       │   ├── summarizer.py
    │       │   ├── task_architect.py
    │       │   └── sync_manager.py
    │       ├── research/            # Team 2: Research & Advisory
    │       │   ├── state.py         # ResearchTeamState
    │       │   ├── supervisor.py
    │       │   ├── internal_brain.py
    │       │   ├── market_scout.py
    │       │   └── strategic_advisor.py
    │       ├── technical/           # Team 3: Technical Execution
    │       │   ├── state.py         # TechnicalTeamState
    │       │   ├── supervisor.py
    │       │   ├── visual_interpreter.py
    │       │   ├── code_architect.py
    │       │   └── quality_gatekeeper.py
    │       └── knowledge/           # Team 4: Knowledge & Learning
    │           ├── state.py         # KnowledgeTeamState
    │           ├── supervisor.py
    │           ├── knowledge_scout.py
    │           ├── experience_archivist.py
    │           └── yaml_architect.py
    │
    ├── persistence/                 # OUTBOUND: DATA ACCESS
    │   ├── checkpointer.py          # LangGraph PostgresSaver wrapper
    │   ├── models/
    │   │   ├── session.py           # SQLAlchemy: Thread/Session
    │   │   ├── quota.py             # SQLAlchemy: TokenUsage
    │   │   └── user.py              # SQLAlchemy: User
    │   └── repositories/
    │       ├── session_repo.py
    │       └── quota_repo.py
    │
    ├── observability/               # CROSS-CUTTING: OBSERVABILITY
    │   ├── setup.py                 # Arize Phoenix + OTEL init (auto-instrument LangGraph)
    │   ├── metrics.py               # Prometheus: token_count, latency_seconds, agent_calls
    │   └── eval.py                  # DeepEval: LLM-as-judge, golden-set harness
    │
    ├── scheduler/                   # CROSS-CUTTING: BACKGROUND JOBS
    │   ├── setup.py                 # APScheduler init (AsyncIOScheduler)
    │   └── jobs/
    │       ├── reminder.py          # Gửi reminder từ Sync Manager Agent
    │       └── quota_reset.py       # Reset daily/weekly/monthly counters
    │
    └── api/                         # INBOUND ADAPTER
        ├── deps.py                  # FastAPI deps: get_current_user, get_db, get_llm
        ├── middleware.py            # Logging, CORS, global error handler, rate limit
        ├── routers/
        │   ├── health.py            # GET /health
        │   ├── chat.py              # POST /chat → SSE
        │   │                        # POST /chat/{thread_id}/resume → SSE (HITL resume)
        │   ├── sessions.py          # CRUD /sessions, GET /sessions/{id}/checkpoints
        │   ├── quota.py             # GET /quota
        │   └── models.py            # GET /models
        └── dto/
            ├── chat.py              # ChatRequest, ChatEvent, HitlResumeRequest
            └── session.py           # SessionResponse, CheckpointResponse
```

---

## 5. Agent Team Hierarchy

```
👑 Manager Agent (Global Orchestrator)
   ├── Intent Classification
   ├── High-level Planning (Plan-and-Execute)
   ├── Dynamic Router → Teams
   ├── HITL interrupt() → User Approval → Resume
   └── Final Validation
        │
        ├── 📁 Team 1: Documentation & Ops    [DocsTeamState]
        │    ├── Data Processor    Audio/Video → Markdown + Timestamps
        │    ├── Summarizer        → Meeting Minutes, Executive Summary
        │    ├── Task Architect    → Action Items JSON (name, deadline, owner)
        │    └── Sync Manager      → Calendar/Notion/Jira + timed reminders
        │
        ├── 🧠 Team 2: Research & Advisory    [ResearchTeamState]
        │    ├── Internal Brain    RAG trên Personal Knowledge Base (pgvector)
        │    ├── Market Scout      Web search: Tavily/Perplexity (LATS exploration)
        │    └── Strategic Advisor SWOT + strategy synthesis từ Brain + Scout
        │
        ├── 🛠 Team 3: Technical Execution    [TechnicalTeamState]
        │    ├── Visual Interpreter Multimodal Vision: UI/Figma/screenshot → spec
        │    ├── Code Architect     Code generation (frontend/backend/script)
        │    └── Quality Gatekeeper Double-Check Verification → Pass/Fail report
        │
        └── 📚 Team 4: Knowledge & Learning   [KnowledgeTeamState]
             ├── Knowledge Scout    Doc/link → Technical Markdown Summary
             ├── Experience Archivist Phoenix traces → lessons learned
             └── YAML Architect     → YAML guidelines → Git commit vào prompts/
```

---

## 6. LangGraph Topology

```
[User Request via POST /chat]
          │
          ▼
  [FastAPI router] ──── SSE stream ──────────────────────────────────────────┐
          │                                                                   │
          ▼                                                                   │
  [Manager Graph]                                                             │
    ┌─────────────────────────────────────────────────────────┐               │
    │  guardrail_node ──► intent_classifier ──► planner_node  │               │
    │       │                                       │          │               │
    │       ▼                                       ▼          │               │
    │  [HITL interrupt()] ◄── User Approval ── dynamic_router  │               │
    │       │                                       │          │               │
    │       ▼                                       ▼          │               │
    │  [Resume] ──────────────────────► [Team Subgraph]        │               │
    │                                   ┌──────────────────┐   │               │
    │                                   │  team_supervisor  │   │               │
    │                                   │  agent_node_1     │   │               │
    │                                   │  agent_node_2     │   │               │
    │                                   │  agent_node_3     │   │               │
    │                                   └──────────────────┘   │               │
    │                                       │                   │               │
    │                                       ▼                   │               │
    │                                  validator_node           │               │
    └──────────────────────────────────────┘                   │               │
                                                               ▼               │
                                                    [SSE events stream] ───────┘
```

**HITL Flow:**
1. `POST /chat` → SSE stream bắt đầu
2. Manager gặp `interrupt()` → SSE gửi event `type: awaiting_approval`
3. Client nhận event → hiển thị cho user
4. `POST /chat/{thread_id}/resume` với approval/rejection → SSE stream mới từ checkpoint

---

## 7. State Schema Design

### Global AgentState (chia sẻ giữa Manager và Teams)

```python
class AgentState(TypedDict):
    thread_id: str
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Optional[str]           # classified intent
    plan: Optional[list[str]]       # high-level steps
    current_team: Optional[str]     # routing target
    team_output: Optional[str]      # result from team
    hitl_required: bool             # trigger interrupt
    final_response: Optional[str]
```

### Team States (isolated per team subgraph)

```python
class DocsTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    raw_input: str           # audio/video path hoặc text
    script: Optional[str]    # Data Processor output
    summary: Optional[str]   # Summarizer output
    tasks: Optional[list]    # Task Architect output (JSON)
    sync_status: Optional[str]

class ResearchTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    internal_knowledge: Optional[str]  # Internal Brain output
    web_results: Optional[str]         # Market Scout output
    recommendation: Optional[str]      # Strategic Advisor output

class TechnicalTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    design_spec: Optional[str]    # Visual Interpreter output
    code_output: Optional[str]    # Code Architect output
    review_report: Optional[str]  # Quality Gatekeeper output
    verdict: Optional[str]        # "PASS" | "FAIL"

class KnowledgeTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    source_docs: Optional[list[str]]
    technical_summary: Optional[str]  # Knowledge Scout output
    lessons: Optional[str]            # Experience Archivist output
    yaml_output: Optional[str]        # YAML Architect output
```

---

## 8. Pattern Reference

Mọi pattern trong note.md đều có vị trí cụ thể trong codebase:

| Pattern | File |
|---------|------|
| Hexagonal Architecture | Xem mục 3 |
| Human-in-the-Loop (HITL) | `agents/manager/graph.py` — `interrupt()` |
| Multi-modal I/O | `teams/docs/data_processor.py` + `teams/technical/visual_interpreter.py` |
| Reflection & Self-Correction | Retry loop trong mỗi agent subgraph |
| Hierarchical Teams | `agents/teams/*/supervisor.py` |
| Plan-and-Execute | `agents/manager/planner.py` |
| Corrective RAG | `memory/long_term.py` → retrieve → grade → rewrite → retrieve |
| Dynamic Router | `agents/manager/router.py` |
| State Persistence & Time Travel | `persistence/checkpointer.py` (PostgresSaver) |
| Hierarchical Supervisor | LangGraph `create_supervisor()` per team |
| Thinking Block | `config/agents.yaml` field `thinking: extended` per agent |
| Dynamic Few-Shot Prompting | `nodes/few_shot.py` |
| Meta-Prompting & Prompt Engineering | `nodes/meta_prompt.py` |
| Self-Discovery | `nodes/self_discovery.py` |
| LATS | `teams/research/market_scout.py` (tree search) |
| Double-Check Verification | `teams/technical/quality_gatekeeper.py` |
| Conflict Resolution Policy | `core/conflict.py` |
| Caching Layer | `core/caching.py` (Redis semantic cache) |
| Node Guardrail | `core/guardrails.py` — inject vào đầu mỗi graph |
| Parallel Tool Execution | LangGraph `Send` API trong agent nodes |
| Short-term Memory | `memory/short_term.py` |
| Long-term Memory | `memory/long_term.py` |
| Memory Pruning/Condensation | `memory/condensation.py` |
| Summarization Node | `nodes/summarization.py` |
| Entity Memory | `memory/entity_memory.py` (Redis Hash) |
| Tracing — Arize Phoenix | `observability/setup.py` |
| Evaluation — DeepEval | `observability/eval.py` |
| Prompt Management YAML+Git | `prompts/` + `tools/builtin/guidelines.py` |
| Token Usage Control | `llm/token_counter.py` + `persistence/repositories/quota_repo.py` |
| Tool Use / Function Calling | `tools/registry.py` |
| MCP | `tools/mcp/client.py` |
| Background Scheduler | `scheduler/jobs/` |

---

## 9. LLM Adapters & Model Selection

### Adapters (từ note.md)

| Adapter | File | Ghi chú |
|---------|------|---------|
| Gemini | `llm/gemini.py` | Primary production LLM |
| Beeknoee | `llm/beeknoee.py` | Custom provider |
| Mock | `llm/mock.py` | Testing, CI/CD |

Anthropic/OpenAI có thể thêm sau — implement `BaseLLMAdapter`.

### Mock Adapter Requirements

`llm/mock.py` phải đáp ứng:
- Dùng `GenericFakeChatModel` từ `langchain_core.language_models.fake_chat_models`
- Compile được với `LangGraph StateGraph.compile()`
- **Deterministic Mocking**: seed-based → cùng input → cùng output (reproducible tests)
- **Simulated Latency**: configurable `delay_ms` per call
- **Simulated Errors**: configurable `error_rate` → raise exception randomly
- **Full tool calling**: trả về `AIMessage` với `tool_calls` đúng format

### Model Selection per Agent (agents.yaml)

```yaml
agents:
  manager:
    model: gemini-2.0-flash-thinking-exp   # Thinking Block cho complex reasoning
    thinking: extended
  internal_brain:
    model: gemini-1.5-pro                  # Pro cho RAG synthesis
  market_scout:
    model: gemini-1.5-flash                # Flash cho web search (nhanh, rẻ)
  code_architect:
    model: gemini-2.0-flash-exp            # Code generation
  quality_gatekeeper:
    model: gemini-1.5-pro
  # ...các agent còn lại
```

---

## 10. Tool Registry

`tools/registry.py` implement 3-layer pattern:

```
Layer 1: Registration  → tool_registry.register(name, func, schema)
Layer 2: Representation → tool_registry.get_langchain_tools() → list[BaseTool]
Layer 3: Execution     → tool_registry.execute(name, args) → result
```

**Built-in tools:**

| Tool | File | Agent sử dụng |
|------|------|---------------|
| `read_guidelines` | `builtin/guidelines.py` | Mọi agent (đọc YAML prompt) |
| `web_search` | `builtin/web_search.py` | Market Scout |
| `read_file` | `builtin/file_reader.py` | Knowledge Scout, Data Processor |
| `sync_calendar` | `builtin/calendar.py` | Sync Manager |
| `run_code` | `builtin/code_runner.py` | Quality Gatekeeper |

**MCP tools:** `tools/mcp/client.py` bridge tất cả MCP server tools vào ToolRegistry.

---

## 11. Memory System

```
Short-term (Redis)
  └── Lưu N messages gần nhất của session hiện tại
  └── Tự động expire theo TTL

Long-term (pgvector) — Corrective RAG
  └── embed(doc) → store → retrieve(query) → grade(relevance)
      → nếu irrelevant: rewrite(query) → retrieve lại

Entity Memory (Redis Hash)
  └── facts về entities: {user_id, project_name, preferences, ...}
  └── Agents đọc/ghi để maintain context across sessions

Memory Condensation
  └── Khi messages > threshold → summarize_node() → thay thế bằng summary
  └── Chạy tự động trong background hoặc trigger bởi agent
```

---

## 12. Security Model

- **JWT**: Bearer token, validate trong `api/deps.py` → inject `current_user` vào route
- **Quota check**: Middleware kiểm tra token budget trước mỗi request
- **Rate limiting**: Sliding window trong `api/middleware.py`
- **Agent-level**: Mỗi agent chỉ có quyền dùng tools được khai báo trong `agents.yaml`
- **Guardrail**: `core/guardrails.py` inject vào đầu mỗi graph để filter unsafe content

---

## 13. Prompt Management

```
prompts/
├── manager/system.yaml         # System prompt cho Manager Agent
└── teams/
    ├── docs/
    │   ├── data_processor.yaml
    │   ├── summarizer.yaml
    │   ├── task_architect.yaml
    │   └── sync_manager.yaml
    ├── research/
    │   ├── internal_brain.yaml
    │   ├── market_scout.yaml
    │   └── strategic_advisor.yaml
    ├── technical/
    │   ├── visual_interpreter.yaml
    │   ├── code_architect.yaml
    │   └── quality_gatekeeper.yaml
    └── knowledge/
        ├── knowledge_scout.yaml
        ├── experience_archivist.yaml
        └── yaml_architect.yaml
```

Agents đọc prompt qua tool `read_guidelines(agent_name)` → nội dung YAML → inject vào system message.
YAML Architect Agent commit guidelines mới vào `prompts/` qua Git → versioned prompt evolution.

---

## 14. Observability

```
Arize Phoenix (port 6016)
  └── Auto-instrument LangGraph runs → trace mọi node/edge
  └── Experience Archivist đọc traces → phân tích success patterns

Prometheus (port 9290)
  └── pagent_token_count{model, user}
  └── pagent_agent_latency_seconds{agent, team}
  └── pagent_tool_calls_total{tool}

DeepEval
  └── LLM-as-judge cho agent outputs
  └── Golden-set evaluation harness
  └── Chạy trong CI/CD pipeline
```

---

## 15. Docker & Local Dev

### Local Development (venv + VSCode debug)

```bash
# Bước 1: Start infrastructure services
docker-compose up -d

# Bước 2: Setup Python environment
uv venv && uv sync

# Bước 3: Debug trong VSCode
# F5 → "FastAPI: Debug" (cấu hình trong .vscode/launch.json)
```

`.vscode/launch.json` config:
- Program: `uvicorn`
- Module args: `src.main:app --reload --port 8200`
- EnvFile: `${workspaceFolder}/.env`
- Python: `${workspaceFolder}/.venv/Scripts/python` (Windows)

### Infrastructure Services (docker-compose.yml)

App **KHÔNG** nằm trong docker-compose.yml khi dev — chạy bằng venv để VSCode debug hoạt động.

| Service | Image | External Port |
|---------|-------|--------------|
| pagent-mysql | mysql:8.0 | **3316** |
| pagent-postgres | pgvector/pgvector:pg16 | **5442** |
| pagent-redis | redis:7-alpine | **6389** |
| pagent-phoenix | arizephoenix/phoenix | **6016** |
| pagent-prometheus | prom/prometheus | **9290** |

Ports configurable trong `.env`: `MYSQL_PORT`, `POSTGRES_PORT`, `REDIS_PORT`, v.v.

### Production (docker-compose.prod.yml)

Thêm `pagent-app` service (Dockerfile build) + resource limits + restart policies.

---

## 16. API Endpoints

### Public
- `GET /health` — health check

### Authenticated (JWT Bearer)
- `POST /chat` → SSE stream — gửi request, nhận agent response theo dòng
- `POST /chat/{thread_id}/resume` → SSE stream — HITL: resume sau khi user approve
- `GET /sessions` — danh sách sessions
- `GET /sessions/{id}` — session detail
- `GET /sessions/{id}/checkpoints` — time-travel: list checkpoints
- `POST /sessions/{id}/rollback/{checkpoint_id}` — rollback về checkpoint
- `DELETE /sessions/{id}` — xóa session
- `GET /quota` — token usage hiện tại (per model, per period)
- `GET /models` — danh sách LLM models available

---

## 17. Naming Conventions

| Loại | Pattern | Ví dụ |
|------|---------|-------|
| Agent file | `{name}.py` (lowercase snake) | `data_processor.py`, `market_scout.py` |
| State TypedDict | `{Team}TeamState` | `DocsTeamState`, `ResearchTeamState` |
| Graph function | `build_{name}_graph()` | `build_docs_graph()` |
| Node function | `{action}_node()` | `classify_intent_node()`, `route_team_node()` |
| Tool function | `{verb}_{noun}()` | `search_web()`, `read_guidelines()` |
| LLM Adapter class | `{Provider}Adapter` | `GeminiAdapter`, `BeeknoeeAdapter` |
| Repository class | `{Entity}Repository` | `SessionRepository`, `QuotaRepository` |
| FastAPI router | `{resource}_router` | `chat_router`, `session_router` |
| DTO class | `{Action}Request`, `{Entity}Response` | `ChatRequest`, `SessionResponse` |
| Pydantic Settings field | SCREAMING_SNAKE in env, snake in class | `MYSQL_PORT` → `mysql_port` |

---

## 18. Strict Rules

1. **Agents chỉ nằm trong `src/agents/`** — không tạo agent logic trong `api/` hay `tools/`
2. **LangGraph nodes phải là pure functions** — không có side effects ngoài state update
3. **Tất cả LLM calls qua `LLMRegistry`** — không import thẳng provider SDK trong agents
4. **Tools phải đăng ký qua `ToolRegistry`** — không gọi tool function trực tiếp trong node
5. **State mutation chỉ qua TypedDict return** — không mutate state object trực tiếp
6. **Async everywhere** — mọi node, tool, repo method phải là `async def`
7. **Tất cả API response dùng `ApiResponse[T]`** — không trả về raw dict
8. **Prompt không hardcode trong Python** — luôn đọc từ `prompts/*.yaml` qua `read_guidelines`
9. **Mock adapter cho testing** — không dùng real LLM trong unit tests
10. **Port allocation**: app=8200, mysql=3316, postgres=5442, redis=6389, phoenix=6016, prometheus=9290
