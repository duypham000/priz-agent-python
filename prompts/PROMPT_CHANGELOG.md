# Prompt Changelog

Format: `version | file | date | description`

## v1.0 — 2026-04-26

| Version | File | Date | Description |
|---------|------|------|-------------|
| v1.0 | `manager/system.yaml` | 2026-04-26 | Initial: intent_classifier, planner, validator sub-prompts |
| v1.0 | `teams/docs/data_processor.yaml` | 2026-04-26 | Initial: audio/video → transcript with timestamps |
| v1.0 | `teams/docs/summarizer.yaml` | 2026-04-26 | Initial: script → meeting minutes + executive summary |
| v1.0 | `teams/docs/task_architect.yaml` | 2026-04-26 | Initial: discussion → JSON action items array |
| v1.0 | `teams/docs/sync_manager.yaml` | 2026-04-26 | Initial: push tasks to Calendar/Notion/Jira |
| v1.0 | `teams/research/internal_brain.yaml` | 2026-04-26 | Initial: RAG synthesis with knowledge gaps |
| v1.0 | `teams/research/market_scout.yaml` | 2026-04-26 | Initial: web search + LATS market intelligence |
| v1.0 | `teams/research/strategic_advisor.yaml` | 2026-04-26 | Initial: SWOT synthesis from internal + market data |
| v1.0 | `teams/technical/code_architect.yaml` | 2026-04-26 | Initial: spec → production-ready code |
| v1.0 | `teams/technical/quality_gatekeeper.yaml` | 2026-04-26 | Initial: strict code review + execution verdict |
| v1.0 | `teams/technical/visual_interpreter.yaml` | 2026-04-26 | Initial: UI/Figma → component spec |
| v1.0 | `teams/knowledge/knowledge_scout.yaml` | 2026-04-26 | Initial: doc/link → technical markdown summary |
| v1.0 | `teams/knowledge/experience_archivist.yaml` | 2026-04-26 | Initial: Phoenix traces → lessons learned |
| v1.0 | `teams/knowledge/yaml_architect.yaml` | 2026-04-26 | Initial: synthesize findings → YAML guidelines + git commit |

## How to version prompts

When modifying a prompt file:
1. Bump `version` field in the YAML (e.g., `"1.0"` → `"1.1"`)
2. Add a row to this changelog with the new version, file, date, and description of change
3. Commit both files together so version and changelog stay in sync
