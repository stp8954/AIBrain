# Implementation Plan: UI Document Manager

**Branch**: `002-ui-document-manager` | **Date**: 2025-12-25 | **Spec**: [.specify/specs/ui-document-manager.md](../../.specify/specs/ui-document-manager.md)
**Input**: Feature specification from `.specify/specs/ui-document-manager.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a web UI-based document management system with left sidebar navigation, data sources table view, background agents/jobs monitoring, and the ability to add new resources (URLs, files) with background ingestion and progress tracking. Uses SQLite for metadata storage, integrates with existing Vespa vector search, and extends the FastAPI backend with new CRUD endpoints.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: FastAPI, Uvicorn, SQLite (aiosqlite), Jinja2, SentenceTransformers, existing crawly module  
**Storage**: SQLite for metadata (data_sources, jobs, conversations, messages tables), Vespa for vector search  
**Testing**: pytest with `make test` or `pytest -sv ./src/`  
**Target Platform**: Linux server (Docker), web browser clients  
**Project Type**: single (extending existing web application)  
**Performance Goals**: Sub-200ms API responses, real-time progress updates via SSE  
**Constraints**: Max 2 concurrent ingestion jobs, 50MB max file upload size  
**Scale/Scope**: ~4 new pages/views, ~15 new API endpoints, ~10 new source files

**Terminology Note**: UI uses "Agents" for user-facing labels (page name, navigation), while API/database uses "Jobs" for technical consistency. The Agents page displays Job entities.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-Driven Design | ✅ PASS | Feature uses existing YAML config for Vespa/embedding settings; UI behavior is data-driven |
| II. CLI-First Interface | ✅ PASS | Extends existing system; CLI remains primary for batch operations, UI is complementary |
| III. Dual Deployment Modes | ✅ PASS | SQLite is portable; Vespa integration works with both local Docker and Cloud modes |
| IV. Modular Pipeline Architecture | ✅ PASS | New modules (database.py, sources.py) follow existing pattern; reuses crawly/ and process.py |
| V. Observability & Debugging | ✅ PASS | Uses loguru logging, job progress tracking, JSONL-compatible patterns |
| Code Quality Standards | ✅ PASS | Will follow black/isort/flake8, 119 char lines, type hints on public functions |
| Type Safety | ✅ PASS | Pydantic models for API contracts, Literal types for status enums |

**Gate Status**: ✅ PASSED - No violations. Feature aligns with all constitution principles.

## Project Structure

### Documentation (this feature)

```text
specs/002-ui-document-manager/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (OpenAPI specs)
│   └── openapi.yaml     # Full API specification
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/nyrag/
├── database.py          # NEW: SQLite connection manager and models
├── sources.py           # NEW: Data source CRUD and ingestion logic
├── api.py               # MODIFIED: Add new endpoints for sources, jobs, conversations
├── static/
│   ├── app.css          # NEW: Shared styles for new layout
│   ├── app.js           # NEW: Shared JS (sidebar, navigation)
│   ├── sources.js       # NEW: Data Sources page logic
│   ├── agents.js        # NEW: Agents page logic
│   ├── history.js       # NEW: Chat history logic
│   ├── chat.css         # EXISTING
│   └── chat.js          # MODIFIED: Add history integration
└── templates/
    ├── base.html        # NEW: Base template with sidebar navigation
    ├── chat.html        # MODIFIED: Extend base.html, add history sidebar
    ├── sources.html     # NEW: Data Sources page
    └── agents.html      # NEW: Agents page

tests/
├── test_database.py     # NEW: Database model tests
├── test_sources_api.py  # NEW: Sources API endpoint tests
└── test_jobs_api.py     # NEW: Jobs API endpoint tests
```

**Structure Decision**: Single project extending existing `src/nyrag/` structure. New modules (`database.py`, `sources.py`) follow the existing flat module pattern. Templates and static files extend existing directories. No new top-level packages required.

## Complexity Tracking

> Constitution Check passed with no violations - no justifications needed.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
