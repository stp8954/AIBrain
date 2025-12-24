# Implementation Plan: Notes & Blog Generation

**Branch**: `feat/notes-blog-generation` | **Date**: 2025-12-23 | **Spec**: [notes-blog-generation.md](../notes-blog-generation.md)

## Summary

Add a web UI for capturing markdown notes with images, store them locally and index in Vespa for RAG retrieval, and provide a background agent that generates Substack-compatible blog posts using all RAG sources (notes + docs + crawled web content). Blog templates are YAML-configurable.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: FastAPI, Pydantic, SentenceTransformers, pyvespa, OpenAI SDK, Jinja2  
**Storage**: Local JSONL files + Vespa (separate notes schema) + local image assets  
**Testing**: pytest  
**Target Platform**: Linux server (local Docker or Vespa Cloud)  
**Project Type**: Single Python package with web UI  
**Performance Goals**: Note save <1s, Vespa index <5s, blog generation <2min  
**Constraints**: Max image size 5MB, async blog generation (non-blocking)  
**Scale/Scope**: 100+ notes, single user

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-Driven | ✅ Pass | Notes params in YAML config, templates in `blog_templates/` |
| II. CLI-First | ✅ Pass | UI extends existing FastAPI app, no new CLI commands required |
| III. Dual Deployment | ✅ Pass | Notes use same Vespa connection (local/cloud) as existing features |
| IV. Modular Pipeline | ✅ Pass | New modules: `notes.py`, `blog.py`, `templates/` - composable |
| V. Observability | ✅ Pass | Logging via loguru, job status tracking for blog generation |

## Project Structure

### Documentation (this feature)

```text
.specify/specs/notes-blog-generation/
├── plan.md              # This file
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code

```text
src/nyrag/
├── notes.py             # NEW: Note CRUD operations, Vespa indexing
├── blog.py              # NEW: Blog generation agent, template engine
├── jobs.py              # NEW: Background job queue for async tasks
├── config.py            # MODIFY: Add NotesParams, BlogParams
├── schema.py            # MODIFY: Add notes schema generation
├── api.py               # MODIFY: Add /notes routes, /blog routes
├── static/
│   ├── notes.css        # NEW: Notes UI styles
│   └── notes.js         # NEW: Notes UI logic (markdown editor)
├── templates/
│   └── notes.html       # NEW: Notes page template
└── blog_templates/      # NEW: YAML blog templates
    ├── tutorial.yaml
    ├── opinion.yaml
    └── roundup.yaml

output/<project>/
├── notes/
│   └── notes.jsonl      # Note storage
├── assets/
│   └── <note_id>/       # Image attachments per note
└── blogs/
    └── <blog_id>.md     # Generated blog posts
```

**Structure Decision**: Extends existing single-package structure. New modules (`notes.py`, `blog.py`, `jobs.py`) follow existing patterns. UI assets in `static/` and `templates/` consistent with existing chat UI.

## Implementation Phases

> **Note**: Detailed task breakdown with explicit dependencies is in [tasks.md](./tasks.md). 
> Tasks.md includes a "Setup" phase (file creation) and "Foundational" phase (shared models) 
> before the user story phases below.

### Phase 1: Notes Backend & Storage (P1 Foundation)

**Goal**: Create note CRUD operations with local + Vespa storage

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| 1.1 | Add `NotesParams` to config | `config.py` | 1h |
| 1.2 | Create Note Pydantic models | `notes.py` | 1h |
| 1.3 | Implement note save (local JSONL) | `notes.py` | 2h |
| 1.4 | Generate notes Vespa schema | `schema.py` | 2h |
| 1.5 | Implement note indexing in Vespa | `notes.py` | 2h |
| 1.6 | Implement note retrieval/search | `notes.py` | 2h |

**Deliverable**: `notes.py` module with full CRUD + Vespa integration  
**Test**: Unit tests for note creation, storage, retrieval

---

### Phase 2: Notes Web UI (P1 Completion)

**Goal**: Web interface for adding/viewing notes with markdown + images

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| 2.1 | Create notes HTML template | `templates/notes.html` | 2h |
| 2.2 | Add markdown editor JS (EasyMDE via CDN) | `static/notes.js` | 3h |
| 2.3 | Implement image upload endpoint | `api.py` | 2h |
| 2.4 | Add notes CRUD API routes | `api.py` | 2h |
| 2.5 | Style notes UI | `static/notes.css` | 1h |
| 2.6 | Integrate notes in chat search results | `api.py` | 2h |

**Deliverable**: `/notes` route with full note capture functionality  
**Test**: E2E test - create note with image, verify in Vespa search

---

### Phase 3: Background Job System (P2 Foundation)

**Goal**: Async task queue for long-running blog generation

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| 3.1 | Create job queue with asyncio | `jobs.py` | 3h |
| 3.2 | Add job status tracking (in-memory or file) | `jobs.py` | 2h |
| 3.3 | Add job status API endpoint | `api.py` | 1h |
| 3.4 | Add job cleanup/expiry | `jobs.py` | 1h |

**Deliverable**: `jobs.py` module with submit/status/cancel operations  
**Test**: Unit test for job lifecycle

---

### Phase 4: Blog Generation Agent (P2 Completion)

**Goal**: Generate Substack-compatible blogs using RAG

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| 4.1 | Create BlogPost Pydantic models | `blog.py` | 1h |
| 4.2 | Implement multi-source RAG retrieval | `blog.py` | 3h |
| 4.3 | Create blog generation prompt | `blog.py` | 2h |
| 4.4 | Implement blog generation with LLM | `blog.py` | 3h |
| 4.5 | Save generated blog to output | `blog.py` | 1h |
| 4.6 | Add blog generation API endpoint | `api.py` | 2h |
| 4.7 | Add blog status/download endpoints | `api.py` | 1h |

**Deliverable**: `/blog/generate` endpoint that spawns background job  
**Test**: Integration test - generate blog from notes, verify output

---

### Phase 5: Blog Templates (P3)

**Goal**: YAML-configurable blog templates

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| 5.1 | Define template YAML schema | `blog.py` | 1h |
| 5.2 | Create default templates | `blog_templates/*.yaml` | 2h |
| 5.3 | Implement template loader | `blog.py` | 2h |
| 5.4 | Integrate templates in generation | `blog.py` | 2h |
| 5.5 | Add template list API endpoint | `api.py` | 1h |

**Deliverable**: Selectable templates: tutorial, opinion, roundup, technical-deep-dive  
**Test**: Generate blogs with different templates, verify structure

---

### Phase 6: Notes Management UI (P4)

**Goal**: Browse, edit, delete notes

| Task | Description | Files | Est. |
|------|-------------|-------|------|
| 6.1 | Add notes list view | `templates/notes.html`, `static/notes.js` | 2h |
| 6.2 | Implement note edit flow | `static/notes.js`, `api.py` | 2h |
| 6.3 | Implement note delete | `notes.py`, `api.py` | 1h |
| 6.4 | Add confirmation dialogs | `static/notes.js` | 1h |

**Deliverable**: Full notes CRUD in UI  
**Test**: E2E test - create, edit, delete note

## Dependencies

```
Phase 1 ─────► Phase 2 (UI needs backend)
    │
    └───────► Phase 3 ─────► Phase 4 (Blog needs jobs queue)
                               │
                               └───► Phase 5 (Templates enhance generation)

Phase 2 ─────► Phase 6 (Management extends capture UI)
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Vespa schema conflicts | Use separate `nyrag<project>notes` schema namespace |
| Large image uploads blocking | Async upload with progress, 5MB limit |
| Blog generation timeout | Configurable timeout, partial save on failure |
| EasyMDE dependency | Fallback to plain textarea with preview |

## Total Estimates

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1: Notes Backend | 10h | P1 |
| Phase 2: Notes UI | 12h | P1 |
| Phase 3: Job System | 7h | P2 |
| Phase 4: Blog Agent | 13h | P2 |
| Phase 5: Templates | 8h | P3 |
| Phase 6: Notes Management | 6h | P4 |
| **Total** | **56h** | |

**MVP (P1+P2)**: 22h - Notes capture with UI and Vespa indexing  
**Core Feature (P1-P4)**: 42h - Full notes + blog generation  
**Complete (P1-P6)**: 56h - All features including templates and management
