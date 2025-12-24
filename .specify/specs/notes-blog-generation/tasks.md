# Tasks: Notes & Blog Generation

**Input**: Design documents from `.specify/specs/notes-blog-generation/`  
**Prerequisites**: plan.md ✅, spec (notes-blog-generation.md) ✅

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1=Notes Capture, US2=Blog Generation, US3=Templates, US4=Notes Management

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create new module files and configure dependencies

- [X] T001 [P] Create `src/nyrag/notes.py` with module docstring and imports
- [X] T002 [P] Create `src/nyrag/blog.py` with module docstring and imports
- [X] T003 [P] Create `src/nyrag/jobs.py` with module docstring and imports
- [X] T004 [P] Create `src/nyrag/blog_templates/` directory
- [X] T005 [P] Create `src/nyrag/static/notes.css` with base styles
- [X] T006 [P] Create `src/nyrag/static/notes.js` with base structure
- [X] T007 [P] Create `src/nyrag/templates/notes.html` with base template

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Config models and Vespa schema that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 Add `NotesParams` Pydantic model to `src/nyrag/config.py`
  - Fields: `max_image_size_mb: float = 5.0`, `storage_path: str = "notes"`
- [X] T009 Add `BlogParams` Pydantic model to `src/nyrag/config.py`
  - Fields: `templates_path: str = "blog_templates"`, `output_path: str = "blogs"`, `timeout_seconds: int = 120`
- [X] T010 Add `notes_params` and `blog_params` optional fields to `Config` class in `src/nyrag/config.py`
- [X] T011 Create `Note` Pydantic model in `src/nyrag/notes.py`
  - Fields: `id`, `title`, `content`, `images`, `created_at`, `updated_at`, `tags`
- [X] T012 Create `BlogPost` Pydantic model in `src/nyrag/blog.py`
  - Fields: `id`, `topic`, `template`, `content`, `source_notes`, `status`, `created_at`
- [X] T013 Create `BlogTemplate` Pydantic model in `src/nyrag/blog.py`
  - Fields: `name`, `description`, `structure`, `system_prompt`, `example_output`
- [X] T014 Add `generate_notes_schema()` function in `src/nyrag/schema.py`
  - Create `nyrag<project>notes` schema with: id, title, content, embedding, created_at, tags
- [X] T015 Create `Job` and `JobStatus` models in `src/nyrag/jobs.py`
  - Fields: `id`, `type`, `status` (queued/running/complete/failed), `result`, `error`, `created_at`
- [X] T016 Implement `JobQueue` class in `src/nyrag/jobs.py`
  - Methods: `submit()`, `get_status()`, `get_result()`, `cancel()`, `cleanup_expired()`
  - Use asyncio for background execution

**Checkpoint**: Foundation ready - Config, models, schema, job queue all in place

---

## Phase 3: User Story 1 - Notes Capture via Web UI (Priority: P1) 🎯 MVP

**Goal**: Users can add markdown notes with images via web UI, stored locally and indexed in Vespa

**Independent Test**: Open `/notes`, create note with image, verify appears in chat search

### Implementation for User Story 1

- [X] T017 [US1] Implement `save_note_local()` in `src/nyrag/notes.py`
  - Save to `output/<project>/notes/notes.jsonl`
  - Append mode, create directory if not exists
- [X] T018 [US1] Implement `save_image()` in `src/nyrag/notes.py`
  - Save to `output/<project>/assets/<note_id>/`
  - Validate size <= max_image_size_mb, return relative path
- [X] T019 [US1] Implement `index_note_vespa()` in `src/nyrag/notes.py`
  - Generate embedding using SentenceTransformer
  - Feed to Vespa notes schema
  - Handle connection errors gracefully (queue for retry)
- [X] T020 [US1] Implement `save_note()` orchestrator in `src/nyrag/notes.py`
  - Calls `save_note_local()` + `index_note_vespa()`
  - Returns saved Note with id
- [X] T021 [US1] Implement `get_note()` in `src/nyrag/notes.py`
  - Retrieve single note by id from local JSONL
- [X] T022 [US1] Implement `search_notes()` in `src/nyrag/notes.py`
  - Query Vespa notes schema with embedding similarity
- [X] T023 [US1] Add `POST /api/notes` endpoint in `src/nyrag/api.py`
  - Accept: `{title, content, tags[]}`
  - Return: saved Note object
- [X] T024 [US1] Add `POST /api/notes/upload-image` endpoint in `src/nyrag/api.py`
  - Accept: multipart file upload
  - Return: `{url: "/assets/<note_id>/<filename>"}`
- [X] T025 [US1] Add `GET /api/notes/{note_id}` endpoint in `src/nyrag/api.py`
- [X] T026 [US1] Add `GET /notes` HTML route in `src/nyrag/api.py`
  - Render `notes.html` template
- [X] T027 [US1] Implement markdown editor in `src/nyrag/static/notes.js`
  - Use EasyMDE library via CDN (actively maintained SimpleMDE fork)
  - Live preview, image paste support
- [X] T028 [US1] Implement image upload handler in `src/nyrag/static/notes.js`
  - Drag-drop and paste support
  - Show upload progress, insert markdown image link
- [X] T029 [US1] Implement save note handler in `src/nyrag/static/notes.js`
  - Collect title (auto-extract from first heading or user input)
  - POST to `/api/notes`
  - Show success/error feedback
- [X] T030 [US1] Build notes page layout in `src/nyrag/templates/notes.html`
  - Header, editor area, save button, status messages
- [X] T031 [US1] Style notes editor in `src/nyrag/static/notes.css`
  - Match existing chat UI theme
- [X] T032 [US1] Modify search in `src/nyrag/api.py` to include notes schema
  - Query both main schema and notes schema
  - Merge and rank results

**Checkpoint**: User Story 1 complete - Notes can be created, saved, and found in search ✅

---

## Phase 4: User Story 2 - Blog Generation via Background Agent (Priority: P2)

**Goal**: Users request blog generation in chat, background agent creates Substack-compatible markdown

**Independent Test**: Have notes saved, request "Generate blog about X" in chat, receive markdown file

### Implementation for User Story 2

- [X] T033 [US2] Implement `retrieve_rag_context()` in `src/nyrag/blog.py`
  - Query all sources: notes schema + main schema (docs/web)
  - Return top-k chunks with source attribution
- [X] T034 [US2] Implement `build_blog_prompt()` in `src/nyrag/blog.py`
  - System prompt for Substack-style blog
  - Include RAG context, topic, any user instructions
- [X] T035 [US2] Implement `generate_blog_content()` in `src/nyrag/blog.py`
  - Call OpenRouter LLM with blog prompt
  - Stream response, handle errors
  - Return markdown content
- [X] T036 [US2] Implement `save_blog()` in `src/nyrag/blog.py`
  - Save to `output/<project>/blogs/<blog_id>.md`
  - Include frontmatter: title, date, sources
- [X] T037 [US2] Implement `generate_blog_task()` async function in `src/nyrag/blog.py`
  - Full orchestration: retrieve → prompt → generate → save
  - Update job status throughout
  - Handle timeouts and errors
- [X] T038 [US2] Add `POST /api/blog/generate` endpoint in `src/nyrag/api.py`
  - Accept: `{topic, template?, instructions?}`
  - Submit to JobQueue, return job_id
- [X] T039 [US2] Add `GET /api/jobs/{job_id}` endpoint in `src/nyrag/api.py`
  - Return job status, result if complete
- [X] T040 [US2] Add `GET /api/blog/{blog_id}` endpoint in `src/nyrag/api.py`
  - Return generated blog content
- [X] T041 [US2] Add `GET /api/blog/{blog_id}/download` endpoint in `src/nyrag/api.py`
  - Return file download response
- [X] T042 [US2] Detect blog generation intent in chat in `src/nyrag/api.py`
  - Pattern: "generate blog", "write blog", "create post"
  - Trigger `/api/blog/generate` flow
  - Return job status in chat response
- [X] T043 [US2] Add job status polling in `src/nyrag/static/chat.js`
  - Show progress indicator for blog generation
  - Display download link when complete

**Checkpoint**: User Story 2 complete - Blogs can be generated from chat with background processing

---

## Phase 5: User Story 3 - Blog Templates (Priority: P3)

**Goal**: Users can select blog template (tutorial, opinion, roundup) for customized output

**Independent Test**: Request blog with "tutorial" template, verify output has intro/steps/conclusion structure

### Implementation for User Story 3

- [X] T044 [P] [US3] Create `src/nyrag/blog_templates/tutorial.yaml`
  - Structure: introduction, prerequisites, steps[], conclusion
  - System prompt for instructional tone
- [X] T045 [P] [US3] Create `src/nyrag/blog_templates/opinion.yaml`
  - Structure: hook, thesis, arguments[], counterpoint, conclusion
  - System prompt for persuasive tone
- [X] T046 [P] [US3] Create `src/nyrag/blog_templates/roundup.yaml`
  - Structure: introduction, items[], comparison, recommendation
  - System prompt for curated list style
- [X] T047 [P] [US3] Create `src/nyrag/blog_templates/technical.yaml`
  - Structure: problem, background, solution, implementation, results
  - System prompt for technical depth
- [X] T048 [US3] Implement `load_template()` in `src/nyrag/blog.py`
  - Load YAML from `blog_templates/` directory
  - Parse into BlogTemplate model
  - Cache loaded templates
- [X] T049 [US3] Implement `list_templates()` in `src/nyrag/blog.py`
  - Scan `blog_templates/` directory
  - Return list of available template names
- [X] T050 [US3] Modify `build_blog_prompt()` to use template in `src/nyrag/blog.py`
  - Include template structure in prompt
  - Use template system_prompt
- [X] T051 [US3] Add `GET /api/blog/templates` endpoint in `src/nyrag/api.py`
  - Return list of available templates with descriptions
- [X] T052 [US3] Add template selector in blog generation UI in `src/nyrag/static/chat.js`
  - Dropdown or buttons for template selection
  - Include in generation request

**Checkpoint**: User Story 3 complete - Blogs can be generated with different templates ✅

---

## Phase 6: User Story 4 - Notes Management (Priority: P4)

**Goal**: Users can browse, edit, and delete their saved notes

**Independent Test**: View notes list, edit a note, delete another, verify changes persist

### Implementation for User Story 4

- [X] T053 [US4] Implement `list_notes()` in `src/nyrag/notes.py`
  - Read all notes from JSONL
  - Return list sorted by created_at desc
- [X] T054 [US4] Implement `update_note()` in `src/nyrag/notes.py`
  - Update in local JSONL (rewrite file)
  - Re-index in Vespa with new embedding
- [X] T055 [US4] Implement `delete_note()` in `src/nyrag/notes.py`
  - Remove from local JSONL
  - Delete from Vespa
  - Delete associated images from assets/
- [X] T056 [US4] Add `GET /api/notes` endpoint in `src/nyrag/api.py`
  - Return list of all notes (summary: id, title, created_at)
- [X] T057 [US4] Add `PUT /api/notes/{note_id}` endpoint in `src/nyrag/api.py`
  - Accept updated note content
- [X] T058 [US4] Add `DELETE /api/notes/{note_id}` endpoint in `src/nyrag/api.py`
- [X] T059 [US4] Add notes list view in `src/nyrag/static/notes.js`
  - Sidebar or separate view showing all notes
  - Click to load note in editor
- [X] T060 [US4] Add edit mode in `src/nyrag/static/notes.js`
  - Load existing note content
  - PUT on save
- [X] T061 [US4] Add delete functionality in `src/nyrag/static/notes.js`
  - Confirmation dialog
  - DELETE request
  - Remove from list

**Checkpoint**: User Story 4 complete - Full notes CRUD in UI ✅

---

## Phase 7: Polish & Integration

**Purpose**: Final touches and cross-cutting improvements

- [X] T062 [P] Add navigation link to `/notes` in `src/nyrag/templates/chat.html`
- [X] T063 [P] Update README.md with notes and blog generation documentation
- [X] T064 [P] Add example notes config to `configs/example.yml`
- [X] T065 Run `make quality` and fix any style issues
- [ ] T066 Validate full flow: create note → search → generate blog → download

**Checkpoint**: Phase 7 complete - Polish & Integration ✅

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ─────► Phase 2 (Foundational) ─────┬──► Phase 3 (US1: Notes)
                                                    │
                                                    ├──► Phase 4 (US2: Blog) ──► Phase 5 (US3: Templates)
                                                    │
                                                    └──► Phase 6 (US4: Management)
                                                    
All User Stories ──► Phase 7 (Polish)
```

### User Story Dependencies

| Story | Depends On | Can Parallel With |
|-------|------------|-------------------|
| US1 (Notes) | Phase 2 only | US2, US4 (after Phase 2) |
| US2 (Blog) | Phase 2 + US1 (needs notes to retrieve) | US4 |
| US3 (Templates) | US2 (extends blog generation) | - |
| US4 (Management) | US1 (extends notes) | US2 |

### Parallel Opportunities

Within Phase 1:
- T001-T007 all parallel (different files)

Within Phase 2:
- T008-T010 parallel (config changes)
- T011-T013 parallel (different model files)
- T014-T016 parallel (different modules)

Within US1:
- T017-T022 sequential (notes.py implementation)
- T023-T026 parallel (different API routes)
- T027-T031 parallel (different UI files)

Within US3:
- T044-T047 parallel (different template files)

---

## Implementation Strategy

### MVP (US1 Only) - 22h
1. Complete Phase 1 + Phase 2
2. Complete Phase 3 (US1: Notes Capture)
3. **VALIDATE**: Create note with image, find in search

### Core Feature (US1 + US2) - 35h
4. Complete Phase 4 (US2: Blog Generation)
5. **VALIDATE**: Generate blog from notes via chat

### Enhanced (+ US3) - 43h
6. Complete Phase 5 (US3: Templates)
7. **VALIDATE**: Generate blogs with different templates

### Complete (+ US4) - 49h
8. Complete Phase 6 (US4: Notes Management)
9. Complete Phase 7 (Polish)
10. **VALIDATE**: Full E2E flow

---

## Task Summary

| Phase | Tasks | Effort |
|-------|-------|--------|
| Phase 1: Setup | T001-T007 (7 tasks) | 2h |
| Phase 2: Foundational | T008-T016 (9 tasks) | 5h |
| Phase 3: US1 Notes | T017-T032 (16 tasks) | 15h |
| Phase 4: US2 Blog | T033-T043 (11 tasks) | 13h |
| Phase 5: US3 Templates | T044-T052 (9 tasks) | 8h |
| Phase 6: US4 Management | T053-T061 (9 tasks) | 6h |
| Phase 7: Polish | T062-T066 (5 tasks) | 3h |
| **Total** | **66 tasks** | **52h** |
