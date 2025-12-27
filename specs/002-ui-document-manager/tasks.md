# Tasks: UI Document Manager

**Input**: Design documents from `/specs/002-ui-document-manager/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/openapi.yaml ✓

**Tests**: Not explicitly requested in spec - test tasks omitted.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependencies, and base structure

- [X] T001 Add aiosqlite and aiofiles dependencies to pyproject.toml
- [X] T002 [P] Create base template with sidebar navigation in src/nyrag/templates/base.html
- [X] T003 [P] Create shared CSS styles for app layout in src/nyrag/static/app.css
- [X] T004 [P] Create shared JS for sidebar navigation in src/nyrag/static/app.js

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core database infrastructure that MUST be complete before ANY user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Create SQLite database module with async connection manager in src/nyrag/database.py
- [X] T006 Implement schema initialization (create tables for data_sources, jobs, conversations, messages) in src/nyrag/database.py
- [X] T007 [P] Create Pydantic models for DataSource, Job, Conversation, Message in src/nyrag/schema.py
- [X] T008 [P] Create uploads directory management utility in src/nyrag/utils.py
- [X] T009 Initialize database on API startup in src/nyrag/api.py (lifespan event)
- [X] T010 Add system status endpoint GET /api/status in src/nyrag/api.py

**Checkpoint**: Database and base infrastructure ready - user story implementation can begin

---

## Phase 3: User Story 1 - Data Sources Management (Priority: P1) 🎯 MVP

**Goal**: Users can add URLs and files as data sources, view them in a table, and delete them

**Independent Test**: Navigate to /sources, add a URL, see it appear in table with status, delete it

### Implementation for User Story 1

- [X] T011 Create data sources CRUD module in src/nyrag/sources.py
- [X] T012 [US1] Implement create_data_source() function in src/nyrag/sources.py
- [X] T013 [US1] Implement list_data_sources() with filtering in src/nyrag/sources.py
- [X] T014 [US1] Implement get_data_source() function in src/nyrag/sources.py
- [X] T015 [US1] Implement delete_data_source() with Vespa chunk cleanup in src/nyrag/sources.py
- [X] T016 [P] [US1] Add GET /api/sources endpoint (list with filtering) in src/nyrag/api.py
- [X] T017 [P] [US1] Add GET /api/sources/{source_id} endpoint in src/nyrag/api.py
- [X] T018 [US1] Add POST /api/sources/url endpoint for URL crawling in src/nyrag/api.py
- [X] T019 [US1] Add POST /api/sources/files endpoint for file uploads in src/nyrag/api.py
- [X] T020 [US1] Add DELETE /api/sources/{source_id} endpoint in src/nyrag/api.py
- [X] T021 [US1] Implement file upload with size validation (50MB max) in src/nyrag/sources.py
- [X] T022 [P] [US1] Create Data Sources page template in src/nyrag/templates/sources.html
- [X] T023 [P] [US1] Create Data Sources page JavaScript in src/nyrag/static/sources.js
- [X] T024 [US1] Implement Add Source modal (URL tab and File Upload tab) in src/nyrag/static/sources.js
- [X] T025 [US1] Implement data sources table with filtering UI in src/nyrag/static/sources.js
- [X] T026 [US1] Add /sources page route in src/nyrag/api.py

**Checkpoint**: Users can add URLs/files, view sources table, and delete sources

---

## Phase 4: User Story 2 - Background Jobs Monitoring (Priority: P2)

**Goal**: Users can view background ingestion jobs, monitor progress in real-time, cancel running jobs, and retry failed jobs

**Independent Test**: Add a URL source, navigate to /agents, see job running with progress bar, wait for completion or cancel

### Implementation for User Story 2

- [X] T027 [US2] Create job queue integration in src/nyrag/sources.py (extend existing jobs.py)
- [X] T028 [US2] Implement create_job() function in src/nyrag/sources.py
- [X] T029 [US2] Implement list_jobs() with status filtering in src/nyrag/sources.py
- [X] T030 [US2] Implement get_job() with logs in src/nyrag/sources.py
- [X] T031 [US2] Implement update_job_progress() for background task updates in src/nyrag/sources.py
- [X] T032 [US2] Implement run_ingestion_job() integrating crawly and process modules in src/nyrag/sources.py
- [X] T033 [P] [US2] Add GET /api/jobs endpoint (list with filtering) in src/nyrag/api.py
- [X] T034 [P] [US2] Add GET /api/jobs/{job_id} endpoint in src/nyrag/api.py
- [X] T035 [US2] Add POST /api/jobs/{job_id}/cancel endpoint in src/nyrag/api.py
- [X] T036 [US2] Add POST /api/jobs/{job_id}/retry endpoint in src/nyrag/api.py
- [X] T037 [US2] Add GET /api/jobs/{job_id}/stream SSE endpoint for real-time progress in src/nyrag/api.py
- [X] T038 [US5] Add POST /api/sources/{source_id}/sync endpoint for re-indexing in src/nyrag/api.py
- [X] T039 [US2] Add GET /api/sources/{source_id}/progress SSE endpoint in src/nyrag/api.py
- [X] T040 [P] [US2] Create Agents page template in src/nyrag/templates/agents.html
- [X] T041 [P] [US2] Create Agents page JavaScript in src/nyrag/static/agents.js
- [X] T042 [US2] Implement job list with progress bars and status badges in src/nyrag/static/agents.js
- [X] T043 [US2] Implement SSE client for real-time job progress updates in src/nyrag/static/agents.js
- [X] T044 [US2] Implement cancel/retry buttons with API integration in src/nyrag/static/agents.js
- [X] T045 [US2] Add /agents page route in src/nyrag/api.py

**Checkpoint**: Users can monitor jobs with real-time progress, cancel running jobs, retry failed jobs

---

## Phase 5: User Story 3 - Chat Conversation History (Priority: P3)

**Goal**: Users can see conversation history in sidebar, create new conversations, load previous conversations, and delete conversations

**Independent Test**: Go to /chat, send messages, refresh page, see conversation persisted in sidebar, click to reload

### Implementation for User Story 3

- [X] T046 [US3] Implement create_conversation() function in src/nyrag/sources.py
- [X] T047 [US3] Implement list_conversations() function in src/nyrag/sources.py
- [X] T048 [US3] Implement get_conversation_with_messages() function in src/nyrag/sources.py
- [X] T049 [US3] Implement add_message() function in src/nyrag/sources.py
- [X] T050 [US3] Implement delete_conversation() function in src/nyrag/sources.py
- [X] T051 [US3] Implement auto-generate conversation title from first message in src/nyrag/sources.py
- [X] T052 [P] [US3] Add GET /api/conversations endpoint in src/nyrag/api.py
- [X] T053 [P] [US3] Add POST /api/conversations endpoint in src/nyrag/api.py
- [X] T054 [US3] Add GET /api/conversations/{id} endpoint in src/nyrag/api.py
- [X] T055 [US3] Add DELETE /api/conversations/{id} endpoint in src/nyrag/api.py
- [X] T056 [US3] Modify chat endpoint to persist messages to database in src/nyrag/api.py
- [X] T057 [US3] Update chat.html to extend base.html template in src/nyrag/templates/chat.html
- [X] T058 [P] [US3] Create chat history JavaScript module in src/nyrag/static/history.js
- [X] T059 [US3] Modify chat.js to integrate conversation persistence in src/nyrag/static/chat.js
- [X] T060 [US3] Implement conversation history sidebar in chat page in src/nyrag/static/chat.js
- [X] T061 [US3] Implement load conversation and new conversation buttons in src/nyrag/static/chat.js

**Checkpoint**: Chat conversations are persisted, history sidebar works, can load/delete previous chats

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T062 [P] Add Vespa availability banner to all pages in src/nyrag/templates/base.html
- [X] T063 [P] Add error handling and user-friendly error messages across all API endpoints in src/nyrag/api.py
- [X] T064 [P] Add loading states and spinners to all UI components in src/nyrag/static/app.js
- [X] T065 Implement max 2 concurrent jobs constraint in job queue in src/nyrag/sources.py
- [X] T066 Add proper logging with loguru for all new modules in src/nyrag/sources.py and src/nyrag/database.py
- [X] T067 [P] Update notes.html to extend base.html for consistent navigation in src/nyrag/templates/notes.html
- [ ] T068 Run quickstart.md validation (end-to-end test of all features)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational - Uses job creation from US1 context but independently testable
- **User Story 3 (P3)**: Can start after Foundational - Independent from US1/US2

### Within Each User Story

- Service functions before API endpoints
- API endpoints before frontend templates
- Templates before JavaScript
- Core implementation before UI integration

### Parallel Opportunities

**Setup Phase:**
```bash
# Can run in parallel:
T002: Create base template with sidebar navigation in src/nyrag/templates/base.html
T003: Create shared CSS styles for app layout in src/nyrag/static/app.css
T004: Create shared JS for sidebar navigation in src/nyrag/static/app.js
```

**Foundational Phase:**
```bash
# Can run in parallel after T005-T006:
T007: Create Pydantic models in src/nyrag/schema.py
T008: Create uploads directory utility in src/nyrag/utils.py
```

**User Story 1:**
```bash
# Can run in parallel after T015:
T016: GET /api/sources endpoint
T017: GET /api/sources/{source_id} endpoint
T022: Data Sources page template
T023: Data Sources page JavaScript
```

**User Story 2:**
```bash
# Can run in parallel after T031:
T033: GET /api/jobs endpoint
T034: GET /api/jobs/{job_id} endpoint
T040: Agents page template
T041: Agents page JavaScript
```

**User Story 3:**
```bash
# Can run in parallel after T051:
T052: GET /api/conversations endpoint
T053: POST /api/conversations endpoint
T058: Chat history JavaScript module
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T010)
3. Complete Phase 3: User Story 1 (T011-T026)
4. **STOP and VALIDATE**: Test data sources management independently
5. Deploy/demo if ready - users can add and manage data sources

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (**MVP!**)
3. Add User Story 2 → Test independently → Deploy/Demo (adds job monitoring)
4. Add User Story 3 → Test independently → Deploy/Demo (adds chat history)
5. Each story adds value without breaking previous stories

### File Modification Summary

**New Files:**
- src/nyrag/database.py (T005, T006)
- src/nyrag/sources.py (T011-T015, T027-T032, T046-T051, T065)
- src/nyrag/templates/base.html (T002, T062)
- src/nyrag/templates/sources.html (T022)
- src/nyrag/templates/agents.html (T040)
- src/nyrag/static/app.css (T003)
- src/nyrag/static/app.js (T004, T064)
- src/nyrag/static/sources.js (T023-T025)
- src/nyrag/static/agents.js (T041-T044)
- src/nyrag/static/history.js (T058)

**Modified Files:**
- pyproject.toml (T001)
- src/nyrag/schema.py (T007)
- src/nyrag/utils.py (T008)
- src/nyrag/api.py (T009-T010, T016-T020, T026, T033-T039, T045, T052-T056, T063)
- src/nyrag/templates/chat.html (T057)
- src/nyrag/templates/notes.html (T067)
- src/nyrag/static/chat.js (T059-T061)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Max 2 concurrent jobs is enforced in T065 (Polish phase)
- SSE streaming for real-time progress implemented in T037, T039, T043
- **Terminology**: UI uses "Agents" (page name), API/DB uses "Jobs" (entities)
- **US5 Coverage**: Sync/re-index endpoint is T038 (labeled [US5])
