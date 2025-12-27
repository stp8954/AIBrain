# UI Document Manager - Implementation Tasks

## Phase 1: Database & Core Infrastructure (Priority: P0)

**Goal**: Set up SQLite database and core data models

### Tasks

- [ ] T001 [P] Create `src/nyrag/database.py` with SQLite connection manager
  - Singleton pattern for database connection
  - Auto-create database file if not exists
  - Connection pooling for async access

- [ ] T002 [P] Create data_sources table schema in `database.py`
  - id, name, type, source_path, date_added, status, progress, chunk_count, error_message, metadata

- [ ] T003 [P] Create jobs table schema in `database.py`
  - id, source_id, job_type, status, progress, current_task, started_at, completed_at, error_message, logs

- [ ] T004 [P] Create conversations and messages tables in `database.py`
  - conversations: id, title, created_at, updated_at
  - messages: id, conversation_id, role, content, created_at

- [ ] T005 Create `src/nyrag/sources.py` with DataSource model class
  - CRUD operations for data sources
  - Status enum: pending, processing, indexed, failed

- [ ] T006 Create Job model class in `sources.py` or separate file
  - CRUD operations for jobs
  - Link to existing job queue system

- [ ] T007 Add database initialization to app startup in `api.py`
  - Create tables on first run
  - Log database location

**Checkpoint**: Database schema created, models working

---

## Phase 2: UI Layout & Navigation (Priority: P1)

**Goal**: Create the new multi-page layout with sidebar navigation

### Tasks

- [ ] T008 [P] Create `src/nyrag/templates/base.html` base template
  - Left sidebar with navigation icons (Chat, Data Sources, Agents)
  - Main content area
  - Responsive design

- [ ] T009 [P] Create `src/nyrag/static/app.css` shared styles
  - Sidebar styles matching mockups
  - Navigation icons (active state)
  - Dark theme consistent with existing

- [ ] T010 [P] Create `src/nyrag/static/app.js` shared JavaScript
  - Sidebar toggle
  - Navigation state management
  - Common utilities

- [ ] T011 Update `chat.html` to extend `base.html`
  - Add chat history sidebar panel
  - Keep existing chat functionality
  - Add "Indexed: X documents • Y chunks" header

- [ ] T012 Create `src/nyrag/templates/sources.html` page
  - Extends base.html
  - Data filters sidebar (All, URLs, Files)
  - Table placeholder
  - "+ Add Source" button

- [ ] T013 Create `src/nyrag/templates/agents.html` page
  - Extends base.html
  - Agents list sidebar
  - Job cards container

- [ ] T014 Add routes in `api.py` for new pages
  - GET /sources → sources.html
  - GET /agents → agents.html
  - Update / to redirect or show chat

- [ ] T015 Update navigation link styling
  - Active page indicator
  - Hover states
  - Icons for each nav item

**Checkpoint**: Multi-page app with navigation working

---

## Phase 3: Data Sources Page (Priority: P2)

**Goal**: Implement the Data Sources table and Add Source modal

### Tasks

- [ ] T016 [P] Create `src/nyrag/static/sources.js`
  - Load and render data sources table
  - Filter by type (All, URLs, Files)
  - Status badge rendering

- [ ] T017 [P] Add `GET /api/sources` endpoint in `api.py`
  - Query data_sources table
  - Support type filter query param
  - Return list with all fields

- [ ] T018 Create Add Source modal HTML in `sources.html`
  - Tabs: "Upload Files" | "Add URL"
  - URL input field
  - Drag-drop file zone
  - File queue list
  - "Start Ingestion" button

- [ ] T019 Implement modal JavaScript in `sources.js`
  - Tab switching
  - File drag-drop handling
  - File queue management (add/remove)
  - Form validation

- [ ] T020 Add file upload styles to `app.css`
  - Drag-drop zone styling
  - File queue list styling
  - Modal styling

- [ ] T021 Implement table rendering in `sources.js`
  - Name, Type, Date Added, Status columns
  - Status badges (green=indexed, yellow=processing, red=failed)
  - Action buttons (Del, Sync, Cancel)

- [ ] T022 Add `DELETE /api/sources/{id}` endpoint
  - Remove from database
  - Remove from Vespa index
  - Clean up uploaded files

- [ ] T023 Add delete confirmation dialog
  - Modal or native confirm
  - Show source name in message

- [ ] T024 Add `POST /api/sources/{id}/sync` endpoint
  - Create new job for re-indexing
  - Update source status to processing

**Checkpoint**: Data Sources page fully functional (display only, no ingestion yet)

---

## Phase 4: Ingestion Integration (Priority: P2)

**Goal**: Connect ingestion to the new UI system

### Tasks

- [ ] T025 [P] Add `POST /api/sources/url` endpoint
  - Accept URL and optional exclude patterns
  - Create data_source record
  - Create job record
  - Start background crawl task

- [ ] T026 [P] Add `POST /api/sources/files` endpoint
  - Accept multipart file upload
  - Save files to uploads directory
  - Create data_source records for each file
  - Create job records
  - Start background processing tasks

- [ ] T027 [P] Create ingestion task wrapper in `sources.py`
  - Wrap existing crawly.crawl() for URLs
  - Wrap existing process.process_documents() for files
  - Update job progress during execution
  - Update data_source status on completion/failure

- [ ] T028 Implement progress tracking via database
  - Update job.progress periodically
  - Update job.current_task with status message
  - Store chunk count on completion

- [ ] T029 Add `GET /api/sources/{id}/progress` SSE endpoint
  - Stream progress updates for a specific source
  - Send completion/error events

- [ ] T030 Connect modal submit to API in `sources.js`
  - POST URL to /api/sources/url
  - POST files to /api/sources/files
  - Show progress in table
  - Handle errors

- [ ] T031 Add real-time table updates
  - Poll for status changes or use SSE
  - Update progress percentage
  - Flash row on status change

- [ ] T032 Handle Cancel action for processing items
  - POST /api/jobs/{id}/cancel
  - Stop running job
  - Update status to cancelled

**Checkpoint**: Full ingestion flow working from UI

---

## Phase 5: Agents/Jobs Page (Priority: P3)

**Goal**: Implement the Agents page with job cards

### Tasks

- [ ] T033 [P] Create `src/nyrag/static/agents.js`
  - Load and render job cards
  - Filter by status

- [ ] T034 Add `GET /api/jobs` endpoint enhancements
  - Include source name in response
  - Support status filter
  - Order by started_at desc

- [ ] T035 Create job card component in `agents.js`
  - Icon based on job type
  - Name and status
  - Progress bar with percentage
  - Current task message
  - Action buttons based on status

- [ ] T036 Add job card styles to `app.css`
  - Card layout matching mockups
  - Progress bar colors (blue=running, green=complete, red=failed)
  - Status text colors

- [ ] T037 Implement View Logs functionality
  - Modal or expandable section
  - Show job.logs content
  - Auto-scroll to bottom

- [ ] T038 Implement Stop Agent action
  - POST /api/jobs/{id}/cancel
  - Confirm before stopping
  - Update UI on success

- [ ] T039 Implement Retry action for failed jobs
  - POST /api/jobs/{id}/retry
  - Create new job with same params
  - Navigate to new job

- [ ] T040 Add real-time job updates
  - SSE or polling for progress
  - Update cards without full reload
  - Show toast on completion/failure

**Checkpoint**: Agents page fully functional

---

## Phase 6: Chat History (Priority: P3)

**Goal**: Persist chat conversations and show history

### Tasks

- [ ] T041 [P] Add conversation CRUD endpoints
  - GET /api/conversations - list all
  - POST /api/conversations - create new
  - GET /api/conversations/{id} - get with messages
  - DELETE /api/conversations/{id} - delete

- [ ] T042 Update chat-stream to save messages
  - Create conversation on first message if new
  - Save user message before processing
  - Save assistant message after completion

- [ ] T043 Add chat history sidebar to `chat.html`
  - List of conversations with titles
  - "New Chat +" button at top
  - Click to load conversation

- [ ] T044 Create `src/nyrag/static/history.js` or update `chat.js`
  - Load conversation list
  - Render in sidebar
  - Handle click to load
  - Handle new chat creation

- [ ] T045 Implement conversation loading
  - GET /api/conversations/{id}
  - Render messages in chat area
  - Set current conversation ID

- [ ] T046 Auto-generate conversation titles
  - Use first user message (truncated)
  - Or extract topic from content

- [ ] T047 Add delete conversation action
  - Swipe or button to delete
  - Confirm before deleting
  - Update list after deletion

- [ ] T048 Update URL routing for conversations
  - /chat → new chat or last conversation
  - /chat/{id} → specific conversation
  - Update browser URL on navigation

**Checkpoint**: Chat history fully functional

---

## Phase 7: Polish & Integration (Priority: P4)

**Goal**: Final touches and integration testing

### Tasks

- [ ] T049 Add "Indexed: X documents • Y chunks" to chat header
  - Query total from data_sources
  - Update periodically

- [ ] T050 Add loading states throughout
  - Skeleton loaders for tables
  - Spinner for job cards
  - Button loading states

- [ ] T051 Add error handling throughout
  - Toast notifications for errors
  - Retry buttons where appropriate
  - Graceful degradation

- [ ] T052 Add empty states
  - "No data sources yet" with CTA
  - "No jobs running" message
  - "No conversations yet" message

- [ ] T053 Mobile responsive adjustments
  - Collapsible sidebar on mobile
  - Touch-friendly buttons
  - Swipe gestures where appropriate

- [ ] T054 Update README with new UI documentation
  - Screenshots of new pages
  - Feature descriptions
  - Usage guide

- [ ] T055 Run `make quality` and fix issues

- [ ] T056 End-to-end testing
  - Add URL → verify indexed
  - Upload file → verify indexed
  - Query content in chat
  - Delete and verify removed

**Checkpoint**: Feature complete and polished

---

## Dependencies

```
Phase 1 (Database) ─────► Phase 2 (Layout) ─────┬──► Phase 3 (Sources UI)
                                                 │
                                                 ├──► Phase 5 (Agents UI)
                                                 │
                                                 └──► Phase 6 (Chat History)
                                                 
Phase 3 + Phase 4 ──► Phase 4 (Ingestion)

All Phases ──► Phase 7 (Polish)
```

---

## Estimated Effort

- **Phase 1**: 2-3 hours (database setup)
- **Phase 2**: 3-4 hours (UI layout)
- **Phase 3**: 4-5 hours (data sources page)
- **Phase 4**: 4-5 hours (ingestion integration)
- **Phase 5**: 3-4 hours (agents page)
- **Phase 6**: 3-4 hours (chat history)
- **Phase 7**: 2-3 hours (polish)

**Total**: ~22-28 hours of implementation
