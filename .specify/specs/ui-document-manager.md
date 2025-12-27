# UI Document Manager

## Overview

**Feature**: UI-based document management with a left sidebar navigation, data sources table view, background agents/jobs monitoring, and the ability to add new resources (URLs, files) with background ingestion and progress tracking.

**Goal**: Allow users to manage all content ingestion from the web UI without needing to edit config files or run CLI commands.

**Reference**: See mockup images for UI design.

---

## UI Structure (from mockups)

### Left Sidebar Navigation
- **Chat** - Main chat interface with chat history list
- **Data Sources** - Table view of all indexed resources
- **Agents** - Background jobs/agents monitoring

### Pages

1. **Chat Page** (existing, enhanced)
   - Left panel: Chat history list (New Chat +, previous conversations)
   - Main area: Chat interface with "Indexed: X documents • Y chunks"

2. **Data Sources Page** (new)
   - Left panel: Data Filters (All, URLs, Files)
   - Main area: Table with Name, Type, Date Added, Status, Actions
   - "+ Add Source" button → Modal with tabs: "Upload Files" | "Add URL"
   - Status: ● Indexed (green), ● Processing (X%) (yellow)
   - Actions: [Del] [Sync] or [Cancel] for processing items

3. **Agents Page** (new)
   - Left panel: Agents List (All)
   - Main area: Cards showing background jobs
   - Job card: Icon, Name, Status (Running/Completed/Failed), Progress bar
   - Actions: [View Logs] [Stop Agent] or [View Report] [Run Now] or [View Error Log] [Retry]

---

## User Stories

### US1: View Data Sources
As a user, I want to see all my indexed resources in a table view, so I can understand what content is available.

**Acceptance Criteria**:
- Table shows: Name, Type (URL/PDF/Markdown/etc), Date Added, Status, Actions
- Filter by type (All, URLs, Files)
- Status shows "Indexed" (green) or "Processing (X%)" (yellow) or "Failed" (red)
- Actions: Delete, Sync (re-index), Cancel (for processing)
- **If Vespa unavailable**: Show error banner "Vespa not available - Run `docker-compose up -d` to start" at top of page; disable Add Source button

### US2: Add URL for Crawling
As a user, I want to add a base URL via modal and have the system crawl and index it.

**Acceptance Criteria**:
- "+ Add Source" button opens modal
- "Add URL" tab: URL input field
- Submit starts background crawl job
- New entry appears in table with "Processing" status
- Progress updates in real-time

### US3: Upload Files
As a user, I want to upload PDF/MD/TXT files via drag-drop or file picker.

**Acceptance Criteria**:
- "Upload Files" tab in modal
- Drag-drop zone: "Drag and drop files here or click to browse"
- Supported: .pdf, .md, .txt
- **Maximum file size: 50MB per file** (show error for oversized files)
- Files stored in `uploads/` directory within data folder
- File queue shows files ready to upload with X to remove
- "Start Ingestion" button processes all files
- Files appear in table as they're processed

### US4: Delete Resources
As a user, I want to delete indexed resources from the table.

**Acceptance Criteria**:
- [Del] action on each row
- Confirmation before deletion
- Removes from Vespa index and metadata storage
- **Chat conversations referencing deleted chunks remain intact** (historical preservation)

### US5: Sync/Re-index Resources
As a user, I want to re-crawl a URL or re-process a file.

**Acceptance Criteria**:
- [Sync] action on indexed items
- Triggers new ingestion job
- Updates existing chunks in Vespa

### US6: View Background Agents/Jobs
As a user, I want to monitor all background ingestion jobs.

**Acceptance Criteria**:
- Agents page shows job cards
- Each card: icon, name, status, progress bar, current task
- **Max 2 concurrent jobs**; additional jobs queued with "Queued" status
- Running: blue progress bar, [View Logs] [Stop Agent]
- Completed: green checkmark, [View Report] [Run Now]
- Failed: red X, progress at failure point, [View Error Log] [Retry]

### US7: Chat History
As a user, I want to see my previous chat conversations.

**Acceptance Criteria**:
- Left panel on Chat page shows conversation list
- Click to load previous conversation
- "New Chat +" to start fresh

---

## Technical Approach

### Storage: SQLite Database

Use SQLite for metadata storage (not Vespa, which is for vector search):

```sql
-- Data sources metadata
CREATE TABLE data_sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'url', 'pdf', 'markdown', 'txt'
    source_path TEXT,    -- URL or original file path
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending',  -- 'pending', 'processing', 'indexed', 'failed'
    progress INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSON
);

-- Ingestion jobs
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES data_sources(id),
    job_type TEXT NOT NULL,  -- 'crawl', 'process_file', 'sync'
    status TEXT DEFAULT 'queued',  -- 'queued', 'running', 'completed', 'failed', 'cancelled'
    progress INTEGER DEFAULT 0,
    current_task TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    logs TEXT
);

-- Chat conversations
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    role TEXT NOT NULL,  -- 'user', 'assistant'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Backend API Endpoints

**Data Sources:**
- `GET /api/sources` - List all data sources (with filter by type)
- `POST /api/sources/url` - Add URL for crawling
- `POST /api/sources/files` - Upload files for processing
- `DELETE /api/sources/{id}` - Delete a data source
- `POST /api/sources/{id}/sync` - Re-index a source

**Jobs/Agents:**
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{id}` - Get job details with logs
- `POST /api/jobs/{id}/cancel` - Cancel a running job
- `POST /api/jobs/{id}/retry` - Retry a failed job
- `GET /api/jobs/{id}/stream` - SSE stream for job progress

**Chat History:**
- `GET /api/conversations` - List conversations
- `POST /api/conversations` - Create new conversation
- `GET /api/conversations/{id}` - Get conversation with messages
- `DELETE /api/conversations/{id}` - Delete conversation

### Frontend Structure

```
/                    → Redirect to /chat
/chat                → Chat page with history sidebar
/chat/{conv_id}      → Load specific conversation
/sources             → Data Sources page
/agents              → Agents/Jobs page
/notes               → Notes editor (existing)
```

### File Structure (new files)

```
src/nyrag/
├── database.py          # SQLite connection and models
├── sources.py           # Data source management logic
├── static/
│   ├── app.css          # Shared styles for new layout
│   ├── app.js           # Shared JS (sidebar, navigation)
│   ├── sources.js       # Data Sources page logic
│   └── agents.js        # Agents page logic
└── templates/
    ├── base.html        # Base template with sidebar
    ├── sources.html     # Data Sources page
    └── agents.html      # Agents page
```

---

## Implementation Phases

### Phase 1: Database & Core Infrastructure
- SQLite database setup
- Data source model and CRUD
- Job tracking integration

### Phase 2: UI Layout & Navigation
- Base template with left sidebar
- Navigation between Chat/Sources/Agents
- Update existing pages to use new layout

### Phase 3: Data Sources Page
- Table view with filtering
- Add Source modal (URL + File upload)
- Delete and status display

### Phase 4: Ingestion Integration
- Connect URL crawling to new system
- File upload and processing
- Progress tracking via SSE

### Phase 5: Agents/Jobs Page
- Job cards UI
- Real-time progress updates
- Cancel/Retry functionality

### Phase 6: Chat History
- Conversation persistence
- History sidebar
- Load previous conversations

---

## Clarifications

### Session 2025-12-25
- Q: Where should uploaded files be stored and what is the maximum file size? → A: Store in `uploads/` folder with 50MB per-file limit
- Q: What is the URL crawl scope and depth behavior? → A: Use existing crawling logic (already implemented in crawly module)
- Q: How should concurrent ingestion jobs be handled? → A: Max 2 concurrent jobs, queue additional jobs
- Q: How should the UI behave when Vespa is unavailable? → A: Show error banner "Vespa not available" with setup instructions
- Q: When deleting a data source, should existing chat conversations be affected? → A: Keep conversations intact (historical record preserved)

---

## Out of Scope (Future)

- Image processing/OCR
- Batch URL import from file
- Scheduled re-crawls
- URL depth/scope configuration in UI
- Document version history
- Collaborative features

