# Feature Specification: UI Document Manager

**Feature Branch**: `002-ui-document-manager`  
**Created**: 2025-12-25  
**Status**: In Planning  
**Full Spec**: [.specify/specs/ui-document-manager.md](../../.specify/specs/ui-document-manager.md)

## Summary

This feature adds a web-based document management UI to NyRAG with:
- Left sidebar navigation (Chat, Data Sources, Agents)
- Data sources table view with filtering and CRUD operations
- File upload (drag-drop) and URL crawling via modal
- Background job monitoring with real-time progress
- Chat conversation history persistence

See the [full specification](../../.specify/specs/ui-document-manager.md) for complete details including:
- User stories with acceptance criteria
- Technical approach with SQLite schema
- API endpoints and file structure
- Implementation phases

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Data Sources Management (Priority: P1) 🎯 MVP

As a user, I want to add URLs and files as data sources, view them in a table, and delete them, so I can manage what content is available for RAG queries.

**Why this priority**: This is the core value proposition - users need to be able to add content before they can query it. Without data sources, the system has nothing to search.

**Independent Test**: Navigate to /sources, add a URL, see it appear in table with processing status, wait for indexing, then delete it. Delivers immediate value: content management without CLI.

**Acceptance Scenarios**:

1. **Given** the Data Sources page is loaded, **When** user clicks "+ Add Source" and enters a URL, **Then** a new row appears in the table with "Processing" status
2. **Given** a data source exists in the table, **When** user clicks [Del] and confirms, **Then** the source is removed from the table and Vespa index
3. **Given** files are dragged into the upload zone, **When** user clicks "Start Ingestion", **Then** files are uploaded and appear in table with progress indicators
4. **Given** a file larger than 50MB is selected, **When** user attempts to upload, **Then** an error message is shown and upload is prevented

---

### User Story 2 - Background Jobs Monitoring (Priority: P2)

As a user, I want to view background ingestion jobs, monitor progress in real-time, cancel running jobs, and retry failed jobs, so I understand what the system is doing.

**Why this priority**: Once users can add sources (P1), they need visibility into processing status. This provides transparency and control over long-running operations.

**Independent Test**: Add a URL source, navigate to /agents, see job running with progress bar, wait for completion or cancel it. Delivers value: observability without checking logs.

**Acceptance Scenarios**:

1. **Given** an ingestion job is running, **When** user views /agents page, **Then** a job card shows with progress bar and current task description
2. **Given** a job is running, **When** user clicks [Stop Agent], **Then** the job is cancelled and status updates to "Cancelled"
3. **Given** a job has failed, **When** user clicks [Retry], **Then** a new job is created with the same parameters
4. **Given** 2 jobs are already running, **When** a third source is added, **Then** the new job shows "Queued" status until a slot opens

---

### User Story 3 - Chat Conversation History (Priority: P3)

As a user, I want to see my previous chat conversations, load them, and start new ones, so I can continue previous discussions and reference past answers.

**Why this priority**: Chat history enhances the existing chat functionality but isn't required for core RAG operations. Users can query content (P1) and monitor jobs (P2) first.

**Independent Test**: Go to /chat, send a message, refresh the page, see conversation persisted in sidebar, click to reload it. Delivers value: conversation persistence.

**Acceptance Scenarios**:

1. **Given** a user sends a message in chat, **When** the page is refreshed, **Then** the conversation appears in the history sidebar
2. **Given** multiple conversations exist, **When** user clicks one in the sidebar, **Then** the chat area loads that conversation's messages
3. **Given** user is viewing a conversation, **When** they click "New Chat +", **Then** the chat area clears for a fresh conversation
4. **Given** a conversation exists, **When** user deletes it, **Then** it is removed from the sidebar and database

---

### Edge Cases

- **Vespa unavailable**: Show error banner "Vespa not available - Run `docker-compose up -d` to start" and disable Add Source button
- **File too large**: Reject uploads over 50MB with clear error message before upload starts
- **Concurrent job limit**: Queue jobs beyond the 2-concurrent limit with "Queued" status
- **Delete during processing**: Cancel the associated job before removing the data source
- **Deleted source referenced in chat**: Keep chat conversations intact (historical preservation)
- **Duplicate URL**: Allow re-adding same URL (creates new source, user can delete old one)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display all data sources in a filterable table (by type: All, URLs, Files)
- **FR-002**: System MUST allow adding URLs for crawling via modal with real-time progress feedback
- **FR-003**: System MUST allow uploading PDF, MD, TXT files via drag-drop with 50MB max size per file
- **FR-004**: System MUST store uploaded files in `uploads/` directory within data folder
- **FR-005**: System MUST allow deleting data sources, removing them from both SQLite and Vespa
- **FR-006**: System MUST allow re-indexing (sync) existing sources
- **FR-007**: System MUST display background jobs with real-time progress via SSE
- **FR-008**: System MUST limit concurrent running jobs to 2, queuing additional jobs
- **FR-009**: System MUST allow cancelling running jobs and retrying failed jobs
- **FR-010**: System MUST persist chat conversations to SQLite database
- **FR-011**: System MUST display chat history in sidebar with ability to load/delete conversations
- **FR-012**: System MUST show Vespa availability banner when Vespa is unreachable

### Key Entities *(include if feature involves data)*

- **DataSource**: Indexed content source (URL or file). Key attributes: id, name, type, source_path, status, progress, chunk_count. Status transitions: pending → processing → indexed/failed.
- **Job**: Background ingestion task. Key attributes: id, source_id, job_type, status, progress, current_task, logs. Types: crawl, process_file, sync. Max 2 concurrent.
- **Conversation**: Chat session container. Key attributes: id, title (auto-generated), created_at, updated_at. Has many Messages.
- **Message**: Single chat message. Key attributes: id, conversation_id, role (user/assistant), content, created_at.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can add a URL and see it indexed within 60 seconds for single-page sites
- **SC-002**: File uploads complete within 30 seconds for files up to 50MB
- **SC-003**: Job progress updates appear in UI within 2 seconds of backend progress change (via SSE)
- **SC-004**: API endpoints respond in under 200ms for list operations
- **SC-005**: Users can manage data sources entirely through UI without using CLI commands
- **SC-006**: Chat conversations persist across browser sessions and page refreshes
- **SC-007**: System gracefully handles Vespa unavailability with clear user messaging
