# Research: UI Document Manager

**Date**: 2025-12-25  
**Feature**: UI Document Manager  
**Status**: Complete

## Research Areas

### 1. SQLite Async Integration with FastAPI

**Decision**: Use `aiosqlite` for async SQLite access

**Rationale**:
- FastAPI is async-first; blocking SQLite calls would reduce throughput
- `aiosqlite` provides async context managers and cursor operations
- Minimal overhead, production-proven for small-medium metadata stores
- Alternative `databases` library adds complexity without benefit for SQLite

**Alternatives Considered**:
- **Sync SQLite + thread pool**: Works but adds complexity with `run_in_executor`
- **SQLAlchemy async**: Overkill for simple CRUD; adds ORM overhead
- **databases library**: Supports async but adds dependency for same result

**Implementation Pattern**:
```python
import aiosqlite

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self):
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
    
    async def execute(self, query: str, params: tuple = ()):
        async with self._connection.execute(query, params) as cursor:
            return await cursor.fetchall()
```

---

### 2. File Upload Handling in FastAPI

**Decision**: Use FastAPI's built-in `UploadFile` with chunked streaming to disk

**Rationale**:
- `UploadFile` provides SpooledTemporaryFile with configurable threshold
- Chunked writes avoid memory issues for large files (up to 50MB limit)
- Simple validation before processing

**Alternatives Considered**:
- **python-multipart only**: Already used by FastAPI internally
- **S3/object storage**: Out of scope; adds infrastructure complexity
- **Base64 encoding**: Inefficient for binary files

**Implementation Pattern**:
```python
from fastapi import UploadFile, HTTPException
import aiofiles

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

async def save_upload(file: UploadFile, dest_path: Path) -> int:
    size = 0
    async with aiofiles.open(dest_path, 'wb') as f:
        while chunk := await file.read(8192):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                raise HTTPException(413, "File too large (max 50MB)")
            await f.write(chunk)
    return size
```

---

### 3. Real-time Progress Updates (SSE vs WebSocket)

**Decision**: Use Server-Sent Events (SSE) for progress streaming

**Rationale**:
- Unidirectional server→client fits job progress pattern perfectly
- Native browser support via EventSource, simpler than WebSocket
- FastAPI's `StreamingResponse` handles SSE elegantly
- Automatic reconnection built into EventSource API

**Alternatives Considered**:
- **WebSocket**: Bidirectional overhead not needed; more complex client code
- **Polling**: Higher latency, more server load
- **Long polling**: Complex to implement correctly

**Implementation Pattern**:
```python
from fastapi.responses import StreamingResponse
import asyncio

async def progress_generator(job_id: str):
    while True:
        job = await get_job(job_id)
        yield f"data: {json.dumps({'progress': job.progress, 'status': job.status})}\n\n"
        if job.status in ('completed', 'failed', 'cancelled'):
            break
        await asyncio.sleep(0.5)

@app.get("/api/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    return StreamingResponse(
        progress_generator(job_id),
        media_type="text/event-stream"
    )
```

---

### 4. Background Job Execution Strategy

**Decision**: Extend existing `jobs.py` queue system with database persistence

**Rationale**:
- Project already has `JobQueue` in `jobs.py` with async task execution
- Adding SQLite persistence maintains job state across restarts
- Max 2 concurrent jobs aligns with existing queue design
- No new dependencies required

**Alternatives Considered**:
- **Celery/RQ**: Heavy dependencies, requires Redis/RabbitMQ
- **Dramatiq**: Good but adds complexity for simple use case
- **APScheduler**: Better for scheduled tasks, not ad-hoc jobs

**Integration Pattern**:
```python
# Wrap existing crawl/process functions
async def run_ingestion_job(job_id: str, source_id: str):
    try:
        await update_job_status(job_id, 'running')
        # Call existing pipeline functions
        if source.type == 'url':
            await crawl_and_ingest(source.source_path, progress_callback)
        else:
            await process_file(source.source_path, progress_callback)
        await update_job_status(job_id, 'completed')
    except Exception as e:
        await update_job_status(job_id, 'failed', error=str(e))
```

---

### 5. Template Inheritance Strategy

**Decision**: Create `base.html` with Jinja2 blocks for sidebar and content

**Rationale**:
- Existing templates use Jinja2; consistent approach
- Block structure allows per-page customization
- Sidebar navigation shared across all pages
- Minimal changes to existing `chat.html`

**Alternatives Considered**:
- **React/Vue SPA**: Complete rewrite, out of scope
- **HTMX**: Good for dynamic updates but adds learning curve
- **Server components**: Not applicable to Python/Jinja2

**Template Structure**:
```html
<!-- base.html -->
<body>
  <nav class="sidebar">
    {% block sidebar %}
    <a href="/chat" class="{% if active == 'chat' %}active{% endif %}">Chat</a>
    <a href="/sources" class="{% if active == 'sources' %}active{% endif %}">Data Sources</a>
    <a href="/agents" class="{% if active == 'agents' %}active{% endif %}">Agents</a>
    {% endblock %}
  </nav>
  <main>{% block content %}{% endblock %}</main>
</body>
```

---

### 6. Vespa Chunk Deletion Strategy

**Decision**: Use Vespa document API with parent document ID filtering

**Rationale**:
- Chunks are stored with `source_id` field linking to data source
- Vespa supports deletion by query/selection
- Atomic deletion ensures consistency

**Implementation Pattern**:
```python
async def delete_source_chunks(vespa_client, schema_name: str, source_id: str):
    # Delete all documents where source_id matches
    response = await vespa_client.delete_all_docs(
        schema=schema_name,
        selection=f"{schema_name}.source_id == '{source_id}'"
    )
    return response
```

---

## Clarifications Resolved

| Question | Resolution | Source |
|----------|------------|--------|
| File storage location | `uploads/` directory within project data folder | Spec clarification |
| Max file size | 50MB per file | Spec clarification |
| Concurrent job limit | Max 2 concurrent, queue additional | Spec clarification |
| Vespa unavailable behavior | Error banner with setup instructions | Spec clarification |
| Deleted source → chat impact | Conversations preserved (historical) | Spec clarification |

---

## Dependencies to Add

```toml
# pyproject.toml additions
aiosqlite = "^0.19.0"
aiofiles = "^23.2.1"
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SQLite lock contention | Low | Medium | Use WAL mode, connection pooling |
| Large file memory issues | Low | High | Chunked streaming, size validation |
| Job state inconsistency | Medium | Medium | Database transactions, status checks |
| Vespa timeout on deletion | Low | Medium | Async with timeout, retry logic |
