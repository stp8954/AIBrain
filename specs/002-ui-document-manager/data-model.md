# Data Model: UI Document Manager

**Date**: 2025-12-25  
**Feature**: UI Document Manager

## Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐
│   DataSource    │───1:N─│      Job        │
├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │
│ name            │       │ source_id (FK)  │
│ type            │       │ job_type        │
│ source_path     │       │ status          │
│ date_added      │       │ progress        │
│ status          │       │ current_task    │
│ progress        │       │ started_at      │
│ chunk_count     │       │ completed_at    │
│ error_message   │       │ error_message   │
│ metadata        │       │ logs            │
└─────────────────┘       └─────────────────┘

┌─────────────────┐       ┌─────────────────┐
│  Conversation   │───1:N─│    Message      │
├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │
│ title           │       │ conversation_id │
│ created_at      │       │ role            │
│ updated_at      │       │ content         │
└─────────────────┘       │ created_at      │
                          └─────────────────┘
```

## Entities

### DataSource

Represents an indexed content source (URL or uploaded file).

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PRIMARY KEY | UUID, auto-generated |
| `name` | TEXT | NOT NULL | Display name (extracted from URL/filename) |
| `type` | TEXT | NOT NULL | `url`, `pdf`, `markdown`, `txt` |
| `source_path` | TEXT | NULLABLE | Original URL or file path |
| `date_added` | TIMESTAMP | DEFAULT NOW | When source was added |
| `status` | TEXT | DEFAULT 'pending' | `pending`, `processing`, `indexed`, `failed` |
| `progress` | INTEGER | DEFAULT 0 | Processing progress 0-100 |
| `chunk_count` | INTEGER | DEFAULT 0 | Number of chunks after indexing |
| `error_message` | TEXT | NULLABLE | Error details if failed |
| `metadata` | JSON | NULLABLE | Additional source metadata |

**Validation Rules**:
- `type` must be one of: `url`, `pdf`, `markdown`, `txt`
- `status` must be one of: `pending`, `processing`, `indexed`, `failed`
- `progress` must be between 0 and 100
- `name` cannot be empty

**State Transitions**:
```
pending → processing → indexed
                    ↘ failed
indexed → processing (sync/re-index)
```

---

### Job

Represents a background ingestion or processing task.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PRIMARY KEY | UUID, auto-generated |
| `source_id` | TEXT | FOREIGN KEY | Reference to DataSource |
| `job_type` | TEXT | NOT NULL | `crawl`, `process_file`, `sync` |
| `status` | TEXT | DEFAULT 'queued' | `queued`, `running`, `completed`, `failed`, `cancelled` |
| `progress` | INTEGER | DEFAULT 0 | Job progress 0-100 |
| `current_task` | TEXT | NULLABLE | Human-readable current step |
| `started_at` | TIMESTAMP | DEFAULT NOW | When job was created/started |
| `completed_at` | TIMESTAMP | NULLABLE | When job finished |
| `error_message` | TEXT | NULLABLE | Error details if failed |
| `logs` | TEXT | NULLABLE | Full job logs (newline-separated) |

**Validation Rules**:
- `job_type` must be one of: `crawl`, `process_file`, `sync`
- `status` must be one of: `queued`, `running`, `completed`, `failed`, `cancelled`
- `progress` must be between 0 and 100
- `source_id` must reference valid DataSource

**State Transitions**:
```
queued → running → completed
              ↘ failed
              ↘ cancelled
failed → queued (retry)
```

**Concurrency Rules**:
- Maximum 2 jobs can be in `running` state simultaneously
- Additional jobs remain in `queued` state until slot available

---

### Conversation

Represents a chat conversation session.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PRIMARY KEY | UUID, auto-generated |
| `title` | TEXT | NULLABLE | Display title (auto-generated from first message) |
| `created_at` | TIMESTAMP | DEFAULT NOW | When conversation started |
| `updated_at` | TIMESTAMP | DEFAULT NOW | Last activity timestamp |

**Validation Rules**:
- `title` auto-generated if not provided (first 50 chars of first user message)
- `updated_at` auto-updates on new message

---

### Message

Represents a single message within a conversation.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PRIMARY KEY | UUID, auto-generated |
| `conversation_id` | TEXT | FOREIGN KEY, NOT NULL | Reference to Conversation |
| `role` | TEXT | NOT NULL | `user` or `assistant` |
| `content` | TEXT | NOT NULL | Message text content |
| `created_at` | TIMESTAMP | DEFAULT NOW | When message was created |

**Validation Rules**:
- `role` must be one of: `user`, `assistant`
- `content` cannot be empty
- `conversation_id` must reference valid Conversation

---

## SQLite Schema

```sql
-- Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Data sources metadata
CREATE TABLE IF NOT EXISTS data_sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('url', 'pdf', 'markdown', 'txt')),
    source_path TEXT,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'indexed', 'failed')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    chunk_count INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSON
);

-- Ingestion jobs
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES data_sources(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL CHECK (job_type IN ('crawl', 'process_file', 'sync')),
    status TEXT DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    current_task TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    logs TEXT
);

-- Chat conversations
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat messages
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_data_sources_status ON data_sources(status);
CREATE INDEX IF NOT EXISTS idx_data_sources_type ON data_sources(type);
CREATE INDEX IF NOT EXISTS idx_jobs_source_id ON jobs(source_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
```

---

## Pydantic Models

```python
from datetime import datetime
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
import uuid

# Enums as Literals
DataSourceType = Literal['url', 'pdf', 'markdown', 'txt']
DataSourceStatus = Literal['pending', 'processing', 'indexed', 'failed']
JobType = Literal['crawl', 'process_file', 'sync']
JobStatus = Literal['queued', 'running', 'completed', 'failed', 'cancelled']
MessageRole = Literal['user', 'assistant']


class DataSourceCreate(BaseModel):
    """Request to create a new data source"""
    name: str = Field(..., min_length=1)
    type: DataSourceType
    source_path: str


class DataSourceResponse(BaseModel):
    """Data source response model"""
    id: str
    name: str
    type: DataSourceType
    source_path: Optional[str]
    date_added: datetime
    status: DataSourceStatus
    progress: int = Field(ge=0, le=100)
    chunk_count: int = 0
    error_message: Optional[str] = None


class JobResponse(BaseModel):
    """Job response model"""
    id: str
    source_id: str
    source_name: Optional[str] = None  # Joined from data_sources
    job_type: JobType
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    current_task: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ConversationResponse(BaseModel):
    """Conversation response model"""
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0  # Optional: computed field


class MessageResponse(BaseModel):
    """Message response model"""
    id: str
    role: MessageRole
    content: str
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    """Conversation with messages"""
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]
```

---

## Vespa Integration

### Chunk Document Structure

Chunks stored in Vespa will include `source_id` for linking back to data sources:

```python
{
    "id": "chunk-uuid",
    "source_id": "datasource-uuid",  # Links to SQLite data_sources.id
    "content": "...",
    "embedding": [...],
    "url": "...",  # Original source URL/path
    "title": "...",
    # ... other existing fields
}
```

### Deletion Query

When deleting a data source, remove all associated chunks:

```
DELETE FROM schema WHERE source_id = '<source_id>'
```
