"""
SQLite database module for NyRAG UI Document Manager.

Provides async connection management and schema initialization for:
- Data sources metadata
- Ingestion jobs tracking
- Chat conversations and messages
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiosqlite

from nyrag.logger import get_logger


logger = get_logger(__name__)

# Default database path (can be overridden via config)
DEFAULT_DB_PATH = Path("data/nyrag.db")

# Global database path (set during initialization)
_db_path: Optional[Path] = None


# =============================================================================
# Schema Definition
# =============================================================================

SCHEMA_SQL = """
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
    image_count INTEGER DEFAULT 0,
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

-- Image chunks for extracted images from documents
CREATE TABLE IF NOT EXISTS image_chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    page_number INTEGER,
    dimensions TEXT NOT NULL,  -- JSON: {"width": int, "height": int}
    caption TEXT,
    created_at TEXT NOT NULL,
    indexed INTEGER DEFAULT 0,
    FOREIGN KEY (source_id) REFERENCES data_sources(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_source ON image_chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_image_hash ON image_chunks(content_hash);
"""

# Migration SQL for existing databases (adds new columns if they don't exist)
MIGRATION_SQL = """
-- Add image_count column to data_sources if it doesn't exist
-- SQLite doesn't support IF NOT EXISTS for columns, so we use a try/except in Python
"""


# =============================================================================
# Database Initialization
# =============================================================================


def get_db_path() -> Path:
    """Get the current database path."""
    global _db_path
    if _db_path is None:
        # Check for environment variable override
        env_path = os.getenv("NYRAG_DB_PATH")
        if env_path:
            _db_path = Path(env_path)
        else:
            _db_path = DEFAULT_DB_PATH
    return _db_path


def set_db_path(path: Path) -> None:
    """Set the database path (call before init_database)."""
    global _db_path
    _db_path = path
    logger.debug(f"Database path set to: {path}")


async def init_database(db_path: Optional[Path] = None) -> None:
    """Initialize the database with schema.

    Creates the database file and all required tables if they don't exist.

    Args:
        db_path: Optional path to database file. Uses default if not provided.
    """
    if db_path:
        set_db_path(db_path)

    path = get_db_path()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing database at: {path}")

    async with aiosqlite.connect(path) as db:
        # Execute schema creation
        await db.executescript(SCHEMA_SQL)
        await db.commit()

        # Run migrations for existing databases
        await _run_migrations(db)

    logger.info("Database schema initialized successfully")


async def _run_migrations(db: aiosqlite.Connection) -> None:
    """Run database migrations for existing databases.

    Adds new columns that may not exist in older database versions.
    """
    # Check if image_count column exists in data_sources
    try:
        cursor = await db.execute("SELECT image_count FROM data_sources LIMIT 1")
        await cursor.fetchone()
    except Exception:
        # Column doesn't exist, add it
        logger.info("Adding image_count column to data_sources table")
        await db.execute("ALTER TABLE data_sources ADD COLUMN image_count INTEGER DEFAULT 0")
        await db.commit()


# =============================================================================
# Connection Management
# =============================================================================


@asynccontextmanager
async def get_db():
    """Async context manager for database connections.

    Usage:
        async with get_db() as db:
            await db.execute("SELECT * FROM data_sources")
            rows = await db.fetchall()

    Yields:
        aiosqlite.Connection: Database connection with row factory set
    """
    path = get_db_path()

    if not path.exists():
        raise RuntimeError(f"Database not initialized. Call init_database() first. " f"Expected path: {path}")

    async with aiosqlite.connect(path) as db:
        # Enable foreign keys for this connection
        await db.execute("PRAGMA foreign_keys=ON")
        # Return rows as sqlite3.Row for dict-like access
        db.row_factory = aiosqlite.Row
        yield db


# =============================================================================
# Database Statistics
# =============================================================================


async def get_database_stats() -> dict:
    """Get statistics about the database contents.

    Returns:
        dict: Statistics including counts of sources, jobs, conversations, etc.
    """
    try:
        async with get_db() as db:
            stats = {}

            # Count data sources by status
            cursor = await db.execute("SELECT status, COUNT(*) as count FROM data_sources GROUP BY status")
            source_stats = await cursor.fetchall()
            stats["sources"] = {
                "total": sum(row["count"] for row in source_stats),
                "by_status": {row["status"]: row["count"] for row in source_stats},
            }

            # Total chunks
            cursor = await db.execute("SELECT COALESCE(SUM(chunk_count), 0) as total FROM data_sources")
            row = await cursor.fetchone()
            stats["chunks"] = row["total"] if row else 0

            # Count jobs by status
            cursor = await db.execute("SELECT status, COUNT(*) as count FROM jobs GROUP BY status")
            job_stats = await cursor.fetchall()
            stats["jobs"] = {
                "total": sum(row["count"] for row in job_stats),
                "by_status": {row["status"]: row["count"] for row in job_stats},
            }

            # Running jobs count (for concurrency limit)
            cursor = await db.execute("SELECT COUNT(*) as count FROM jobs WHERE status = 'running'")
            row = await cursor.fetchone()
            stats["running_jobs"] = row["count"] if row else 0

            # Count conversations
            cursor = await db.execute("SELECT COUNT(*) as count FROM conversations")
            row = await cursor.fetchone()
            stats["conversations"] = row["count"] if row else 0

            return stats

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {
            "sources": {"total": 0, "by_status": {}},
            "chunks": 0,
            "jobs": {"total": 0, "by_status": {}},
            "running_jobs": 0,
            "conversations": 0,
        }


async def check_database_available() -> bool:
    """Check if database is available and initialized.

    Returns:
        bool: True if database is accessible, False otherwise
    """
    try:
        path = get_db_path()
        if not path.exists():
            return False

        async with get_db() as db:
            # Simple query to verify connection
            await db.execute("SELECT 1")
            return True
    except Exception as e:
        logger.warning(f"Database availability check failed: {e}")
        return False


# =============================================================================
# Image Chunk CRUD Operations
# =============================================================================


async def create_image_chunk(
    image_id: str,
    source_id: str,
    content_hash: str,
    file_path: str,
    dimensions: dict,
    page_number: int | None = None,
    caption: str | None = None,
) -> dict:
    """Create a new image chunk record.

    Args:
        image_id: Unique ID for the image chunk
        source_id: Parent data source ID
        content_hash: SHA-256 hash of image content
        file_path: Relative path in images directory
        dimensions: Dict with 'width' and 'height' keys
        page_number: Page number in source document (1-indexed)
        caption: Optional image caption

    Returns:
        Created image chunk record as dict
    """
    import json
    from datetime import datetime

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO image_chunks (id, source_id, content_hash, file_path, page_number, dimensions, caption, created_at, indexed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                image_id,
                source_id,
                content_hash,
                file_path,
                page_number,
                json.dumps(dimensions),
                caption,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()

    return {
        "id": image_id,
        "source_id": source_id,
        "content_hash": content_hash,
        "file_path": file_path,
        "page_number": page_number,
        "dimensions": dimensions,
        "caption": caption,
        "indexed": False,
    }


async def get_image_chunk(image_id: str) -> dict | None:
    """Get an image chunk by ID.

    Args:
        image_id: UUID of the image chunk

    Returns:
        Image chunk record as dict, or None if not found
    """
    import json

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM image_chunks WHERE id = ?",
            (image_id,),
        )
        row = await cursor.fetchone()

        if row:
            result = dict(row)
            result["dimensions"] = json.loads(result["dimensions"])
            result["indexed"] = bool(result["indexed"])
            return result
        return None


async def get_image_chunks_by_source(source_id: str) -> list[dict]:
    """Get all image chunks for a data source.

    Args:
        source_id: UUID of the data source

    Returns:
        List of image chunk records
    """
    import json

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM image_chunks WHERE source_id = ? ORDER BY page_number, created_at",
            (source_id,),
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            result = dict(row)
            result["dimensions"] = json.loads(result["dimensions"])
            result["indexed"] = bool(result["indexed"])
            results.append(result)
        return results


async def get_image_chunk_by_hash(content_hash: str) -> dict | None:
    """Get an image chunk by content hash (for deduplication).

    Args:
        content_hash: SHA-256 hash of image content

    Returns:
        Image chunk record if exists, None otherwise
    """
    import json

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM image_chunks WHERE content_hash = ?",
            (content_hash,),
        )
        row = await cursor.fetchone()

        if row:
            result = dict(row)
            result["dimensions"] = json.loads(result["dimensions"])
            result["indexed"] = bool(result["indexed"])
            return result
        return None


async def mark_image_indexed(image_id: str) -> bool:
    """Mark an image chunk as indexed in Vespa.

    Args:
        image_id: UUID of the image chunk

    Returns:
        True if updated, False if not found
    """
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE image_chunks SET indexed = 1 WHERE id = ?",
            (image_id,),
        )
        await db.commit()
        return cursor.rowcount > 0


async def delete_image_chunks_by_source(source_id: str) -> int:
    """Delete all image chunks for a data source.

    Args:
        source_id: UUID of the data source

    Returns:
        Number of deleted records
    """
    async with get_db() as db:
        cursor = await db.execute(
            "DELETE FROM image_chunks WHERE source_id = ?",
            (source_id,),
        )
        await db.commit()
        return cursor.rowcount


async def count_image_chunks_by_source(source_id: str) -> int:
    """Count image chunks for a data source.

    Args:
        source_id: UUID of the data source

    Returns:
        Number of image chunks
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) as count FROM image_chunks WHERE source_id = ?",
            (source_id,),
        )
        row = await cursor.fetchone()
        return row["count"] if row else 0
