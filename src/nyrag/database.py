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

    logger.info("Database schema initialized successfully")


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
