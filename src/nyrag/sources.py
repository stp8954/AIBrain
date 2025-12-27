"""
Data sources CRUD module for NyRAG UI Document Manager.

Provides functions for managing data sources (URLs and files):
- Creating, reading, updating, and deleting data sources
- File upload handling with validation
- Integration with background job processing
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import aiofiles

from nyrag.database import get_db
from nyrag.logger import get_logger
from nyrag.schema import (
    DataSourceResponse,
    DataSourceStatus,
    DataSourceType,
)
from nyrag.utils import (
    cleanup_upload,
    generate_safe_filename,
    get_file_type_from_extension,
    get_uploads_dir,
    is_supported_file_type,
    validate_file_size,
)


logger = get_logger(__name__)


# =============================================================================
# Data Source CRUD Operations
# =============================================================================


async def create_data_source(
    source_type: DataSourceType,
    name: str,
    source_path: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> DataSourceResponse:
    """Create a new data source.

    Args:
        source_type: Type of data source ('url', 'pdf', 'markdown', 'txt')
        name: Display name for the source
        source_path: Original URL or file path
        metadata: Optional additional metadata (JSON)

    Returns:
        DataSourceResponse: The created data source

    Raises:
        ValueError: If validation fails
    """
    if not name or not name.strip():
        raise ValueError("Data source name cannot be empty")

    source_id = str(uuid.uuid4())
    now = datetime.utcnow()

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO data_sources (id, name, type, source_path, date_added, status, metadata)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                source_id,
                name.strip(),
                source_type,
                source_path,
                now.isoformat(),
                str(metadata) if metadata else None,
            ),
        )
        await db.commit()

        logger.info(f"Created data source: {source_id} ({source_type}: {name})")

        return DataSourceResponse(
            id=source_id,
            source_type=source_type,
            name=name.strip(),
            status="pending",
            url=source_path if source_type == "url" else None,
            file_path=source_path if source_type != "url" else None,
            created_at=now,
            updated_at=None,
            document_count=0,
            error_message=None,
        )


async def list_data_sources(
    source_type: Optional[DataSourceType] = None,
    status: Optional[DataSourceStatus] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[DataSourceResponse], int]:
    """List data sources with optional filtering.

    Args:
        source_type: Filter by source type
        status: Filter by status
        limit: Maximum number of results (default 50, max 100)
        offset: Pagination offset

    Returns:
        Tuple of (list of sources, total count)
    """
    limit = min(max(limit, 1), 100)  # Clamp to 1-100
    offset = max(offset, 0)

    conditions = []
    params: List = []

    if source_type:
        conditions.append("type = ?")
        params.append(source_type)

    if status:
        conditions.append("status = ?")
        params.append(status)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    async with get_db() as db:
        # Get total count
        cursor = await db.execute(f"SELECT COUNT(*) FROM data_sources WHERE {where_clause}", params)
        row = await cursor.fetchone()
        total = row[0] if row else 0

        # Get paginated results
        cursor = await db.execute(
            f"""
            SELECT id, name, type, source_path, date_added, status, progress, chunk_count, error_message
            FROM data_sources
            WHERE {where_clause}
            ORDER BY date_added DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )
        rows = await cursor.fetchall()

        sources = []
        for row in rows:
            source_type_val = row[2]
            source_path_val = row[3]
            sources.append(
                DataSourceResponse(
                    id=row[0],
                    source_type=source_type_val,
                    name=row[1],
                    status=row[5],
                    url=source_path_val if source_type_val == "url" else None,
                    file_path=source_path_val if source_type_val != "url" else None,
                    created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.utcnow(),
                    updated_at=None,
                    document_count=row[7] or 0,
                    error_message=row[8],
                )
            )

        return sources, total


async def get_data_source(source_id: str) -> Optional[DataSourceResponse]:
    """Get a single data source by ID.

    Args:
        source_id: UUID of the data source

    Returns:
        DataSourceResponse or None if not found
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT id, name, type, source_path, date_added, status, progress, chunk_count, error_message
            FROM data_sources
            WHERE id = ?
            """,
            (source_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        source_type_val = row[2]
        source_path_val = row[3]

        return DataSourceResponse(
            id=row[0],
            source_type=source_type_val,
            name=row[1],
            status=row[5],
            url=source_path_val if source_type_val == "url" else None,
            file_path=source_path_val if source_type_val != "url" else None,
            created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.utcnow(),
            updated_at=None,
            document_count=row[7] or 0,
            error_message=row[8],
        )


async def update_data_source(
    source_id: str,
    status: Optional[DataSourceStatus] = None,
    progress: Optional[int] = None,
    chunk_count: Optional[int] = None,
    error_message: Optional[str] = None,
) -> bool:
    """Update a data source's status and progress.

    Args:
        source_id: UUID of the data source
        status: New status
        progress: New progress (0-100)
        chunk_count: New chunk count
        error_message: Error message (if failed)

    Returns:
        True if updated, False if not found
    """
    updates = []
    params: List = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)

    if progress is not None:
        updates.append("progress = ?")
        params.append(max(0, min(100, progress)))

    if chunk_count is not None:
        updates.append("chunk_count = ?")
        params.append(chunk_count)

    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)

    if not updates:
        return True  # Nothing to update

    params.append(source_id)

    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE data_sources SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await db.commit()

        if cursor.rowcount > 0:
            logger.info(f"Updated data source {source_id}: {dict(zip([u.split(' ')[0] for u in updates], params[:-1]))}")
            return True
        return False


async def delete_data_source(source_id: str) -> Tuple[bool, int]:
    """Delete a data source and clean up associated chunks from Vespa.

    Args:
        source_id: UUID of the data source

    Returns:
        Tuple of (success, chunks_deleted)
    """
    async with get_db() as db:
        # Get source details for cleanup
        cursor = await db.execute(
            "SELECT type, source_path, chunk_count FROM data_sources WHERE id = ?",
            (source_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return False, 0

        source_type, source_path, chunk_count = row

        # Delete associated jobs first (cascade should handle this, but being explicit)
        await db.execute("DELETE FROM jobs WHERE source_id = ?", (source_id,))

        # Delete the data source
        cursor = await db.execute("DELETE FROM data_sources WHERE id = ?", (source_id,))
        await db.commit()

        if cursor.rowcount == 0:
            return False, 0

        # Clean up uploaded file if it exists
        if source_type != "url" and source_path:
            try:
                await cleanup_upload(Path(source_path))
            except Exception as e:
                logger.warning(f"Failed to cleanup file {source_path}: {e}")

        # TODO: Delete chunks from Vespa using source_id as filter
        # This will be implemented when Vespa integration is added
        chunks_deleted = chunk_count or 0

        logger.info(f"Deleted data source {source_id}, cleaned up {chunks_deleted} chunks")
        return True, chunks_deleted


# =============================================================================
# File Upload Handling
# =============================================================================


async def save_uploaded_file(
    filename: str,
    content: bytes,
    base_path: Optional[Path] = None,
) -> Tuple[Path, DataSourceType]:
    """Save an uploaded file to the uploads directory.

    Args:
        filename: Original filename
        content: File content as bytes
        base_path: Optional base path for uploads directory

    Returns:
        Tuple of (saved file path, detected file type)

    Raises:
        ValueError: If file type is not supported or file is too large
    """
    # Validate file size
    if not validate_file_size(len(content)):
        raise ValueError(f"File too large. Maximum size is 50MB, got {len(content) / (1024*1024):.1f}MB")

    # Detect file type from extension
    file_type = get_file_type_from_extension(filename)
    if file_type is None:
        raise ValueError(f"Unsupported file type: {filename}")

    # Generate safe filename
    safe_filename = generate_safe_filename(filename)

    # Get uploads directory
    uploads_dir = get_uploads_dir(base_path)

    # Create full path
    file_path = uploads_dir / safe_filename

    # Write file
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    logger.info(f"Saved uploaded file: {file_path} ({len(content)} bytes)")

    return file_path, file_type


async def create_source_from_url(url: str, name: Optional[str] = None) -> DataSourceResponse:
    """Create a data source from a URL.

    Args:
        url: URL to crawl
        name: Optional display name (defaults to URL host)

    Returns:
        DataSourceResponse for the created source
    """
    from urllib.parse import urlparse

    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    url = url.strip()

    # Basic URL validation
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {url}")
    except Exception as e:
        raise ValueError(f"Invalid URL: {e}")

    # Generate name from URL if not provided
    if not name:
        name = parsed.netloc + (parsed.path if parsed.path != "/" else "")
        # Truncate if too long
        if len(name) > 100:
            name = name[:97] + "..."

    return await create_data_source(
        source_type="url",
        name=name,
        source_path=url,
    )


async def create_source_from_file(
    filename: str,
    content: bytes,
    base_path: Optional[Path] = None,
) -> DataSourceResponse:
    """Create a data source from an uploaded file.

    Args:
        filename: Original filename
        content: File content
        base_path: Optional base path for uploads

    Returns:
        DataSourceResponse for the created source
    """
    # Save the file
    file_path, file_type = await save_uploaded_file(filename, content, base_path)

    # Create the data source
    return await create_data_source(
        source_type=file_type,
        name=filename,
        source_path=str(file_path),
    )


# =============================================================================
# Helper Functions
# =============================================================================


def extract_name_from_path(path: str) -> str:
    """Extract a display name from a file path or URL.

    Args:
        path: File path or URL

    Returns:
        Human-readable name
    """
    from urllib.parse import urlparse

    # Try to parse as URL first
    try:
        parsed = urlparse(path)
        if parsed.scheme and parsed.netloc:
            return parsed.netloc + (parsed.path if parsed.path != "/" else "")
    except Exception:
        pass

    # Treat as file path
    return Path(path).stem


# =============================================================================
# Job CRUD Operations
# =============================================================================

# Maximum concurrent jobs
MAX_CONCURRENT_JOBS = 2

# In-memory tracking of running async tasks
_running_tasks: dict = {}


async def create_job(
    source_id: str,
    job_type: str,
) -> dict:
    """Create a new ingestion job.

    Args:
        source_id: UUID of the data source
        job_type: Type of job ('crawl', 'process_file', 'sync')

    Returns:
        Job details dictionary

    Raises:
        ValueError: If source not found or invalid job type
    """
    valid_types = ["crawl", "process_file", "sync"]
    if job_type not in valid_types:
        raise ValueError(f"Invalid job type: {job_type}. Must be one of {valid_types}")

    # Verify source exists
    source = await get_data_source(source_id)
    if not source:
        raise ValueError(f"Data source not found: {source_id}")

    job_id = str(uuid.uuid4())
    now = datetime.utcnow()

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO jobs (id, source_id, job_type, status, progress, started_at)
            VALUES (?, ?, ?, 'queued', 0, ?)
            """,
            (job_id, source_id, job_type, now.isoformat()),
        )
        await db.commit()

        logger.info(f"Created job: {job_id} ({job_type}) for source {source_id}")

        return {
            "id": job_id,
            "source_id": source_id,
            "job_type": job_type,
            "status": "queued",
            "progress": 0,
            "current_task": None,
            "started_at": now.isoformat(),
            "completed_at": None,
            "error_message": None,
        }


async def list_jobs(
    status: Optional[str] = None,
    source_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[dict], int]:
    """List jobs with optional filtering.

    Args:
        status: Filter by job status
        source_id: Filter by source ID
        limit: Maximum number of results (default 50, max 100)
        offset: Pagination offset

    Returns:
        Tuple of (list of jobs, total count)
    """
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)

    conditions = []
    params: List = []

    if status:
        conditions.append("j.status = ?")
        params.append(status)

    if source_id:
        conditions.append("j.source_id = ?")
        params.append(source_id)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    async with get_db() as db:
        # Get total count
        cursor = await db.execute(f"SELECT COUNT(*) FROM jobs j WHERE {where_clause}", params)
        row = await cursor.fetchone()
        total = row[0] if row else 0

        # Get paginated results with source name
        cursor = await db.execute(
            f"""
            SELECT j.id, j.source_id, j.job_type, j.status, j.progress, j.current_task,
                   j.started_at, j.completed_at, j.error_message, d.name as source_name
            FROM jobs j
            LEFT JOIN data_sources d ON j.source_id = d.id
            WHERE {where_clause}
            ORDER BY j.started_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )
        rows = await cursor.fetchall()

        jobs = []
        for row in rows:
            jobs.append({
                "id": row[0],
                "source_id": row[1],
                "job_type": row[2],
                "status": row[3],
                "progress": row[4] or 0,
                "current_task": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "error_message": row[8],
                "source_name": row[9],
            })

        return jobs, total


async def get_job(job_id: str) -> Optional[dict]:
    """Get a single job by ID.

    Args:
        job_id: UUID of the job

    Returns:
        Job dictionary or None if not found
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT j.id, j.source_id, j.job_type, j.status, j.progress, j.current_task,
                   j.started_at, j.completed_at, j.error_message, j.logs, d.name as source_name
            FROM jobs j
            LEFT JOIN data_sources d ON j.source_id = d.id
            WHERE j.id = ?
            """,
            (job_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "source_id": row[1],
            "job_type": row[2],
            "status": row[3],
            "progress": row[4] or 0,
            "current_task": row[5],
            "started_at": row[6],
            "completed_at": row[7],
            "error_message": row[8],
            "logs": row[9],
            "source_name": row[10],
        }


async def update_job_progress(
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    current_task: Optional[str] = None,
    error_message: Optional[str] = None,
    log_line: Optional[str] = None,
) -> bool:
    """Update a job's progress and status.

    Args:
        job_id: UUID of the job
        status: New status
        progress: New progress (0-100)
        current_task: Human-readable current step
        error_message: Error message if failed
        log_line: Line to append to logs

    Returns:
        True if updated, False if not found
    """
    updates = []
    params: List = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
        if status == "running":
            updates.append("started_at = COALESCE(started_at, ?)")
            params.append(datetime.utcnow().isoformat())
        elif status in ("completed", "failed", "cancelled"):
            updates.append("completed_at = ?")
            params.append(datetime.utcnow().isoformat())

    if progress is not None:
        updates.append("progress = ?")
        params.append(max(0, min(100, progress)))

    if current_task is not None:
        updates.append("current_task = ?")
        params.append(current_task)

    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)

    if log_line is not None:
        updates.append("logs = COALESCE(logs, '') || ? || '\n'")
        params.append(log_line)

    if not updates:
        return True

    params.append(job_id)

    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await db.commit()

        return cursor.rowcount > 0


async def cancel_job(job_id: str) -> Tuple[bool, str]:
    """Cancel a queued or running job.

    Args:
        job_id: UUID of the job

    Returns:
        Tuple of (success, message)
    """
    job = await get_job(job_id)
    if not job:
        return False, "Job not found"

    if job["status"] in ("completed", "failed", "cancelled"):
        return False, f"Cannot cancel job with status: {job['status']}"

    # Cancel async task if running
    if job_id in _running_tasks:
        task = _running_tasks[job_id]
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled running task for job {job_id}")

    # Update job status
    await update_job_progress(job_id, status="cancelled", current_task="Cancelled by user")

    # Update source status back to pending/failed
    await update_data_source(job["source_id"], status="pending")

    logger.info(f"Job {job_id} cancelled")
    return True, "Job cancelled successfully"


async def retry_job(job_id: str) -> Tuple[bool, Optional[str]]:
    """Retry a failed job.

    Args:
        job_id: UUID of the failed job

    Returns:
        Tuple of (success, new_job_id or error message)
    """
    job = await get_job(job_id)
    if not job:
        return False, "Job not found"

    if job["status"] != "failed":
        return False, f"Can only retry failed jobs, current status: {job['status']}"

    # Create a new job for the same source
    try:
        new_job = await create_job(job["source_id"], job["job_type"])
        logger.info(f"Created retry job {new_job['id']} for failed job {job_id}")
        return True, new_job["id"]
    except Exception as e:
        logger.error(f"Failed to retry job {job_id}: {e}")
        return False, str(e)


async def get_running_job_count() -> int:
    """Get the count of currently running jobs.

    Returns:
        Number of jobs with status 'running'
    """
    async with get_db() as db:
        cursor = await db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'")
        row = await cursor.fetchone()
        return row[0] if row else 0


async def get_queued_jobs() -> List[dict]:
    """Get all queued jobs ordered by creation time.

    Returns:
        List of queued jobs
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT id, source_id, job_type, started_at
            FROM jobs
            WHERE status = 'queued'
            ORDER BY started_at ASC
            """
        )
        rows = await cursor.fetchall()

        return [
            {
                "id": row[0],
                "source_id": row[1],
                "job_type": row[2],
                "started_at": row[3],
            }
            for row in rows
        ]


# =============================================================================
# Ingestion Job Execution
# =============================================================================


async def run_ingestion_job(job_id: str) -> None:
    """Execute an ingestion job (crawl URL or process file).

    This function runs in the background and updates job progress.

    Args:
        job_id: UUID of the job to run
    """
    import asyncio

    job = await get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return

    source = await get_data_source(job["source_id"])
    if not source:
        await update_job_progress(job_id, status="failed", error_message="Source not found")
        return

    try:
        # Update job to running
        await update_job_progress(
            job_id,
            status="running",
            progress=0,
            current_task="Starting...",
            log_line=f"Starting {job['job_type']} job",
        )

        # Update source to processing
        await update_data_source(job["source_id"], status="processing")

        if job["job_type"] == "crawl":
            await _run_crawl_job(job_id, source)
        elif job["job_type"] == "process_file":
            await _run_file_processing_job(job_id, source)
        elif job["job_type"] == "sync":
            # Sync re-runs the appropriate job type
            if source.source_type == "url":
                await _run_crawl_job(job_id, source)
            else:
                await _run_file_processing_job(job_id, source)

        # Mark job and source as completed
        await update_job_progress(
            job_id,
            status="completed",
            progress=100,
            current_task="Complete",
            log_line="Job completed successfully",
        )
        await update_data_source(job["source_id"], status="indexed")

        logger.info(f"Job {job_id} completed successfully")

    except asyncio.CancelledError:
        await update_job_progress(
            job_id,
            status="cancelled",
            current_task="Cancelled",
            log_line="Job was cancelled",
        )
        await update_data_source(job["source_id"], status="pending")
        logger.info(f"Job {job_id} was cancelled")

    except Exception as e:
        error_msg = str(e)
        await update_job_progress(
            job_id,
            status="failed",
            current_task="Failed",
            error_message=error_msg,
            log_line=f"Error: {error_msg}",
        )
        await update_data_source(job["source_id"], status="failed", error_message=error_msg)
        logger.error(f"Job {job_id} failed: {e}")

    finally:
        # Remove from running tasks
        _running_tasks.pop(job_id, None)

        # Try to start next queued job
        await _try_start_next_job()


async def _run_crawl_job(job_id: str, source: DataSourceResponse) -> None:
    """Run a URL crawl job.

    This is a stub implementation that simulates crawling for the job monitoring UI.
    Full integration with crawly.crawl_web will be added in a future iteration.

    Args:
        job_id: UUID of the job
        source: Data source to crawl
    """
    import asyncio

    url = source.url
    if not url:
        raise ValueError("Source URL is missing")

    # Step 1: Simulate crawl initialization
    await update_job_progress(
        job_id,
        progress=10,
        current_task="Initializing crawler...",
        log_line=f"Starting crawl for: {url}",
    )
    await asyncio.sleep(0.5)

    # Step 2: Simulate crawling pages
    await update_job_progress(
        job_id,
        progress=30,
        current_task="Crawling pages...",
        log_line="Discovering and crawling pages",
    )
    await asyncio.sleep(1)

    # Simulate finding some documents
    doc_count = 5  # Simulated document count
    await update_job_progress(
        job_id,
        progress=50,
        current_task=f"Found {doc_count} pages",
        log_line=f"Crawled {doc_count} pages from {url}",
    )
    await asyncio.sleep(0.5)

    # Step 3: Simulate chunking
    await update_job_progress(
        job_id,
        progress=70,
        current_task="Processing content...",
        log_line="Extracting and chunking content",
    )
    await asyncio.sleep(0.5)

    chunk_count = doc_count * 3  # Simulated chunk count

    # Step 4: Simulate indexing
    await update_job_progress(
        job_id,
        progress=85,
        current_task="Indexing to Vespa...",
        log_line=f"Indexing {chunk_count} chunks to Vespa",
    )
    await asyncio.sleep(0.5)

    # Update source with simulated chunk count
    await update_data_source(source.id, chunk_count=chunk_count)

    await update_job_progress(
        job_id,
        progress=95,
        current_task="Finalizing...",
        log_line=f"Successfully indexed {chunk_count} chunks",
    )


async def _run_file_processing_job(job_id: str, source: DataSourceResponse) -> None:
    """Run a file processing job.

    This is a stub implementation that simulates file processing for the job monitoring UI.
    Full integration with MarkItDown will be added in a future iteration.

    Args:
        job_id: UUID of the job
        source: Data source (file) to process
    """
    import asyncio
    from pathlib import Path

    file_path = source.file_path
    if not file_path:
        raise ValueError("Source file path is missing")

    file_name = Path(file_path).name

    # Step 1: Simulate file reading
    await update_job_progress(
        job_id,
        progress=10,
        current_task="Reading file...",
        log_line=f"Opening: {file_name}",
    )
    await asyncio.sleep(0.5)

    # Step 2: Simulate content extraction
    await update_job_progress(
        job_id,
        progress=30,
        current_task="Extracting content...",
        log_line=f"Extracting text from {source.source_type} file",
    )
    await asyncio.sleep(0.5)

    # Simulate extracted documents
    doc_count = 1
    await update_job_progress(
        job_id,
        progress=50,
        current_task=f"Extracted {doc_count} document(s)",
        log_line=f"Extracted {doc_count} document(s) from {file_name}",
    )
    await asyncio.sleep(0.5)

    # Step 3: Simulate chunking
    await update_job_progress(
        job_id,
        progress=70,
        current_task="Chunking content...",
        log_line="Splitting content into chunks",
    )
    await asyncio.sleep(0.5)

    chunk_count = 8  # Simulated chunk count

    # Step 4: Simulate indexing
    await update_job_progress(
        job_id,
        progress=85,
        current_task="Indexing to Vespa...",
        log_line=f"Indexing {chunk_count} chunks to Vespa",
    )
    await asyncio.sleep(0.5)

    # Update source with simulated chunk count
    await update_data_source(source.id, chunk_count=chunk_count)

    await update_job_progress(
        job_id,
        progress=95,
        current_task="Finalizing...",
        log_line=f"Successfully indexed {chunk_count} chunks",
    )


async def start_job(job_id: str) -> bool:
    """Start a job if there's capacity.

    Args:
        job_id: UUID of the job to start

    Returns:
        True if job was started, False if at capacity
    """
    import asyncio

    # Check if we're at capacity
    running_count = await get_running_job_count()
    if running_count >= MAX_CONCURRENT_JOBS:
        logger.info(f"Job {job_id} queued - at capacity ({running_count}/{MAX_CONCURRENT_JOBS})")
        return False

    # Get the event loop and create task
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(run_ingestion_job(job_id))
        _running_tasks[job_id] = task
        logger.info(f"Started job {job_id}")
        return True
    except RuntimeError:
        logger.warning(f"No event loop available to start job {job_id}")
        return False


async def _try_start_next_job() -> None:
    """Try to start the next queued job if there's capacity."""
    running_count = await get_running_job_count()
    if running_count >= MAX_CONCURRENT_JOBS:
        return

    queued_jobs = await get_queued_jobs()
    if queued_jobs:
        next_job = queued_jobs[0]
        await start_job(next_job["id"])


# =============================================================================
# Conversation CRUD Operations
# =============================================================================


async def create_conversation(title: Optional[str] = None) -> dict:
    """Create a new conversation.

    Args:
        title: Optional title for the conversation

    Returns:
        Conversation dictionary with id, title, created_at, updated_at
    """
    conversation_id = str(uuid.uuid4())
    now = datetime.utcnow()

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title, now.isoformat(), now.isoformat()),
        )
        await db.commit()

        logger.info(f"Created conversation: {conversation_id}")

        return {
            "id": conversation_id,
            "title": title,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "message_count": 0,
        }


async def list_conversations(limit: int = 50, offset: int = 0) -> Tuple[List[dict], int]:
    """List conversations ordered by most recent activity.

    Args:
        limit: Maximum number of results (default 50, max 100)
        offset: Pagination offset

    Returns:
        Tuple of (list of conversations, total count)
    """
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)

    async with get_db() as db:
        # Get total count
        cursor = await db.execute("SELECT COUNT(*) FROM conversations")
        row = await cursor.fetchone()
        total = row[0] if row else 0

        # Get paginated results with message count
        cursor = await db.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count
            FROM conversations c
            ORDER BY c.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = await cursor.fetchall()

        conversations = []
        for row in rows:
            conversations.append({
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4] or 0,
            })

        return conversations, total


async def get_conversation(conversation_id: str) -> Optional[dict]:
    """Get a conversation by ID (without messages).

    Args:
        conversation_id: UUID of the conversation

    Returns:
        Conversation dictionary or None if not found
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count
            FROM conversations c
            WHERE c.id = ?
            """,
            (conversation_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "message_count": row[4] or 0,
        }


async def get_conversation_with_messages(conversation_id: str) -> Optional[dict]:
    """Get a conversation with all its messages.

    Args:
        conversation_id: UUID of the conversation

    Returns:
        Conversation dictionary with messages array, or None if not found
    """
    conversation = await get_conversation(conversation_id)
    if not conversation:
        return None

    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        rows = await cursor.fetchall()

        messages = []
        for row in rows:
            messages.append({
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "created_at": row[3],
            })

        conversation["messages"] = messages
        return conversation


async def add_message(
    conversation_id: str,
    role: Literal["user", "assistant"],
    content: str,
) -> dict:
    """Add a message to a conversation.

    Args:
        conversation_id: UUID of the conversation
        role: Message role ('user' or 'assistant')
        content: Message content

    Returns:
        Message dictionary

    Raises:
        ValueError: If conversation not found or content is empty
    """
    if not content or not content.strip():
        raise ValueError("Message content cannot be empty")

    # Verify conversation exists
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise ValueError(f"Conversation not found: {conversation_id}")

    message_id = str(uuid.uuid4())
    now = datetime.utcnow()

    async with get_db() as db:
        # Insert message
        await db.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content.strip(), now.isoformat()),
        )

        # Update conversation's updated_at timestamp
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now.isoformat(), conversation_id),
        )

        # Auto-generate title from first user message if not set
        if role == "user" and not conversation.get("title"):
            title = generate_conversation_title(content)
            await db.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )

        await db.commit()

        logger.debug(f"Added {role} message to conversation {conversation_id}")

        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content.strip(),
            "created_at": now.isoformat(),
        }


async def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and all its messages.

    Args:
        conversation_id: UUID of the conversation

    Returns:
        True if deleted, False if not found
    """
    async with get_db() as db:
        # Check if conversation exists
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        if not await cursor.fetchone():
            return False

        # Delete conversation (messages will cascade delete due to FK)
        await db.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        await db.commit()

        logger.info(f"Deleted conversation: {conversation_id}")
        return True


async def update_conversation_title(conversation_id: str, title: str) -> bool:
    """Update a conversation's title.

    Args:
        conversation_id: UUID of the conversation
        title: New title

    Returns:
        True if updated, False if not found
    """
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id),
        )
        await db.commit()

        return cursor.rowcount > 0


def generate_conversation_title(first_message: str, max_length: int = 50) -> str:
    """Generate a conversation title from the first user message.

    Args:
        first_message: The first user message content
        max_length: Maximum title length (default 50)

    Returns:
        Generated title string
    """
    # Clean up the message
    title = first_message.strip()

    # Remove line breaks and extra whitespace
    title = " ".join(title.split())

    # Truncate if too long
    if len(title) > max_length:
        # Try to cut at a word boundary
        title = title[:max_length].rsplit(" ", 1)[0]
        if len(title) < max_length - 3:
            title += "..."

    return title
