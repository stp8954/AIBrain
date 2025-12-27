import asyncio
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from nyrag.blog import BlogPost, generate_blog_task, get_blog, get_blog_path
from nyrag.config import Config
from nyrag.jobs import JobStatus, get_job_queue
from nyrag.logger import get_logger
from nyrag.notes import Note, get_note, save_image, save_note, search_notes
from nyrag.utils import (
    DEFAULT_EMBEDDING_MODEL,
    get_vespa_tls_config,
    is_vespa_cloud,
    make_vespa_client,
    resolve_vespa_cloud_mtls_paths,
    resolve_vespa_port,
)


DEFAULT_ENDPOINT = "http://localhost:8080"
DEFAULT_RANKING = "default"
DEFAULT_SUMMARY = "top_k_chunks"


class SearchRequest(BaseModel):
    query: str = Field(..., description="User query string")
    hits: int = Field(10, description="Number of Vespa hits to return")
    k: int = Field(3, description="Top-k chunks to keep per hit")
    ranking: Optional[str] = Field(None, description="Ranking profile to use (defaults to schema default)")
    summary: Optional[str] = Field(None, description="Document summary to request (defaults to top_k_chunks)")


def _resolve_mtls_paths(vespa_url: str, project_folder: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    cert_env = (os.getenv("VESPA_CLIENT_CERT") or "").strip() or None
    key_env = (os.getenv("VESPA_CLIENT_KEY") or "").strip() or None

    if not is_vespa_cloud(vespa_url):
        return cert_env, key_env

    if cert_env or key_env:
        if not (cert_env and key_env):
            raise RuntimeError("Vespa Cloud requires both VESPA_CLIENT_CERT and VESPA_CLIENT_KEY.")
        return cert_env, key_env

    if not project_folder:
        raise RuntimeError(
            "Vespa Cloud mTLS credentials not found. "
            "Export VESPA_CLIENT_CERT and VESPA_CLIENT_KEY with the paths to these files."
        )

    cert_path, key_path = resolve_vespa_cloud_mtls_paths(project_folder)
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    raise RuntimeError(
        "Vespa Cloud mTLS credentials not found at "
        f"{cert_path} and {key_path}. "
        "Export VESPA_CLIENT_CERT and VESPA_CLIENT_KEY with the paths to these files."
    )


def _load_settings() -> Dict[str, Any]:
    """Load schema, model, and Vespa connection from env or YAML config."""
    config_path = os.getenv("NYRAG_CONFIG")
    vespa_url = (os.getenv("VESPA_URL") or "").strip() or "http://localhost"
    vespa_port = resolve_vespa_port(vespa_url)

    if config_path and Path(config_path).exists():
        cfg = Config.from_yaml(config_path)
        rag_params = cfg.rag_params or {}
        return {
            "app_package_name": cfg.get_app_package_name(),
            "schema_name": cfg.get_schema_name(),
            "embedding_model": rag_params.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            "vespa_url": vespa_url,
            "vespa_port": vespa_port,
        }

    return {
        "app_package_name": None,
        "schema_name": os.getenv("VESPA_SCHEMA", "nyragwebrag"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "vespa_url": vespa_url,
        "vespa_port": vespa_port,
    }


settings = _load_settings()
logger = get_logger("api")
app = FastAPI(title="nyrag API", version="0.1.0")
model = SentenceTransformer(settings["embedding_model"])

# Get mTLS credentials (with Vespa Cloud fallback)
_cert, _key = _resolve_mtls_paths(settings["vespa_url"], settings.get("app_package_name"))
_, _, _ca, _verify = get_vespa_tls_config()

vespa_app = make_vespa_client(
    settings["vespa_url"],
    settings["vespa_port"],
    _cert,
    _key,
    _ca,
    _verify,
)

base_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

# Mount assets directory for serving note images
# Assets are stored in output/<project>/assets/
_assets_base = Path("output")
if not _assets_base.exists():
    _assets_base.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def init_database_on_startup():
    """Initialize the SQLite database on startup."""
    from nyrag.database import init_database

    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")


@app.on_event("startup")
async def mount_assets_on_startup():
    """Mount assets directories dynamically on startup."""
    config_path = os.getenv("NYRAG_CONFIG")
    if config_path and Path(config_path).exists():
        from nyrag.config import Config

        cfg = Config.from_yaml(config_path)
        assets_dir = cfg.get_output_path() / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
            logger.info(f"Mounted assets directory: {assets_dir}")


def _get_config_for_notes() -> "Config":
    """Load config for notes operations."""
    config_path = os.getenv("NYRAG_CONFIG")
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    # Create minimal config for notes if no config file exists
    return Config(
        name="default",
        mode="web",
        start_loc="http://localhost",
    )


def _deep_find_numeric_field(obj: Any, key: str) -> Optional[float]:
    if isinstance(obj, dict):
        if key in obj:
            value = obj.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return None
        for v in obj.values():
            found = _deep_find_numeric_field(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _deep_find_numeric_field(item, key)
            if found is not None:
                return found
    return None


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    """Return simple corpus statistics from Vespa (documents and chunks)."""
    doc_count: Optional[int] = None
    chunk_count: Optional[int] = None

    try:
        res = vespa_app.query(
            body={"yql": "select * from sources * where true", "hits": 0},
            schema=settings["schema_name"],
        )
        total = res.json.get("root", {}).get("fields", {}).get("totalCount")
        if isinstance(total, int):
            doc_count = total
        elif isinstance(total, str) and total.isdigit():
            doc_count = int(total)
    except Exception as e:
        logger.warning(f"Failed to fetch Vespa doc count: {e}")

    try:
        # Requires schema field `chunk_count` (added in this repo); if absent, this will likely return null.
        yql = "select * from sources * where true | " "all(group(1) each(output(count(), sum(chunk_count))))"
        res = vespa_app.query(
            body={"yql": yql, "hits": 0},
            schema=settings["schema_name"],
        )
        sum_value = _deep_find_numeric_field(res.json, "sum(chunk_count)")
        if sum_value is None:
            sum_value = _deep_find_numeric_field(res.json, "sum(chunk_count())")
        if sum_value is not None:
            chunk_count = int(sum_value)
    except Exception as e:
        logger.warning(f"Failed to fetch Vespa chunk count: {e}")

    return {
        "schema": settings["schema_name"],
        "documents": doc_count,
        "chunks": chunk_count,
    }


@app.get("/api/status")
async def get_system_status() -> Dict[str, Any]:
    """Get system status including Vespa and database availability.

    Returns comprehensive status for the UI to display system health.
    """
    from nyrag.database import check_database_available, get_database_stats
    from nyrag.schema import SystemStatusResponse

    # Check Vespa availability
    vespa_available = False
    vespa_docs = 0
    vespa_chunks = 0

    try:
        res = vespa_app.query(
            body={"yql": "select * from sources * where true", "hits": 0},
            schema=settings["schema_name"],
        )
        total = res.json.get("root", {}).get("fields", {}).get("totalCount")
        if isinstance(total, int):
            vespa_docs = total
            vespa_available = True
        elif isinstance(total, str) and total.isdigit():
            vespa_docs = int(total)
            vespa_available = True

        # Get chunk count
        yql = "select * from sources * where true | all(group(1) each(output(sum(chunk_count))))"
        res = vespa_app.query(body={"yql": yql, "hits": 0}, schema=settings["schema_name"])
        sum_value = _deep_find_numeric_field(res.json, "sum(chunk_count)")
        if sum_value is not None:
            vespa_chunks = int(sum_value)
    except Exception as e:
        logger.debug(f"Vespa not available: {e}")
        vespa_available = False

    # Check database availability and get stats
    db_available = await check_database_available()
    db_stats = await get_database_stats() if db_available else {}

    # Get running and queued jobs count
    running_jobs = db_stats.get("running_jobs", 0)
    queued_jobs = db_stats.get("jobs", {}).get("by_status", {}).get("queued", 0)

    # Get counts for data sources and conversations
    total_data_sources = db_stats.get("sources", {}).get("total", 0)
    total_conversations = db_stats.get("conversations", 0)

    return SystemStatusResponse(
        vespa_available=vespa_available,
        database_available=db_available,
        running_jobs=running_jobs,
        queued_jobs=queued_jobs,
        total_data_sources=total_data_sources,
        total_conversations=total_conversations,
    ).model_dump()


# =============================================================================
# Data Sources Endpoints
# =============================================================================


@app.get("/api/sources")
async def list_sources(
    type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List all data sources with optional filtering.

    Args:
        type: Filter by source type (url, pdf, markdown, txt)
        status: Filter by status (pending, processing, indexed, failed)
        limit: Maximum number of results (default 50, max 100)
        offset: Pagination offset

    Returns:
        Dictionary with items list and total count
    """
    from nyrag.sources import list_data_sources

    sources, total = await list_data_sources(
        source_type=type,
        status=status,
        limit=limit,
        offset=offset,
    )

    return {
        "items": [s.model_dump() for s in sources],
        "total": total,
    }


@app.get("/api/sources/{source_id}")
async def get_source(source_id: str) -> Dict[str, Any]:
    """Get details of a specific data source.

    Args:
        source_id: UUID of the data source

    Returns:
        Data source details

    Raises:
        HTTPException: 404 if source not found
    """
    from nyrag.sources import get_data_source

    source = await get_data_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Data source not found: {source_id}")

    return source.model_dump()


@app.post("/api/sources/url", status_code=201)
async def add_url_source(request: Request) -> Dict[str, Any]:
    """Add a URL for crawling.

    Request body:
        url: URL to crawl
        name: Optional display name

    Returns:
        Created source and job information

    Raises:
        HTTPException: 400 for invalid URL, 503 if Vespa unavailable
    """
    from nyrag.sources import create_job, create_source_from_url, start_job

    body = await request.json()
    url = body.get("url")
    name = body.get("name")

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        source = await create_source_from_url(url, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create and start a crawl job for the new source
    try:
        job = await create_job(source.id, "crawl")
        job_id = job["id"]
        # Start the job in the background
        asyncio.create_task(start_job(job_id))
    except Exception as e:
        logger.warning(f"Failed to create/start job for source {source.id}: {e}")
        job_id = None

    return {
        "source": source.model_dump(),
        "job_id": job_id,
        "message": "Source created successfully",
    }


@app.post("/api/sources/files", status_code=201)
async def upload_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Upload files for processing.

    Args:
        files: List of files to upload (max 50MB each)

    Returns:
        List of created sources with job information

    Raises:
        HTTPException: 400 for invalid files, 413 for files too large
    """
    from nyrag.sources import create_job, create_source_from_file, start_job

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    errors = []

    for file in files:
        try:
            content = await file.read()
            source = await create_source_from_file(file.filename or "unknown", content)

            # Create and start a processing job for the file
            try:
                job = await create_job(source.id, "process_file")
                job_id = job["id"]
                # Start the job in the background
                asyncio.create_task(start_job(job_id))
            except Exception as e:
                logger.warning(f"Failed to create/start job for source {source.id}: {e}")
                job_id = None

            results.append({
                "source": source.model_dump(),
                "job_id": job_id,
            })
        except ValueError as e:
            errors.append({
                "filename": file.filename,
                "error": str(e),
            })

    if errors and not results:
        raise HTTPException(status_code=400, detail={"message": "All files failed", "errors": errors})

    return {
        "sources": results,
        "errors": errors if errors else None,
        "message": f"Created {len(results)} source(s)",
    }


@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: str) -> Dict[str, Any]:
    """Delete a data source and its associated chunks.

    Args:
        source_id: UUID of the data source

    Returns:
        Deletion confirmation with chunks deleted count

    Raises:
        HTTPException: 404 if source not found
    """
    from nyrag.sources import delete_data_source

    success, chunks_deleted = await delete_data_source(source_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Data source not found: {source_id}")

    return {
        "message": "Source deleted successfully",
        "chunks_deleted": chunks_deleted,
    }


# =============================================================================
# Jobs API Endpoints
# =============================================================================


@app.get("/api/jobs")
async def list_jobs_endpoint(
    status: Optional[str] = None,
    source_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List background jobs with optional filtering.

    Args:
        status: Filter by job status (queued, running, completed, failed, cancelled)
        source_id: Filter by data source ID
        limit: Maximum number of results (default 50, max 100)
        offset: Pagination offset

    Returns:
        Dict with 'jobs' list and 'total' count
    """
    from nyrag.sources import list_jobs

    jobs, total = await list_jobs(
        status=status,
        source_id=source_id,
        limit=limit,
        offset=offset,
    )

    return {
        "jobs": jobs,
        "total": total,
        "limit": min(max(limit, 1), 100),
        "offset": max(offset, 0),
    }


@app.get("/api/jobs/{job_id}")
async def get_job_endpoint(job_id: str) -> Dict[str, Any]:
    """Get details of a specific job.

    Args:
        job_id: UUID of the job

    Returns:
        Job details including logs

    Raises:
        HTTPException: 404 if job not found
    """
    from nyrag.sources import get_job

    job = await get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return job


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str) -> Dict[str, Any]:
    """Cancel a queued or running job.

    Args:
        job_id: UUID of the job

    Returns:
        Cancellation status message

    Raises:
        HTTPException: 404 if job not found, 400 if cannot be cancelled
    """
    from nyrag.sources import cancel_job

    success, message = await cancel_job(job_id)

    if not success:
        if "not found" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=400, detail=message)

    return {"message": message, "job_id": job_id}


@app.post("/api/jobs/{job_id}/retry")
async def retry_job_endpoint(job_id: str) -> Dict[str, Any]:
    """Retry a failed job.

    Args:
        job_id: UUID of the failed job

    Returns:
        New job details

    Raises:
        HTTPException: 404 if job not found, 400 if cannot be retried
    """
    from nyrag.sources import get_job, retry_job, start_job

    success, result = await retry_job(job_id)

    if not success:
        if "not found" in result.lower():
            raise HTTPException(status_code=404, detail=result)
        raise HTTPException(status_code=400, detail=result)

    # Try to start the new job
    new_job_id = result
    await start_job(new_job_id)

    # Return the new job details
    new_job = await get_job(new_job_id)
    return {
        "message": "Job retry initiated",
        "new_job": new_job,
    }


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str) -> StreamingResponse:
    """Stream real-time job progress updates via SSE.

    Args:
        job_id: UUID of the job

    Returns:
        Server-Sent Events stream with progress updates
    """
    import asyncio

    from nyrag.sources import get_job

    async def event_generator():
        """Generate SSE events for job progress."""
        last_progress = -1
        last_status = None

        while True:
            job = await get_job(job_id)

            if not job:
                yield f"event: error\ndata: {{\"error\": \"Job not found\"}}\n\n"
                break

            # Send update if progress or status changed
            current_progress = job.get("progress", 0)
            current_status = job.get("status")

            if current_progress != last_progress or current_status != last_status:
                import json

                data = json.dumps({
                    "id": job_id,
                    "status": current_status,
                    "progress": current_progress,
                    "current_task": job.get("current_task"),
                    "error_message": job.get("error_message"),
                })
                yield f"data: {data}\n\n"
                last_progress = current_progress
                last_status = current_status

            # Stop streaming if job is complete
            if current_status in ("completed", "failed", "cancelled"):
                yield f"event: complete\ndata: {{\"status\": \"{current_status}\"}}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/sources/{source_id}/sync")
async def sync_source(source_id: str) -> Dict[str, Any]:
    """Re-index a data source by creating a new sync job.

    Args:
        source_id: UUID of the data source

    Returns:
        New job details

    Raises:
        HTTPException: 404 if source not found
    """
    from nyrag.sources import create_job, get_data_source, get_job, start_job

    # Verify source exists
    source = await get_data_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Data source not found: {source_id}")

    # Create a sync job
    job = await create_job(source_id, "sync")

    # Try to start the job
    started = await start_job(job["id"])

    # Return job details
    job_details = await get_job(job["id"])
    return {
        "message": "Sync job created",
        "started": started,
        "job": job_details,
    }


@app.get("/api/sources/{source_id}/progress")
async def stream_source_progress(source_id: str) -> StreamingResponse:
    """Stream real-time progress for the active job on a source.

    Args:
        source_id: UUID of the data source

    Returns:
        Server-Sent Events stream with progress updates
    """
    import asyncio

    from nyrag.sources import get_data_source, list_jobs

    async def event_generator():
        """Generate SSE events for source processing progress."""
        while True:
            # Check if source exists
            source = await get_data_source(source_id)
            if not source:
                yield f"event: error\ndata: {{\"error\": \"Source not found\"}}\n\n"
                break

            # Find active job for this source
            jobs, _ = await list_jobs(source_id=source_id, limit=1)
            if not jobs:
                yield f"event: no_job\ndata: {{\"message\": \"No active job\"}}\n\n"
                break

            job = jobs[0]
            import json

            data = json.dumps({
                "job_id": job.get("id"),
                "status": job.get("status"),
                "progress": job.get("progress", 0),
                "current_task": job.get("current_task"),
                "error_message": job.get("error_message"),
            })
            yield f"data: {data}\n\n"

            # Stop if job is complete
            if job.get("status") in ("completed", "failed", "cancelled"):
                yield f"event: complete\ndata: {{\"status\": \"{job.get('status')}\"}}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# =============================================================================
# Conversations API Endpoints
# =============================================================================


class CreateConversationRequest(BaseModel):
    """Request model for creating a conversation."""

    title: Optional[str] = Field(None, description="Optional conversation title")


class AddMessageRequest(BaseModel):
    """Request model for adding a message to a conversation."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


@app.get("/api/conversations")
async def list_conversations_endpoint(
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List conversations ordered by most recent activity.

    Args:
        limit: Maximum number of results (default 50, max 100)
        offset: Pagination offset

    Returns:
        Dict with 'conversations' list and 'total' count
    """
    from nyrag.sources import list_conversations

    conversations, total = await list_conversations(limit=limit, offset=offset)

    return {
        "conversations": conversations,
        "total": total,
        "limit": min(max(limit, 1), 100),
        "offset": max(offset, 0),
    }


@app.post("/api/conversations", status_code=201)
async def create_conversation_endpoint(
    request: Optional[CreateConversationRequest] = None,
) -> Dict[str, Any]:
    """Create a new conversation.

    Args:
        request: Optional request with title

    Returns:
        Created conversation details
    """
    from nyrag.sources import create_conversation

    title = request.title if request else None
    conversation = await create_conversation(title=title)

    return conversation


@app.get("/api/conversations/{conversation_id}")
async def get_conversation_endpoint(conversation_id: str) -> Dict[str, Any]:
    """Get a conversation with all its messages.

    Args:
        conversation_id: UUID of the conversation

    Returns:
        Conversation details with messages array

    Raises:
        HTTPException: 404 if conversation not found
    """
    from nyrag.sources import get_conversation_with_messages

    conversation = await get_conversation_with_messages(conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    return conversation


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str) -> Dict[str, Any]:
    """Delete a conversation and all its messages.

    Args:
        conversation_id: UUID of the conversation

    Returns:
        Deletion confirmation

    Raises:
        HTTPException: 404 if conversation not found
    """
    from nyrag.sources import delete_conversation

    success = await delete_conversation(conversation_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    return {"message": "Conversation deleted successfully", "id": conversation_id}


@app.post("/api/conversations/{conversation_id}/messages", status_code=201)
async def add_message_endpoint(
    conversation_id: str,
    request: AddMessageRequest,
) -> Dict[str, Any]:
    """Add a message to a conversation.

    Args:
        conversation_id: UUID of the conversation
        request: Message data (role and content)

    Returns:
        Created message details

    Raises:
        HTTPException: 400 if invalid role, 404 if conversation not found
    """
    from nyrag.sources import add_message

    if request.role not in ("user", "assistant"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'assistant'")

    try:
        message = await add_message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
        )
        return message
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search")
async def search(req: SearchRequest) -> Dict[str, Any]:
    """Query Vespa using YQL with a precomputed query embedding."""
    embedding = model.encode(req.query, convert_to_numpy=True).tolist()
    body = {
        "yql": "select * from sources * where userInput(@query)",
        "query": req.query,
        "hits": req.hits,
        "summary": req.summary or DEFAULT_SUMMARY,
        "ranking.profile": req.ranking or DEFAULT_RANKING,
        "input.query(embedding)": embedding,
        "input.query(k)": req.k,
    }
    vespa_response = vespa_app.query(body=body, schema=settings["schema_name"])

    status_code = getattr(vespa_response, "status_code", 200)
    if status_code >= 400:
        detail = getattr(vespa_response, "json", vespa_response)
        raise HTTPException(status_code=status_code, detail=detail)

    return vespa_response.json


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation messages as list of {role, content} dicts",
    )
    hits: int = Field(5, description="Number of Vespa hits to retrieve")
    k: int = Field(3, description="Top-k chunks per hit to keep")
    query_k: int = Field(
        3,
        ge=0,
        description="Number of alternate search queries to generate with the LLM",
    )
    model: Optional[str] = Field(None, description="OpenRouter model id (optional, uses env default if set)")


def _fetch_chunks(query: str, hits: int, k: int) -> List[Dict[str, Any]]:
    embedding = model.encode(query, convert_to_numpy=True).tolist()
    body = {
        "yql": "select * from sources * where userInput(@query)",
        "query": query,
        "hits": hits,
        "summary": DEFAULT_SUMMARY,
        "ranking.profile": DEFAULT_RANKING,
        "input.query(embedding)": embedding,
        "input.query(k)": k,
        "presentation.summaryFeatures": True,
    }
    vespa_response = vespa_app.query(body=body, schema=settings["schema_name"])
    status_code = getattr(vespa_response, "status_code", 200)
    if status_code >= 400:
        detail = getattr(vespa_response, "json", vespa_response)
        raise HTTPException(status_code=status_code, detail=detail)

    hits_data = vespa_response.json.get("root", {}).get("children", []) or []
    chunks: List[Dict[str, Any]] = []
    for hit in hits_data:
        fields = hit.get("fields", {}) or {}
        loc = fields.get("loc") or fields.get("id") or ""
        chunk_texts = fields.get("chunks_topk") or []
        hit_score_raw = hit.get("relevance", 0.0)
        logger.info(f"Hit loc={loc} score={hit_score_raw} chunks={len(chunk_texts)}")
        try:
            hit_score = float(hit_score_raw)
        except (TypeError, ValueError):
            hit_score = 0.0
        summary_features = (
            hit.get("summaryfeatures") or hit.get("summaryFeatures") or fields.get("summaryfeatures") or {}
        )
        chunk_score_raw = summary_features.get("best_chunk_score", hit_score)
        logger.info(f"  best_chunk_score={chunk_score_raw}")
        try:
            chunk_score = float(chunk_score_raw)
        except (TypeError, ValueError):
            chunk_score = hit_score

        for chunk in chunk_texts:
            chunks.append(
                {
                    "loc": loc,
                    "chunk": chunk,
                    "score": chunk_score,
                    "hit_score": hit_score,
                    "source_query": query,
                }
            )
    return chunks


def _fetch_notes_as_chunks(query: str, hits: int) -> List[Dict[str, Any]]:
    """Fetch notes matching the query and return them as chunks for RAG context.

    This integrates notes with the main search results, treating each note's
    content as a chunk for the RAG pipeline.
    """
    try:
        config = _get_config_for_notes()
        notes = search_notes(query, config, limit=hits)

        chunks: List[Dict[str, Any]] = []
        for note in notes:
            # Use note title as location, content as chunk
            chunks.append(
                {
                    "loc": f"note:{note.id}:{note.title}",
                    "chunk": note.content[:2000],  # Limit chunk size
                    "score": 0.5,  # Default score for notes
                    "hit_score": 0.5,
                    "source_query": query,
                    "source_type": "note",
                }
            )
        return chunks
    except Exception as e:
        logger.warning(f"Failed to fetch notes: {e}")
        return []


async def _fetch_chunks_async(query: str, hits: int, k: int) -> List[Dict[str, Any]]:
    """Fetch chunks from both main schema and notes schema."""
    loop = asyncio.get_running_loop()
    # Fetch from main schema
    main_chunks = await loop.run_in_executor(None, partial(_fetch_chunks, query, hits, k))
    # Also fetch from notes
    note_chunks = await loop.run_in_executor(None, partial(_fetch_notes_as_chunks, query, min(hits, 3)))
    # Combine and return
    return main_chunks + note_chunks


def _get_openrouter_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    default_headers = {}
    referer = os.getenv("OPENROUTER_REFERRER")
    if referer:
        default_headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_TITLE")
    if title:
        default_headers["X-Title"] = title
    return AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=default_headers or None)


def _extract_message_text(content: Any) -> str:
    """Handle OpenAI response content that may be str or list of text blocks."""
    if content is None:
        return ""
    if isinstance(content, dict) and "text" in content:
        return str(content.get("text", ""))
    if hasattr(content, "text"):
        return str(getattr(content, "text", ""))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif hasattr(part, "text"):
                texts.append(str(getattr(part, "text", "")))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join([t for t in texts if t])
    return str(content)


async def _generate_search_queries_stream(
    user_message: str,
    model_id: str,
    num_queries: int,
    hits: int,
    k: int,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Use the chat LLM to propose focused search queries grounded in retrieved chunks."""
    if num_queries <= 0:
        yield "result", []
        return

    grounding_chunks = (await _fetch_chunks_async(user_message, hits=hits, k=k))[:5]
    grounding_text = "\n".join(f"- [{c.get('loc','')}] {c.get('chunk','')}" for c in grounding_chunks)

    system_prompt = (
        "You generate concise, to-the-point search queries that help retrieve"
        " factual context for answering the user."
        " Do not change the meaning of the question."
        " Do not introduce any new information, words, concepts, or ideas."
        " Do not add any new words."
        " Prefer to reuse the provided context to stay on-topic."
        "Return only valid JSON."
    )

    # Build conversation context if history exists
    conversation_context = ""
    if history:
        conversation_context = "Previous conversation:\n"
        for msg in history[-4:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            conversation_context += f"{role}: {content}\n"
        conversation_context += "\n"

    user_prompt = (
        f"{conversation_context}"
        f"Create {num_queries} diverse, specific search queries (max 12 words each)"
        f' that would retrieve evidence to answer:\n"{user_message}".\n'
        f"Grounding context:\n{grounding_text or '(no context found)'}\n"
        'Respond as a JSON object like {"queries": ["query 1", "query 2"]}.'
    )

    full_content = ""
    try:
        client = _get_openrouter_client()
        stream = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            stream=True,
            extra_body={"reasoning": {"enabled": True}},
        )

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            reasoning = getattr(delta, "reasoning", None)
            reasoning_text = _extract_message_text(reasoning)
            if reasoning_text:
                yield "thinking", reasoning_text

            content_piece = _extract_message_text(getattr(delta, "content", None))
            if content_piece:
                full_content += content_piece
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    queries: List[str] = []
    try:
        parsed = json.loads(full_content)
        candidates = parsed.get("queries") if isinstance(parsed, dict) else parsed
        if isinstance(candidates, list):
            queries = [str(q).strip() for q in candidates if str(q).strip()]
    except Exception:
        queries = []

    # Fallback: try to parse line-separated text if JSON parsing fails
    if not queries:
        for line in full_content.splitlines():
            candidate = line.strip(" -•\t")
            if candidate:
                queries.append(candidate)

    cleaned: List[str] = []
    seen: Set[str] = set()
    for q in queries:
        q_norm = q.strip()
        key = q_norm.lower()
        if q_norm and key not in seen:
            cleaned.append(q_norm)
            seen.add(key)
        if len(cleaned) >= num_queries:
            break
    yield "result", cleaned


async def _prepare_queries_stream(
    user_message: str,
    model_id: str,
    query_k: int,
    hits: int,
    k: int,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Build the list of queries (original + enhanced) for retrieval."""
    enhanced = []
    async for event_type, payload in _generate_search_queries_stream(
        user_message, model_id, query_k, hits=hits, k=k, history=history
    ):
        if event_type == "thinking":
            yield "thinking", payload
        elif event_type == "result":
            enhanced = payload

    queries = [user_message] + enhanced

    deduped: List[str] = []
    seen: Set[str] = set()
    for q in queries:
        q_norm = q.strip()
        key = q_norm.lower()
        if q_norm and key not in seen:
            deduped.append(q_norm)
            seen.add(key)
    logger.info(f"Search queries ({len(deduped)}): {deduped}")
    yield "result", deduped


async def _prepare_queries(user_message: str, model_id: str, query_k: int, hits: int, k: int) -> List[str]:
    queries = []
    async for event_type, payload in _prepare_queries_stream(user_message, model_id, query_k, hits, k):
        if event_type == "result":
            queries = payload
    return queries


async def _fuse_chunks(queries: List[str], hits: int, k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Search Vespa for each query and return fused, deduped chunks."""
    all_chunks: List[Dict[str, Any]] = []
    logger.info(f"Fetching chunks for {len(queries)} queries")

    tasks = [_fetch_chunks_async(q, hits=hits, k=k) for q in queries]
    results = await asyncio.gather(*tasks)
    for res in results:
        all_chunks.extend(res)

    logger.info(f"Fetched total {len(all_chunks)} chunks from Vespa")
    if not all_chunks:
        return queries, []

    max_context = hits * k
    if max_context <= 0:
        max_context = len(all_chunks)

    # Aggregate duplicates (same loc+chunk) and average their scores.
    aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chunk in all_chunks:
        key = (chunk.get("loc", ""), chunk.get("chunk", ""))
        score = float(chunk.get("score", chunk.get("hit_score", 0.0)) or 0.0)
        hit_score = float(chunk.get("hit_score", 0.0) or 0.0)
        source_query = chunk.get("source_query")

        if key not in aggregated:
            aggregated[key] = {
                "loc": key[0],
                "chunk": key[1],
                "score_sum": score,
                "hit_sum": hit_score,
                "count": 1,
                "source_queries": [source_query] if source_query else [],
            }
        else:
            agg = aggregated[key]
            agg["score_sum"] += score
            agg["hit_sum"] += hit_score
            agg["count"] += 1
            if source_query and source_query not in agg["source_queries"]:
                agg["source_queries"].append(source_query)

    fused: List[Dict[str, Any]] = []
    for agg in aggregated.values():
        count = max(agg.pop("count", 1), 1)
        agg["score"] = agg.pop("score_sum", 0.0) / count
        agg["hit_score"] = agg.pop("hit_sum", 0.0) / count
        sources = agg.get("source_queries") or []
        agg["source_query"] = sources[0] if sources else ""
        fused.append(agg)

    fused.sort(key=lambda c: c.get("score", c.get("hit_score", 0.0)), reverse=True)
    fused = fused[:max_context]

    return queries, fused


async def _call_openrouter(context: List[Dict[str, str]], user_message: str, model_id: str) -> str:
    system_prompt = (
        "You are a helpful assistant. "
        "Answer user's question using only the provided context. "
        "Provide elaborate and informative answers where possible. "
        "If the context is insufficient, say you don't know."
    )
    context_text = "\n\n".join([f"[{c.get('loc','')}] {c.get('chunk','')}" for c in context])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Grounding context:\n{context_text}\n\nQuestion: {user_message}",
        },
    ]

    try:
        client = _get_openrouter_client()
        resp = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            extra_body={"reasoning": {"enabled": True}},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _extract_message_text(resp.choices[0].message.content)


async def _openrouter_stream(
    context: List[Dict[str, str]],
    user_message: str,
    model_id: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Tuple[str, str], None]:
    system_prompt = (
        "You are a helpful assistant. Answer using only the provided context. "
        "If the context is insufficient, say you don't know."
    )
    context_text = "\n\n".join([f"[{c.get('loc','')}] {c.get('chunk','')}" for c in context])

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided
    if history:
        messages.extend(history)

    # Add current user message with context
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {user_message}",
        }
    )

    try:
        client = _get_openrouter_client()
        stream = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            extra_body={"reasoning": {"enabled": True}},
        )

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            reasoning = _extract_message_text(getattr(delta, "reasoning", None))
            if reasoning:
                yield "thinking", reasoning

            content_piece = _extract_message_text(getattr(delta, "content", None))
            if content_piece:
                yield "token", content_piece
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    model_id = req.model or os.getenv("OPENROUTER_MODEL")
    queries, chunks = await _fuse_chunks(
        await _prepare_queries(req.message, model_id, req.query_k, hits=req.hits, k=req.k),
        hits=req.hits,
        k=req.k,
    )
    if not chunks:
        return {"answer": "No relevant context found.", "chunks": []}
    answer = await _call_openrouter(chunks, req.message, model_id)
    return {"answer": answer, "chunks": chunks, "queries": queries}


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    model_id = req.model or os.getenv("OPENROUTER_MODEL")

    # Check for blog generation intent
    blog_topic = detect_blog_intent(req.message)

    async def event_stream():
        # If blog intent detected, handle blog generation flow
        if blog_topic:
            yield f"data: {json.dumps({'type': 'status', 'payload': 'Blog generation request detected...'})}\n\n"

            try:
                config = _get_config_for_notes()
                job_queue = get_job_queue()

                # Submit blog generation job
                job_id = job_queue.submit(
                    job_type="blog_generation",
                    task_fn=generate_blog_task,
                    topic=blog_topic,
                    config=config,
                    template=None,
                    instructions=None,
                )

                blog_job_data = {"type": "blog_job", "payload": {"job_id": job_id, "topic": blog_topic}}
                yield f"data: {json.dumps(blog_job_data)}\n\n"

                response_text = (
                    f"📝 **Blog Generation Started**\n\n"
                    f"I've started generating a blog post about: **{blog_topic}**\n\n"
                    f"Job ID: `{job_id}`\n\n"
                    f"The blog is being generated in the background. "
                    f"You can check its status or download it once complete."
                )
                yield f"data: {json.dumps({'type': 'token', 'payload': response_text})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            except Exception as e:
                err_msg = f"Failed to start blog generation: {str(e)}"
                yield f"data: {json.dumps({'type': 'token', 'payload': err_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

        # Normal chat flow
        yield f"data: {json.dumps({'type': 'status', 'payload': 'Generating search queries...'})}\n\n"

        queries = []
        async for event_type, payload in _prepare_queries_stream(
            req.message,
            model_id,
            req.query_k,
            hits=req.hits,
            k=req.k,
            history=req.history,
        ):
            if event_type == "thinking":
                yield f"data: {json.dumps({'type': 'thinking', 'payload': payload})}\n\n"
            elif event_type == "result":
                queries = payload

        yield f"data: {json.dumps({'type': 'queries', 'payload': queries})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'payload': 'Retrieving context from Vespa...'})}\n\n"
        queries, chunks = await _fuse_chunks(queries, hits=req.hits, k=req.k)
        yield f"data: {json.dumps({'type': 'chunks', 'payload': chunks})}\n\n"
        if not chunks:
            yield f"data: {json.dumps({'type': 'done', 'payload': 'No relevant context found.'})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'status', 'payload': 'Generating answer...'})}\n\n"
        async for type_, payload in _openrouter_stream(chunks, req.message, model_id, req.history):
            yield f"data: {json.dumps({'type': type_, 'payload': payload})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/", response_class=HTMLResponse)
async def root_ui(request: Request) -> HTMLResponse:
    """Render the chat UI at root path."""
    return templates.TemplateResponse("chat_new.html", {"request": request, "active_page": "chat"})


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui(request: Request) -> HTMLResponse:
    """Render the chat UI."""
    return templates.TemplateResponse("chat_new.html", {"request": request, "active_page": "chat"})


# =============================================================================
# Notes API Endpoints (Phase 3 - US1)
# =============================================================================


class CreateNoteRequest(BaseModel):
    """Request model for creating a note."""

    title: str = Field(..., description="Note title")
    content: str = Field(..., description="Markdown content")
    tags: List[str] = Field(default_factory=list, description="Optional tags")


class NoteResponse(BaseModel):
    """Response model for a note."""

    id: str
    title: str
    content: str
    images: List[str]
    created_at: str
    updated_at: str
    tags: List[str]


class ImageUploadResponse(BaseModel):
    """Response model for image upload."""

    url: str


@app.get("/api/notes", response_model=List[NoteResponse])
async def get_notes():
    """Get all notes."""
    from nyrag.notes import list_notes

    config = _get_config_for_notes()
    notes = list_notes(config)

    return [
        NoteResponse(
            id=note.id,
            title=note.title,
            content=note.content,
            images=note.images,
            created_at=note.created_at.isoformat(),
            updated_at=note.updated_at.isoformat(),
            tags=note.tags,
        )
        for note in notes
    ]


@app.post("/api/notes", response_model=NoteResponse)
async def create_note(req: CreateNoteRequest):
    """Create a new note and save it locally + index in Vespa."""
    config = _get_config_for_notes()

    note = Note(
        id="",  # Will be generated by save_note
        title=req.title,
        content=req.content,
        tags=req.tags,
    )

    saved_note = save_note(note, config)

    return NoteResponse(
        id=saved_note.id,
        title=saved_note.title,
        content=saved_note.content,
        images=saved_note.images,
        created_at=saved_note.created_at.isoformat(),
        updated_at=saved_note.updated_at.isoformat(),
        tags=saved_note.tags,
    )


@app.post("/api/notes/upload-image", response_model=ImageUploadResponse)
async def upload_note_image(
    file: UploadFile = File(...),
    note_id: str = Form(default="temp"),
):
    """Upload an image for a note."""
    config = _get_config_for_notes()

    # Read file content
    image_data = await file.read()

    # Use temp note_id if not provided (for images uploaded before note is saved)
    if not note_id or note_id == "temp":
        import uuid as uuid_module

        note_id = f"temp_{uuid_module.uuid4().hex[:8]}"

    try:
        url = save_image(image_data, note_id, file.filename or "image.png", config)
        return ImageUploadResponse(url=url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/notes/{note_id}", response_model=NoteResponse)
async def get_note_by_id(note_id: str):
    """Retrieve a note by its ID."""
    config = _get_config_for_notes()

    note = get_note(note_id, config)
    if not note:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")

    return NoteResponse(
        id=note.id,
        title=note.title,
        content=note.content,
        images=note.images,
        created_at=note.created_at.isoformat(),
        updated_at=note.updated_at.isoformat(),
        tags=note.tags,
    )


@app.get("/api/notes/search", response_model=List[NoteResponse])
async def search_notes_api(q: str, limit: int = 10):
    """Search notes by query."""
    config = _get_config_for_notes()

    notes = search_notes(q, config, limit=limit)

    return [
        NoteResponse(
            id=note.id,
            title=note.title,
            content=note.content,
            images=note.images,
            created_at=note.created_at.isoformat(),
            updated_at=note.updated_at.isoformat(),
            tags=note.tags,
        )
        for note in notes
    ]


@app.get("/notes", response_class=HTMLResponse)
async def notes_ui(request: Request) -> HTMLResponse:
    """Render the notes capture UI."""
    return templates.TemplateResponse("notes.html", {"request": request, "active_page": "notes"})


@app.get("/sources", response_class=HTMLResponse)
async def sources_ui(request: Request) -> HTMLResponse:
    """Render the data sources management UI."""
    return templates.TemplateResponse("sources.html", {"request": request, "active_page": "sources"})


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_ui(request: Request) -> HTMLResponse:
    """Render the background jobs monitoring UI."""
    return templates.TemplateResponse("jobs.html", {"request": request, "active_page": "jobs"})


@app.get("/agents", response_class=HTMLResponse)
async def agents_ui(request: Request) -> HTMLResponse:
    """Render the AI agents UI."""
    return templates.TemplateResponse("agents.html", {"request": request, "active_page": "agents"})


# =============================================================================
# Blog Generation API Endpoints (Phase 4 - US2)
# =============================================================================


class GenerateBlogRequest(BaseModel):
    """Request model for blog generation."""

    topic: str = Field(..., description="Blog topic/title")
    template: Optional[str] = Field(None, description="Optional template name")
    instructions: Optional[str] = Field(None, description="Optional additional instructions")


class GenerateBlogResponse(BaseModel):
    """Response model for blog generation request."""

    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    id: str
    type: str
    status: str
    progress: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class BlogResponse(BaseModel):
    """Response model for a blog post."""

    id: str
    topic: str
    template: Optional[str]
    content: str
    source_notes: List[str]
    status: str
    created_at: str
    error: Optional[str] = None


# Blog intent detection patterns
BLOG_INTENT_PATTERNS = [
    "generate blog",
    "write blog",
    "create blog",
    "write a blog",
    "create a blog",
    "generate a blog",
    "write post",
    "create post",
    "generate post",
    "blog post about",
    "blog about",
    "write article",
    "create article",
]


def detect_blog_intent(message: str) -> Optional[str]:
    """Detect if a chat message is requesting blog generation.

    Args:
        message: The user's chat message.

    Returns:
        The blog topic if intent detected, None otherwise.
    """
    message_lower = message.lower().strip()

    for pattern in BLOG_INTENT_PATTERNS:
        if pattern in message_lower:
            # Extract topic - everything after the pattern
            idx = message_lower.find(pattern)
            topic = message[idx + len(pattern) :].strip()
            # Remove leading "about" if present
            if topic.lower().startswith("about "):
                topic = topic[6:].strip()
            # Remove quotes if present
            topic = topic.strip("\"'")
            if topic:
                return topic
            # If no topic extracted, use the full message as topic
            return message

    return None


@app.post("/api/blog/generate", response_model=GenerateBlogResponse)
async def generate_blog(req: GenerateBlogRequest):
    """Start background blog generation.

    Submits the blog generation task to the job queue and returns immediately
    with a job ID that can be polled for status.
    """
    config = _get_config_for_notes()
    job_queue = get_job_queue()

    # Submit to job queue
    job_id = job_queue.submit(
        job_type="blog_generation",
        task_fn=generate_blog_task,
        topic=req.topic,
        config=config,
        template=req.template,
        instructions=req.instructions,
    )

    return GenerateBlogResponse(
        job_id=job_id,
        status="queued",
        message=f"Blog generation started for topic: {req.topic[:50]}...",
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a background job."""
    job_queue = get_job_queue()
    job = job_queue.get_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # For blog generation, extract blog_id from result
    result = None
    if job.status == JobStatus.COMPLETE and job.result:
        if isinstance(job.result, BlogPost):
            result = {"blog_id": job.result.id, "topic": job.result.topic}
        else:
            result = job.result

    return JobStatusResponse(
        id=job.id,
        type=job.type,
        status=job.status.value,
        progress=job.progress,
        result=result,
        error=job.error,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@app.get("/api/blog/{blog_id}", response_model=BlogResponse)
async def get_blog_by_id(blog_id: str):
    """Retrieve a generated blog post by ID."""
    config = _get_config_for_notes()
    blog = get_blog(blog_id, config)

    if not blog:
        raise HTTPException(status_code=404, detail=f"Blog {blog_id} not found")

    return BlogResponse(
        id=blog.id,
        topic=blog.topic,
        template=blog.template,
        content=blog.content,
        source_notes=blog.source_notes,
        status=blog.status.value,
        created_at=blog.created_at.isoformat(),
        error=blog.error,
    )


@app.get("/api/blog/{blog_id}/download")
async def download_blog(blog_id: str):
    """Download a generated blog post as a markdown file."""
    config = _get_config_for_notes()
    blog_path = get_blog_path(blog_id, config)

    if not blog_path:
        raise HTTPException(status_code=404, detail=f"Blog {blog_id} not found")

    return FileResponse(
        path=str(blog_path),
        filename=f"blog-{blog_id}.md",
        media_type="text/markdown",
    )
