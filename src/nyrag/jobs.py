"""Background job queue module for NyRAG.

This module provides an async job queue for long-running tasks like blog generation.
Jobs are tracked in-memory with status updates and support for cancellation and cleanup.

Key features:
- Async task execution with asyncio
- Job status tracking (queued, running, complete, failed)
- Result storage and retrieval
- Automatic cleanup of expired jobs
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from nyrag.logger import get_logger


logger = get_logger("jobs")


class JobStatus(str, Enum):
    """Status of a background job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """Pydantic model representing a background job."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique job identifier")
    type: str = Field(..., description="Type of job (e.g., 'blog_generation')")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Current job status")
    result: Optional[Any] = Field(None, description="Job result when complete")
    error: Optional[str] = Field(None, description="Error message if job failed")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    progress: Optional[str] = Field(None, description="Optional progress message")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class JobQueue:
    """In-memory async job queue for background task execution.

    This class manages the lifecycle of background jobs including:
    - Job submission and tracking
    - Status queries
    - Result retrieval
    - Cancellation
    - Cleanup of expired jobs
    """

    def __init__(self, job_ttl_hours: int = 24):
        """Initialize the job queue.

        Args:
            job_ttl_hours: Hours to keep completed jobs before cleanup.
        """
        self._jobs: Dict[str, Job] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._job_ttl = timedelta(hours=job_ttl_hours)
        logger.info(f"JobQueue initialized with TTL={job_ttl_hours}h")

    def submit(
        self,
        job_type: str,
        task_fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Submit a new job to the queue.

        Args:
            job_type: Type identifier for the job.
            task_fn: Async function to execute.
            *args: Positional arguments for the task function.
            **kwargs: Keyword arguments for the task function.

        Returns:
            The job ID.
        """
        job = Job(type=job_type)
        self._jobs[job.id] = job
        logger.info(f"Job {job.id} ({job_type}) submitted")

        # Create and start the async task
        async def run_job():
            try:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()
                logger.info(f"Job {job.id} started")

                result = await task_fn(*args, **kwargs)

                job.result = result
                job.status = JobStatus.COMPLETE
                job.completed_at = datetime.utcnow()
                logger.info(f"Job {job.id} completed successfully")
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                logger.warning(f"Job {job.id} was cancelled")
            except Exception as e:
                job.error = str(e)
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                logger.error(f"Job {job.id} failed: {e}")

        # Schedule the task
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(run_job())
            self._tasks[job.id] = task
        except RuntimeError:
            # No running loop - create one for synchronous contexts
            logger.warning(f"No running event loop, job {job.id} will be queued")

        return job.id

    def get_status(self, job_id: str) -> Optional[Job]:
        """Get the current status of a job.

        Args:
            job_id: The job ID to query.

        Returns:
            The job if found, None otherwise.
        """
        return self._jobs.get(job_id)

    def get_result(self, job_id: str) -> Optional[Any]:
        """Get the result of a completed job.

        Args:
            job_id: The job ID to query.

        Returns:
            The job result if complete, None otherwise.
        """
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.COMPLETE:
            return job.result
        return None

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        Args:
            job_id: The job ID to cancel.

        Returns:
            True if cancellation succeeded, False otherwise.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Job {job_id} cancellation requested")
            return True

        # Job was queued but not started
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        return True

    def cleanup_expired(self) -> int:
        """Remove expired completed jobs.

        Returns:
            Number of jobs removed.
        """
        now = datetime.utcnow()
        expired_ids = []

        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at and (now - job.completed_at) > self._job_ttl:
                    expired_ids.append(job_id)

        for job_id in expired_ids:
            del self._jobs[job_id]
            self._tasks.pop(job_id, None)

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired jobs")

        return len(expired_ids)

    def list_jobs(self, status: Optional[JobStatus] = None) -> list[Job]:
        """List all jobs, optionally filtered by status.

        Args:
            status: Optional status filter.

        Returns:
            List of matching jobs.
        """
        if status is None:
            return list(self._jobs.values())
        return [job for job in self._jobs.values() if job.status == status]


# Global job queue instance - initialized on first access
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get or create the global job queue instance.

    Returns:
        The global JobQueue instance.
    """
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue
