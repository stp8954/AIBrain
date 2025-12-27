/**
 * Agents page JavaScript for NyRAG UI Document Manager
 * Handles background jobs monitoring with real-time progress updates
 */

// =============================================================================
// State Management
// =============================================================================

const state = {
  jobs: [],
  total: 0,
  limit: 20,
  offset: 0,
  filters: {
    status: "",
    search: "",
  },
  stats: {
    running: 0,
    queued: 0,
    completed: 0,
    failed: 0,
  },
  loading: false,
  activeStreams: new Map(), // job_id -> EventSource
};

// =============================================================================
// API Functions
// =============================================================================

/**
 * Fetch jobs from the API with optional filters
 */
async function fetchJobs() {
  const params = new URLSearchParams();

  if (state.filters.status) {
    params.append("status", state.filters.status);
  }

  params.append("limit", state.limit);
  params.append("offset", state.offset);

  try {
    const response = await fetch(`/api/jobs?${params.toString()}`);

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching jobs:", error);
    throw error;
  }
}

/**
 * Fetch a single job by ID
 */
async function fetchJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error fetching job ${jobId}:`, error);
    throw error;
  }
}

/**
 * Cancel a job
 */
async function cancelJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}/cancel`, {
      method: "POST",
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || `HTTP error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error cancelling job ${jobId}:`, error);
    throw error;
  }
}

/**
 * Retry a failed job
 */
async function retryJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}/retry`, {
      method: "POST",
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || `HTTP error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error retrying job ${jobId}:`, error);
    throw error;
  }
}

// =============================================================================
// SSE Streaming
// =============================================================================

/**
 * Start streaming progress updates for a running job
 */
function startJobStream(jobId) {
  if (state.activeStreams.has(jobId)) {
    return; // Already streaming
  }

  const eventSource = new EventSource(`/api/jobs/${jobId}/stream`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateJobInList(data);
    } catch (error) {
      console.error("Error parsing SSE data:", error);
    }
  };

  eventSource.addEventListener("complete", (event) => {
    stopJobStream(jobId);
    loadJobs(); // Refresh to update stats
  });

  eventSource.addEventListener("error", (event) => {
    console.error(`SSE error for job ${jobId}:`, event);
    stopJobStream(jobId);
  });

  eventSource.onerror = () => {
    stopJobStream(jobId);
  };

  state.activeStreams.set(jobId, eventSource);
}

/**
 * Stop streaming for a job
 */
function stopJobStream(jobId) {
  const eventSource = state.activeStreams.get(jobId);
  if (eventSource) {
    eventSource.close();
    state.activeStreams.delete(jobId);
  }
}

/**
 * Stop all active streams
 */
function stopAllStreams() {
  for (const [jobId, eventSource] of state.activeStreams) {
    eventSource.close();
  }
  state.activeStreams.clear();
}

/**
 * Update a job's data in the list without full reload
 */
function updateJobInList(data) {
  const jobCard = document.querySelector(`[data-job-id="${data.id}"]`);
  if (!jobCard) return;

  // Update status badge
  const statusBadge = jobCard.querySelector(".job-status");
  if (statusBadge) {
    statusBadge.className = `job-status ${data.status}`;
    statusBadge.innerHTML = getStatusBadgeContent(data.status);
  }

  // Update progress bar
  const progressFill = jobCard.querySelector(".progress-bar-fill");
  if (progressFill) {
    progressFill.style.width = `${data.progress || 0}%`;
    progressFill.className = `progress-bar-fill ${data.status}`;
  }

  // Update progress text
  const progressText = jobCard.querySelector(".progress-text");
  if (progressText) {
    progressText.textContent = `${data.progress || 0}%`;
  }

  // Update current task
  const progressTask = jobCard.querySelector(".progress-task");
  if (progressTask && data.current_task) {
    progressTask.textContent = data.current_task;
  }

  // Update card class
  jobCard.className = `job-card ${data.status}`;

  // Show/hide actions based on status
  updateJobActions(jobCard, data);
}

// =============================================================================
// UI Rendering
// =============================================================================

/**
 * Load and display jobs
 */
async function loadJobs() {
  if (state.loading) return;

  state.loading = true;
  showLoading(true);

  try {
    const data = await fetchJobs();
    state.jobs = data.jobs || [];
    state.total = data.total || 0;

    // Calculate stats
    calculateStats();

    // Render jobs
    renderJobs();
    renderPagination();

    // Start streaming for running jobs
    state.jobs.forEach((job) => {
      if (job.status === "running") {
        startJobStream(job.id);
      }
    });
  } catch (error) {
    showError("Failed to load jobs. Please try again.");
  } finally {
    state.loading = false;
    showLoading(false);
  }
}

/**
 * Calculate stats from all jobs (fetch without filters to get counts)
 */
async function calculateStats() {
  try {
    // Fetch counts for each status
    const [running, queued, completed, failed] = await Promise.all([
      fetch("/api/jobs?status=running&limit=0").then((r) => r.json()),
      fetch("/api/jobs?status=queued&limit=0").then((r) => r.json()),
      fetch("/api/jobs?status=completed&limit=0").then((r) => r.json()),
      fetch("/api/jobs?status=failed&limit=0").then((r) => r.json()),
    ]);

    state.stats = {
      running: running.total || 0,
      queued: queued.total || 0,
      completed: completed.total || 0,
      failed: failed.total || 0,
    };

    renderStats();
  } catch (error) {
    console.error("Error calculating stats:", error);
  }
}

/**
 * Render stats cards
 */
function renderStats() {
  document.getElementById("stat-running").textContent = state.stats.running;
  document.getElementById("stat-queued").textContent = state.stats.queued;
  document.getElementById("stat-completed").textContent = state.stats.completed;
  document.getElementById("stat-failed").textContent = state.stats.failed;
}

/**
 * Render jobs list
 */
function renderJobs() {
  const container = document.getElementById("jobs-list");
  const emptyState = document.getElementById("empty-state");

  // Filter by search term client-side
  let filteredJobs = state.jobs;
  if (state.filters.search) {
    const searchLower = state.filters.search.toLowerCase();
    filteredJobs = state.jobs.filter(
      (job) =>
        (job.source_name || "").toLowerCase().includes(searchLower) ||
        (job.job_type || "").toLowerCase().includes(searchLower) ||
        (job.id || "").toLowerCase().includes(searchLower)
    );
  }

  if (filteredJobs.length === 0) {
    container.innerHTML = "";
    emptyState.style.display = "flex";
    return;
  }

  emptyState.style.display = "none";
  container.innerHTML = filteredJobs.map(renderJobCard).join("");

  // Add event listeners to job cards
  container.querySelectorAll(".job-card").forEach((card) => {
    const jobId = card.dataset.jobId;

    // View details
    card.querySelector(".btn-view")?.addEventListener("click", (e) => {
      e.stopPropagation();
      showJobDetails(jobId);
    });

    // Cancel
    card.querySelector(".btn-cancel")?.addEventListener("click", (e) => {
      e.stopPropagation();
      handleCancelJob(jobId);
    });

    // Retry
    card.querySelector(".btn-retry")?.addEventListener("click", (e) => {
      e.stopPropagation();
      handleRetryJob(jobId);
    });
  });
}

/**
 * Render a single job card
 */
function renderJobCard(job) {
  const statusClass = job.status || "queued";
  const progress = job.progress || 0;
  const currentTask = job.current_task || "";
  const jobType = formatJobType(job.job_type);
  const startedAt = formatDate(job.started_at);

  return `
    <div class="job-card ${statusClass}" data-job-id="${job.id}">
      <div class="job-header">
        <div class="job-info">
          <h3 class="job-title">${escapeHtml(job.source_name || "Unknown Source")}</h3>
          <div class="job-meta">
            <span class="job-meta-item">${jobType}</span>
            <span class="job-meta-separator">•</span>
            <span class="job-meta-item">${startedAt}</span>
          </div>
        </div>
        <span class="job-status ${statusClass}">
          ${getStatusBadgeContent(job.status)}
        </span>
      </div>

      <div class="job-progress">
        <div class="progress-bar-container">
          <div class="progress-bar">
            <div class="progress-bar-fill ${statusClass}" style="width: ${progress}%"></div>
          </div>
          <span class="progress-text">${progress}%</span>
        </div>
        ${currentTask ? `<div class="progress-task">${escapeHtml(currentTask)}</div>` : ""}
      </div>

      ${job.error_message ? renderJobError(job.error_message) : ""}

      <div class="job-actions">
        <button class="btn btn-secondary btn-sm btn-view">View Details</button>
        ${renderJobActionButtons(job)}
      </div>
    </div>
  `;
}

/**
 * Get status badge HTML content
 */
function getStatusBadgeContent(status) {
  if (status === "running") {
    return '<span class="spinner-small"></span> Running';
  }
  return status.charAt(0).toUpperCase() + status.slice(1);
}

/**
 * Render action buttons based on job status
 */
function renderJobActionButtons(job) {
  if (job.status === "running" || job.status === "queued") {
    return '<button class="btn btn-danger btn-sm btn-cancel">Cancel</button>';
  }
  if (job.status === "failed") {
    return '<button class="btn btn-primary btn-sm btn-retry">Retry</button>';
  }
  return "";
}

/**
 * Update action buttons visibility based on status
 */
function updateJobActions(jobCard, data) {
  const actionsContainer = jobCard.querySelector(".job-actions");
  if (!actionsContainer) return;

  // Get current buttons
  const cancelBtn = actionsContainer.querySelector(".btn-cancel");
  const retryBtn = actionsContainer.querySelector(".btn-retry");

  if (data.status === "running" || data.status === "queued") {
    if (!cancelBtn) {
      const btn = document.createElement("button");
      btn.className = "btn btn-danger btn-sm btn-cancel";
      btn.textContent = "Cancel";
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        handleCancelJob(data.id);
      });
      actionsContainer.appendChild(btn);
    }
    if (retryBtn) retryBtn.remove();
  } else if (data.status === "failed") {
    if (cancelBtn) cancelBtn.remove();
    if (!retryBtn) {
      const btn = document.createElement("button");
      btn.className = "btn btn-primary btn-sm btn-retry";
      btn.textContent = "Retry";
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        handleRetryJob(data.id);
      });
      actionsContainer.appendChild(btn);
    }
  } else {
    if (cancelBtn) cancelBtn.remove();
    if (retryBtn) retryBtn.remove();
  }
}

/**
 * Render job error message
 */
function renderJobError(errorMessage) {
  return `
    <div class="job-error">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="8" x2="12" y2="12"></line>
        <line x1="12" y1="16" x2="12.01" y2="16"></line>
      </svg>
      <span>${escapeHtml(errorMessage)}</span>
    </div>
  `;
}

/**
 * Render pagination controls
 */
function renderPagination() {
  const pagination = document.getElementById("pagination");
  const pageInfo = document.getElementById("page-info");
  const prevBtn = document.getElementById("prev-page");
  const nextBtn = document.getElementById("next-page");

  const totalPages = Math.ceil(state.total / state.limit);
  const currentPage = Math.floor(state.offset / state.limit) + 1;

  if (totalPages <= 1) {
    pagination.style.display = "none";
    return;
  }

  pagination.style.display = "flex";
  pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;

  prevBtn.disabled = currentPage <= 1;
  nextBtn.disabled = currentPage >= totalPages;
}

// =============================================================================
// Event Handlers
// =============================================================================

/**
 * Handle cancel job button click
 */
async function handleCancelJob(jobId) {
  if (!confirm("Are you sure you want to cancel this job?")) {
    return;
  }

  try {
    await cancelJob(jobId);
    stopJobStream(jobId);
    await loadJobs();
    showToast("Job cancelled successfully", "success");
  } catch (error) {
    showToast(error.message || "Failed to cancel job", "error");
  }
}

/**
 * Handle retry job button click
 */
async function handleRetryJob(jobId) {
  try {
    const result = await retryJob(jobId);
    await loadJobs();
    showToast("Job retry initiated", "success");

    // Start streaming for the new job
    if (result.new_job && result.new_job.id) {
      startJobStream(result.new_job.id);
    }
  } catch (error) {
    showToast(error.message || "Failed to retry job", "error");
  }
}

/**
 * Show job details in modal
 */
async function showJobDetails(jobId) {
  const modal = document.getElementById("job-detail-modal");
  const content = document.getElementById("job-detail-content");

  try {
    const job = await fetchJob(jobId);

    content.innerHTML = `
      <div class="detail-section">
        <h4 class="detail-section-title">Overview</h4>
        <div class="detail-grid">
          <div class="detail-item">
            <span class="detail-label">Source</span>
            <span class="detail-value">${escapeHtml(job.source_name || "Unknown")}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Job Type</span>
            <span class="detail-value">${formatJobType(job.job_type)}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Status</span>
            <span class="detail-value">
              <span class="job-status ${job.status}">${getStatusBadgeContent(job.status)}</span>
            </span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Progress</span>
            <span class="detail-value">${job.progress || 0}%</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Started At</span>
            <span class="detail-value">${formatDate(job.started_at)}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Completed At</span>
            <span class="detail-value">${job.completed_at ? formatDate(job.completed_at) : "-"}</span>
          </div>
        </div>
      </div>

      ${
        job.current_task
          ? `
        <div class="detail-section">
          <h4 class="detail-section-title">Current Task</h4>
          <p>${escapeHtml(job.current_task)}</p>
        </div>
      `
          : ""
      }

      ${
        job.error_message
          ? `
        <div class="detail-section">
          <h4 class="detail-section-title">Error</h4>
          <div class="job-error">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <span>${escapeHtml(job.error_message)}</span>
          </div>
        </div>
      `
          : ""
      }

      <div class="detail-section">
        <h4 class="detail-section-title">Logs</h4>
        <pre class="job-logs">${escapeHtml(job.logs || "")}</pre>
      </div>

      <div class="detail-section">
        <h4 class="detail-section-title">Identifiers</h4>
        <div class="detail-grid">
          <div class="detail-item">
            <span class="detail-label">Job ID</span>
            <span class="detail-value" style="font-family: monospace; font-size: 0.85em;">${job.id}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Source ID</span>
            <span class="detail-value" style="font-family: monospace; font-size: 0.85em;">${job.source_id}</span>
          </div>
        </div>
      </div>
    `;

    openModal(modal);
  } catch (error) {
    showToast("Failed to load job details", "error");
  }
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format job type for display
 */
function formatJobType(jobType) {
  const types = {
    crawl: "URL Crawl",
    process_file: "File Processing",
    sync: "Re-sync",
  };
  return types[jobType] || jobType || "Unknown";
}

/**
 * Format date for display
 */
function formatDate(dateString) {
  if (!dateString) return "-";

  try {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString();
  } catch {
    return dateString;
  }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Show loading state
 */
function showLoading(show) {
  const loadingState = document.getElementById("loading-state");
  const jobsList = document.getElementById("jobs-list");

  if (show) {
    loadingState.style.display = "flex";
    jobsList.style.opacity = "0.5";
  } else {
    loadingState.style.display = "none";
    jobsList.style.opacity = "1";
  }
}

/**
 * Show error message
 */
function showError(message) {
  showToast(message, "error");
}

/**
 * Show toast notification
 */
function showToast(message, type = "info") {
  // Use existing toast system if available, otherwise console log
  if (window.showToast) {
    window.showToast(message, type);
  } else {
    console.log(`[${type.toUpperCase()}] ${message}`);

    // Simple toast implementation
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      bottom: 20px;
      right: 20px;
      padding: 12px 20px;
      background: ${type === "error" ? "#ef4444" : type === "success" ? "#22c55e" : "#3b82f6"};
      color: white;
      border-radius: 8px;
      z-index: 10000;
      animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = "slideOut 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }
}

/**
 * Open modal
 */
function openModal(modal) {
  modal.classList.add("active");
  document.body.style.overflow = "hidden";
}

/**
 * Close modal
 */
function closeModal(modal) {
  modal.classList.remove("active");
  document.body.style.overflow = "";
}

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Initial load
  loadJobs();

  // Refresh button
  document.getElementById("refresh-jobs-btn")?.addEventListener("click", () => {
    stopAllStreams();
    loadJobs();
  });

  // Status filter
  document.getElementById("filter-status")?.addEventListener("change", (e) => {
    state.filters.status = e.target.value;
    state.offset = 0;
    stopAllStreams();
    loadJobs();
  });

  // Search filter (debounced)
  let searchTimeout;
  document.getElementById("filter-search")?.addEventListener("input", (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      state.filters.search = e.target.value;
      renderJobs();
    }, 300);
  });

  // Stats card clicks (filter by status)
  document.querySelectorAll(".stat-card").forEach((card) => {
    card.addEventListener("click", () => {
      const status = card.dataset.stat;
      const filterSelect = document.getElementById("filter-status");

      // Toggle filter
      if (state.filters.status === status) {
        state.filters.status = "";
        filterSelect.value = "";
        card.dataset.active = "false";
      } else {
        state.filters.status = status;
        filterSelect.value = status;
        document.querySelectorAll(".stat-card").forEach((c) => (c.dataset.active = "false"));
        card.dataset.active = "true";
      }

      state.offset = 0;
      stopAllStreams();
      loadJobs();
    });
  });

  // Pagination
  document.getElementById("prev-page")?.addEventListener("click", () => {
    if (state.offset >= state.limit) {
      state.offset -= state.limit;
      loadJobs();
    }
  });

  document.getElementById("next-page")?.addEventListener("click", () => {
    if (state.offset + state.limit < state.total) {
      state.offset += state.limit;
      loadJobs();
    }
  });

  // Modal close handlers
  document.querySelectorAll(".modal").forEach((modal) => {
    modal.querySelector(".modal-overlay")?.addEventListener("click", () => closeModal(modal));
    modal.querySelector(".modal-close")?.addEventListener("click", () => closeModal(modal));
    modal.querySelector(".modal-cancel")?.addEventListener("click", () => closeModal(modal));
  });

  // Escape key to close modal
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      document.querySelectorAll(".modal.active").forEach(closeModal);
    }
  });

  // Auto-refresh every 30 seconds
  setInterval(() => {
    if (!document.hidden) {
      loadJobs();
    }
  }, 30000);

  // Cleanup streams on page unload
  window.addEventListener("beforeunload", stopAllStreams);
});
