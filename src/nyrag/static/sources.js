/**
 * Data Sources Page JavaScript
 * Handles source listing, adding URLs/files, and deletion
 */

// =============================================================================
// State Management
// =============================================================================

const sourcesState = {
  sources: [],
  total: 0,
  currentPage: 1,
  pageSize: 20,
  filters: {
    type: '',
    status: '',
    search: ''
  },
  selectedFiles: [],
  isLoading: false,
  deleteSourceId: null
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
  // Table
  sourcesTable: document.getElementById('sources-table'),
  sourcesTbody: document.getElementById('sources-tbody'),
  emptyState: document.getElementById('empty-state'),
  loadingState: document.getElementById('loading-state'),

  // Filters
  filterType: document.getElementById('filter-type'),
  filterStatus: document.getElementById('filter-status'),
  filterSearch: document.getElementById('filter-search'),
  refreshBtn: document.getElementById('refresh-btn'),

  // Pagination
  pagination: document.getElementById('pagination'),
  prevPage: document.getElementById('prev-page'),
  nextPage: document.getElementById('next-page'),
  pageInfo: document.getElementById('page-info'),

  // Add Source Modal
  addSourceBtn: document.getElementById('add-source-btn'),
  emptyAddBtn: document.getElementById('empty-add-btn'),
  addSourceModal: document.getElementById('add-source-modal'),
  urlTab: document.getElementById('url-tab'),
  fileTab: document.getElementById('file-tab'),
  urlForm: document.getElementById('url-form'),
  sourceUrl: document.getElementById('source-url'),
  sourceName: document.getElementById('source-name'),
  fileUploadArea: document.getElementById('file-upload-area'),
  fileInput: document.getElementById('file-input'),
  fileList: document.getElementById('file-list'),
  cancelBtn: document.getElementById('cancel-btn'),
  submitBtn: document.getElementById('submit-btn'),

  // Delete Modal
  deleteModal: document.getElementById('delete-modal'),
  deleteSourceName: document.getElementById('delete-source-name'),
  deleteCancelBtn: document.getElementById('delete-cancel-btn'),
  deleteConfirmBtn: document.getElementById('delete-confirm-btn')
};

// =============================================================================
// API Functions
// =============================================================================

async function fetchSources() {
  const params = new URLSearchParams();
  if (sourcesState.filters.type) params.append('type', sourcesState.filters.type);
  if (sourcesState.filters.status) params.append('status', sourcesState.filters.status);
  params.append('limit', sourcesState.pageSize);
  params.append('offset', (sourcesState.currentPage - 1) * sourcesState.pageSize);

  const response = await fetch(`/api/sources?${params}`);
  if (!response.ok) throw new Error('Failed to fetch sources');
  return response.json();
}

async function addUrlSource(url, name) {
  const response = await fetch('/api/sources/url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, name: name || undefined })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to add URL');
  }

  return response.json();
}

async function uploadFiles(files) {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  const response = await fetch('/api/sources/files', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail?.message || error.detail || 'Failed to upload files');
  }

  return response.json();
}

async function deleteSource(sourceId) {
  const response = await fetch(`/api/sources/${sourceId}`, {
    method: 'DELETE'
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to delete source');
  }

  return response.json();
}

// =============================================================================
// Rendering Functions
// =============================================================================

function renderSources() {
  const { sources, total } = sourcesState;

  // Handle empty state
  if (sources.length === 0 && !sourcesState.isLoading) {
    elements.sourcesTable.style.display = 'none';
    elements.loadingState.style.display = 'none';
    elements.emptyState.style.display = 'flex';
    elements.pagination.style.display = 'none';
    return;
  }

  elements.emptyState.style.display = 'none';
  elements.loadingState.style.display = 'none';
  elements.sourcesTable.style.display = 'table';

  // Render table rows
  elements.sourcesTbody.innerHTML = sources.map(source => `
    <tr data-source-id="${source.id}">
      <td class="col-name">
        <div class="source-name">
          <span class="source-title">${escapeHtml(source.name)}</span>
          <span class="source-path" title="${escapeHtml(source.url || source.file_path || '')}">
            ${escapeHtml(source.url || source.file_path || '')}
          </span>
        </div>
      </td>
      <td class="col-type">
        <span class="type-badge ${source.source_type}">${source.source_type.toUpperCase()}</span>
      </td>
      <td class="col-status">
        ${renderStatusBadge(source.status)}
      </td>
      <td class="col-documents">${source.document_count || 0}</td>
      <td class="col-date">${formatDate(source.created_at)}</td>
      <td class="col-actions">
        <div class="action-buttons">
          ${source.status === 'failed' ? `
            <button class="btn btn-secondary btn-sm" onclick="retrySource('${source.id}')" title="Retry">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
              </svg>
            </button>
          ` : ''}
          <button class="btn btn-danger btn-sm" onclick="confirmDelete('${source.id}', '${escapeHtml(source.name)}')" title="Delete">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
          </button>
        </div>
      </td>
    </tr>
  `).join('');

  // Update pagination
  const totalPages = Math.ceil(total / sourcesState.pageSize);
  if (totalPages > 1) {
    elements.pagination.style.display = 'flex';
    elements.pageInfo.textContent = `Page ${sourcesState.currentPage} of ${totalPages}`;
    elements.prevPage.disabled = sourcesState.currentPage === 1;
    elements.nextPage.disabled = sourcesState.currentPage === totalPages;
  } else {
    elements.pagination.style.display = 'none';
  }
}

function renderStatusBadge(status) {
  const icons = {
    pending: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>',
    processing: '<span class="spinner-xs"></span>',
    indexed: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    failed: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'
  };

  const labels = {
    pending: 'Pending',
    processing: 'Processing',
    indexed: 'Indexed',
    failed: 'Failed'
  };

  return `<span class="status-badge ${status}">${icons[status] || ''} ${labels[status] || status}</span>`;
}

function renderFileList() {
  elements.fileList.innerHTML = sourcesState.selectedFiles.map((file, index) => `
    <div class="file-item">
      <span class="file-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
      </span>
      <div class="file-info">
        <span class="file-name">${escapeHtml(file.name)}</span>
        <span class="file-size">${formatFileSize(file.size)}</span>
      </div>
      <button class="file-remove" onclick="removeFile(${index})" title="Remove">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
  `).join('');
}

// =============================================================================
// Event Handlers
// =============================================================================

async function loadSources() {
  try {
    sourcesState.isLoading = true;
    elements.loadingState.style.display = 'flex';
    elements.sourcesTable.style.display = 'none';
    elements.emptyState.style.display = 'none';

    const data = await fetchSources();
    sourcesState.sources = data.items;
    sourcesState.total = data.total;

    renderSources();
  } catch (error) {
    console.error('Failed to load sources:', error);
    showToast('Failed to load sources', 'error');
  } finally {
    sourcesState.isLoading = false;
    elements.loadingState.style.display = 'none';
  }
}

function openAddModal() {
  // Reset form
  elements.urlForm.reset();
  sourcesState.selectedFiles = [];
  renderFileList();

  // Show URL tab by default
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelector('[data-tab="url-tab"]').classList.add('active');
  elements.urlTab.style.display = 'block';
  elements.fileTab.style.display = 'none';

  // Show modal
  elements.addSourceModal.style.display = 'flex';
  elements.sourceUrl.focus();
}

function closeAddModal() {
  elements.addSourceModal.style.display = 'none';
  // Reset form fields
  if (elements.urlForm) {
    elements.urlForm.reset();
  }
  if (elements.fileInput) {
    elements.fileInput.value = '';
  }
  if (elements.fileList) {
    elements.fileList.innerHTML = '';
  }
  sourcesState.selectedFiles = [];
}

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');

  document.querySelectorAll('.tab-content').forEach(content => content.style.display = 'none');
  document.getElementById(tabId).style.display = 'block';
}

async function handleSubmit() {
  const activeTab = document.querySelector('.tab-content:not([style*="display: none"])');
  const isUrlTab = activeTab.id === 'url-tab';

  elements.submitBtn.classList.add('loading');
  elements.submitBtn.disabled = true;

  try {
    if (isUrlTab) {
      const url = elements.sourceUrl.value.trim();
      const name = elements.sourceName.value.trim();

      if (!url) {
        showToast('Please enter a URL', 'error');
        return;
      }

      await addUrlSource(url, name);
      showToast('Source added successfully!', 'success');
    } else {
      if (sourcesState.selectedFiles.length === 0) {
        showToast('Please select at least one file', 'error');
        return;
      }

      const result = await uploadFiles(sourcesState.selectedFiles);
      if (result.errors?.length > 0) {
        showToast(`${result.sources.length} files added, ${result.errors.length} failed`, 'warning');
      } else {
        showToast(`${result.sources.length} file(s) added successfully!`, 'success');
      }
    }

    closeAddModal();
    loadSources();
  } catch (error) {
    console.error('Failed to add source:', error);
    showToast(error.message, 'error');
  } finally {
    elements.submitBtn.classList.remove('loading');
    elements.submitBtn.disabled = false;
  }
}

function handleFileSelect(files) {
  const maxSize = 50 * 1024 * 1024; // 50MB
  const validTypes = ['.pdf', '.md', '.markdown', '.txt', '.text'];

  for (const file of files) {
    // Check size
    if (file.size > maxSize) {
      showToast(`${file.name} is too large (max 50MB)`, 'error');
      continue;
    }

    // Check type
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validTypes.includes(ext)) {
      showToast(`${file.name} is not a supported file type`, 'error');
      continue;
    }

    // Check for duplicates
    if (sourcesState.selectedFiles.some(f => f.name === file.name)) {
      continue;
    }

    sourcesState.selectedFiles.push(file);
  }

  renderFileList();
}

function removeFile(index) {
  sourcesState.selectedFiles.splice(index, 1);
  renderFileList();
}

function confirmDelete(sourceId, sourceName) {
  sourcesState.deleteSourceId = sourceId;
  elements.deleteSourceName.textContent = sourceName;
  elements.deleteModal.style.display = 'flex';
}

function closeDeleteModal() {
  sourcesState.deleteSourceId = null;
  elements.deleteModal.style.display = 'none';
}

async function handleDelete() {
  if (!sourcesState.deleteSourceId) return;

  elements.deleteConfirmBtn.disabled = true;
  elements.deleteConfirmBtn.textContent = 'Deleting...';

  try {
    await deleteSource(sourcesState.deleteSourceId);
    showToast('Source deleted successfully', 'success');
    closeDeleteModal();
    loadSources();
  } catch (error) {
    console.error('Failed to delete source:', error);
    showToast(error.message, 'error');
  } finally {
    elements.deleteConfirmBtn.disabled = false;
    elements.deleteConfirmBtn.textContent = 'Delete';
  }
}

async function retrySource(sourceId) {
  // TODO: Implement retry functionality in Phase 4
  showToast('Retry functionality coming soon', 'info');
}

// =============================================================================
// Utility Functions
// =============================================================================

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}

function formatDate(dateString) {
  if (!dateString) return '-';
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function showToast(message, type = 'info') {
  // Use the global showToast from app.js if available
  if (window.showToast) {
    window.showToast(message, type);
  } else {
    console.log(`[${type}] ${message}`);
  }
}

// Debounce for search input
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// =============================================================================
// Event Listeners
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Initial load
  loadSources();

  // Add source buttons
  elements.addSourceBtn?.addEventListener('click', openAddModal);
  elements.emptyAddBtn?.addEventListener('click', openAddModal);

  // Modal close
  elements.addSourceModal?.querySelectorAll('.modal-close, .modal-backdrop').forEach(el => {
    el.addEventListener('click', closeAddModal);
  });
  elements.cancelBtn?.addEventListener('click', closeAddModal);

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Form submit
  elements.submitBtn?.addEventListener('click', handleSubmit);
  elements.urlForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSubmit();
  });

  // File upload
  elements.fileUploadArea?.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput?.addEventListener('change', (e) => handleFileSelect(e.target.files));

  // Drag and drop
  elements.fileUploadArea?.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.fileUploadArea.classList.add('dragover');
  });
  elements.fileUploadArea?.addEventListener('dragleave', () => {
    elements.fileUploadArea.classList.remove('dragover');
  });
  elements.fileUploadArea?.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.fileUploadArea.classList.remove('dragover');
    handleFileSelect(e.dataTransfer.files);
  });

  // Filters
  elements.filterType?.addEventListener('change', () => {
    sourcesState.filters.type = elements.filterType.value;
    sourcesState.currentPage = 1;
    loadSources();
  });
  elements.filterStatus?.addEventListener('change', () => {
    sourcesState.filters.status = elements.filterStatus.value;
    sourcesState.currentPage = 1;
    loadSources();
  });
  elements.filterSearch?.addEventListener('input', debounce(() => {
    sourcesState.filters.search = elements.filterSearch.value;
    sourcesState.currentPage = 1;
    loadSources();
  }, 300));

  // Refresh
  elements.refreshBtn?.addEventListener('click', loadSources);

  // Pagination
  elements.prevPage?.addEventListener('click', () => {
    if (sourcesState.currentPage > 1) {
      sourcesState.currentPage--;
      loadSources();
    }
  });
  elements.nextPage?.addEventListener('click', () => {
    const totalPages = Math.ceil(sourcesState.total / sourcesState.pageSize);
    if (sourcesState.currentPage < totalPages) {
      sourcesState.currentPage++;
      loadSources();
    }
  });

  // Delete modal
  elements.deleteModal?.querySelectorAll('.modal-close, .modal-backdrop').forEach(el => {
    el.addEventListener('click', closeDeleteModal);
  });
  elements.deleteCancelBtn?.addEventListener('click', closeDeleteModal);
  elements.deleteConfirmBtn?.addEventListener('click', handleDelete);

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeAddModal();
      closeDeleteModal();
    }
  });
});
