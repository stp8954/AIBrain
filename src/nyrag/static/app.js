/**
 * NyRAG App - Shared JavaScript
 * Sidebar navigation and common utilities
 */

// =============================================================================
// System Status Check
// =============================================================================

async function checkSystemStatus() {
  const statusElement = document.getElementById('system-status');
  if (!statusElement) return;

  const dot = statusElement.querySelector('.status-dot');
  const text = statusElement.querySelector('.status-text');

  try {
    const response = await fetch('/api/status');
    if (response.ok) {
      const data = await response.json();
      dot.classList.add('online');
      dot.classList.remove('offline');
      text.textContent = `${data.documents || 0} docs • ${data.chunks || 0} chunks`;

      // Hide vespa banner if online
      const vespaBanner = document.getElementById('vespa-banner');
      if (vespaBanner && data.vespa_available) {
        vespaBanner.style.display = 'none';
      } else if (vespaBanner && !data.vespa_available) {
        vespaBanner.style.display = 'flex';
      }
    } else {
      throw new Error('Status check failed');
    }
  } catch (error) {
    dot.classList.add('offline');
    dot.classList.remove('online');
    text.textContent = 'Offline';

    // Show vespa banner if offline
    const vespaBanner = document.getElementById('vespa-banner');
    if (vespaBanner) {
      vespaBanner.style.display = 'flex';
    }
  }
}

// Check status on load and periodically
document.addEventListener('DOMContentLoaded', () => {
  checkSystemStatus();
  // Check every 30 seconds
  setInterval(checkSystemStatus, 30000);
});

// =============================================================================
// Mobile Sidebar Toggle
// =============================================================================

function toggleSidebar() {
  const sidebar = document.querySelector('.app-sidebar');
  if (sidebar) {
    sidebar.classList.toggle('open');
  }
}

// Close sidebar when clicking outside on mobile
document.addEventListener('click', (e) => {
  const sidebar = document.querySelector('.app-sidebar');
  const toggle = document.querySelector('.mobile-menu-toggle');

  if (sidebar && sidebar.classList.contains('open')) {
    if (!sidebar.contains(e.target) && (!toggle || !toggle.contains(e.target))) {
      sidebar.classList.remove('open');
    }
  }
});

// =============================================================================
// Modal Management
// =============================================================================

class Modal {
  constructor(overlayId) {
    this.overlay = document.getElementById(overlayId);
    if (!this.overlay) return;

    this.modal = this.overlay.querySelector('.modal');

    // Close on overlay click
    this.overlay.addEventListener('click', (e) => {
      if (e.target === this.overlay) {
        this.close();
      }
    });

    // Close button
    const closeBtn = this.overlay.querySelector('.modal-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => this.close());
    }

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen()) {
        this.close();
      }
    });
  }

  open() {
    if (this.overlay) {
      this.overlay.classList.add('open');
      document.body.style.overflow = 'hidden';
    }
  }

  close() {
    if (this.overlay) {
      this.overlay.classList.remove('open');
      document.body.style.overflow = '';
    }
  }

  isOpen() {
    return this.overlay && this.overlay.classList.contains('open');
  }
}

// =============================================================================
// Tab Management
// =============================================================================

function initTabs(container) {
  const tabsContainer = typeof container === 'string'
    ? document.querySelector(container)
    : container;

  if (!tabsContainer) return;

  const tabs = tabsContainer.querySelectorAll('.tab');
  const panels = tabsContainer.parentElement.querySelectorAll('.tab-panel');

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const targetId = tab.dataset.tab;

      // Update tabs
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      // Update panels
      panels.forEach(panel => {
        if (panel.id === targetId) {
          panel.classList.add('active');
        } else {
          panel.classList.remove('active');
        }
      });
    });
  });
}

// Auto-init tabs on page load
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tabs').forEach(initTabs);
});

// =============================================================================
// Toast Notifications
// =============================================================================

const toastContainer = document.createElement('div');
toastContainer.className = 'toast-container';
toastContainer.style.cssText = `
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 2000;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;
document.body.appendChild(toastContainer);

function showToast(message, type = 'info', duration = 4000) {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.style.cssText = `
    padding: 12px 16px;
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    font-size: 14px;
    box-shadow: var(--shadow-lg);
    display: flex;
    align-items: center;
    gap: 8px;
    animation: slideIn 0.2s ease;
  `;

  // Add icon based on type
  const icons = {
    success: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>',
    error: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    warning: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#eab308" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    info: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
  };

  toast.innerHTML = `${icons[type] || icons.info}<span>${message}</span>`;
  toastContainer.appendChild(toast);

  // Auto remove
  setTimeout(() => {
    toast.style.animation = 'slideOut 0.2s ease';
    setTimeout(() => toast.remove(), 200);
  }, duration);
}

// Add toast animations
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  @keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
  }
`;
document.head.appendChild(style);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format a date for display
 */
function formatDate(dateStr) {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
  });
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * Debounce function calls
 */
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

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// =============================================================================
// Confirmation Dialog
// =============================================================================

function confirm(message, options = {}) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay open';
    overlay.innerHTML = `
      <div class="modal" style="max-width: 400px;">
        <div class="modal-header">
          <h2>${options.title || 'Confirm'}</h2>
          <button class="modal-close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
        <div class="modal-body">
          <p style="margin: 0;">${message}</p>
        </div>
        <div class="modal-footer">
          <button class="btn btn-secondary cancel-btn">${options.cancelText || 'Cancel'}</button>
          <button class="btn ${options.danger ? 'btn-danger' : 'btn-primary'} confirm-btn">
            ${options.confirmText || 'Confirm'}
          </button>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);
    document.body.style.overflow = 'hidden';

    const cleanup = (result) => {
      overlay.classList.remove('open');
      setTimeout(() => {
        overlay.remove();
        document.body.style.overflow = '';
      }, 200);
      resolve(result);
    };

    overlay.querySelector('.modal-close').addEventListener('click', () => cleanup(false));
    overlay.querySelector('.cancel-btn').addEventListener('click', () => cleanup(false));
    overlay.querySelector('.confirm-btn').addEventListener('click', () => cleanup(true));
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) cleanup(false);
    });
  });
}

// =============================================================================
// Export for use in other modules
// =============================================================================

window.NyRAG = {
  Modal,
  initTabs,
  showToast,
  formatDate,
  formatFileSize,
  debounce,
  escapeHtml,
  confirm,
  checkSystemStatus
};
