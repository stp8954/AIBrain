/**
 * Chat page JavaScript for NyRAG UI Document Manager
 * Handles chat functionality with conversation persistence
 */

// =============================================================================
// DOM Elements
// =============================================================================

const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statsEl = document.getElementById("corpus-stats");
const chatTitleEl = document.getElementById("chat-title");

// Settings Modal Elements
const settingsBtn = document.getElementById("settings-btn");
const settingsModal = document.getElementById("settings-modal");
const hitsInput = document.getElementById("hits");
const kInput = document.getElementById("k");
const queryKInput = document.getElementById("query_k");
const saveSettingsBtn = document.getElementById("save-settings");

// History Sidebar Elements
const conversationListEl = document.getElementById("conversation-list");
const newConversationBtn = document.getElementById("new-conversation-btn");
const deleteModal = document.getElementById("delete-conversation-modal");
const confirmDeleteBtn = document.getElementById("confirm-delete-btn");

// =============================================================================
// State
// =============================================================================

const state = {
  currentConversationId: null,
  conversations: [],
  conversationHistory: [], // Messages in current conversation
  pendingDeleteId: null,
};

// Active blog generation jobs for polling
let activeBlogJobs = new Map();

// =============================================================================
// Conversation API Functions
// =============================================================================

async function fetchConversations() {
  try {
    const response = await fetch("/api/conversations?limit=50");
    if (!response.ok) throw new Error("Failed to fetch conversations");
    return await response.json();
  } catch (error) {
    console.error("Error fetching conversations:", error);
    return { conversations: [], total: 0 };
  }
}

async function createConversation(title = null) {
  try {
    const response = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    if (!response.ok) throw new Error("Failed to create conversation");
    return await response.json();
  } catch (error) {
    console.error("Error creating conversation:", error);
    throw error;
  }
}

async function fetchConversation(conversationId) {
  try {
    const response = await fetch(`/api/conversations/${conversationId}`);
    if (!response.ok) throw new Error("Failed to fetch conversation");
    return await response.json();
  } catch (error) {
    console.error("Error fetching conversation:", error);
    throw error;
  }
}

async function deleteConversation(conversationId) {
  try {
    const response = await fetch(`/api/conversations/${conversationId}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new Error("Failed to delete conversation");
    return await response.json();
  } catch (error) {
    console.error("Error deleting conversation:", error);
    throw error;
  }
}

async function addMessage(conversationId, role, content) {
  try {
    const response = await fetch(`/api/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, content }),
    });
    if (!response.ok) throw new Error("Failed to add message");
    return await response.json();
  } catch (error) {
    console.error("Error adding message:", error);
    throw error;
  }
}

// =============================================================================
// Conversation Management
// =============================================================================

async function loadConversations() {
  const data = await fetchConversations();
  state.conversations = data.conversations || [];
  renderConversationList();
}

async function startNewConversation() {
  // Clear chat area
  chatEl.innerHTML = `
    <div class="welcome-message">
      <h2>How can I help you today?</h2>
      <p>I can help you find information from your indexed documents.</p>
    </div>
  `;

  // Reset state
  state.currentConversationId = null;
  state.conversationHistory = [];

  // Update title
  if (chatTitleEl) {
    chatTitleEl.textContent = "New Conversation";
  }

  // Update active state in sidebar
  updateActiveConversation(null);
}

async function loadConversation(conversationId) {
  try {
    const conversation = await fetchConversation(conversationId);

    // Update state
    state.currentConversationId = conversationId;
    state.conversationHistory = (conversation.messages || []).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // Update title
    if (chatTitleEl) {
      chatTitleEl.textContent = conversation.title || "Untitled Conversation";
    }

    // Render messages
    renderConversationMessages(conversation.messages || []);

    // Update active state in sidebar
    updateActiveConversation(conversationId);
  } catch (error) {
    console.error("Error loading conversation:", error);
  }
}

async function ensureConversation() {
  if (!state.currentConversationId) {
    try {
      const conversation = await createConversation();
      state.currentConversationId = conversation.id;
      await loadConversations(); // Refresh list
    } catch (error) {
      console.error("Failed to create conversation:", error);
    }
  }
  return state.currentConversationId;
}

async function handleDeleteConversation(conversationId) {
  state.pendingDeleteId = conversationId;
  openModal(deleteModal);
}

async function confirmDelete() {
  if (!state.pendingDeleteId) return;

  try {
    await deleteConversation(state.pendingDeleteId);

    // If deleted conversation is current, start new
    if (state.currentConversationId === state.pendingDeleteId) {
      await startNewConversation();
    }

    // Refresh list
    await loadConversations();
  } catch (error) {
    console.error("Error deleting conversation:", error);
  } finally {
    state.pendingDeleteId = null;
    closeModal(deleteModal);
  }
}

// =============================================================================
// UI Rendering
// =============================================================================

function renderConversationList() {
  if (!conversationListEl) return;

  if (state.conversations.length === 0) {
    conversationListEl.innerHTML = `
      <div class="empty-conversations">
        <p>No conversations yet</p>
      </div>
    `;
    return;
  }

  conversationListEl.innerHTML = state.conversations
    .map(
      (conv) => `
    <div class="conversation-item ${conv.id === state.currentConversationId ? "active" : ""}" 
         data-conversation-id="${conv.id}">
      <div class="conversation-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
      </div>
      <div class="conversation-info">
        <div class="conversation-title">${escapeHtml(conv.title || "Untitled")}</div>
        <div class="conversation-date">${formatRelativeDate(conv.updated_at)}</div>
      </div>
      <div class="conversation-actions">
        <button class="delete-btn" title="Delete" data-delete-id="${conv.id}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        </button>
      </div>
    </div>
  `
    )
    .join("");

  // Add click handlers
  conversationListEl.querySelectorAll(".conversation-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      // Don't trigger if clicking delete button
      if (e.target.closest(".delete-btn")) return;
      const id = item.dataset.conversationId;
      loadConversation(id);
    });
  });

  conversationListEl.querySelectorAll(".delete-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = btn.dataset.deleteId;
      handleDeleteConversation(id);
    });
  });
}

function updateActiveConversation(conversationId) {
  if (!conversationListEl) return;

  conversationListEl.querySelectorAll(".conversation-item").forEach((item) => {
    if (item.dataset.conversationId === conversationId) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });
}

function renderConversationMessages(messages) {
  // Clear chat and remove welcome message
  chatEl.innerHTML = "";

  messages.forEach((msg) => {
    if (msg.role === "user") {
      appendUserMessage(msg.content);
    } else if (msg.role === "assistant") {
      appendAssistantMessage(msg.content);
    }
  });

  scrollToBottom();
}

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "msg user-msg";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  div.appendChild(bubble);
  chatEl.appendChild(div);
}

function appendAssistantMessage(text) {
  const div = document.createElement("div");
  div.className = "msg assistant-msg";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const assistantText = document.createElement("div");
  assistantText.className = "assistant-text";
  renderMarkdown(assistantText, text);

  bubble.appendChild(assistantText);
  div.appendChild(bubble);
  chatEl.appendChild(div);
}

// =============================================================================
// Utility Functions
// =============================================================================

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatRelativeDate(dateString) {
  if (!dateString) return "";

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
    return "";
  }
}

function openModal(modal) {
  if (modal) {
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
  }
}

function closeModal(modal) {
  if (modal) {
    modal.classList.remove("active");
    document.body.style.overflow = "";
  }
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function renderMarkdown(el, text) {
  if (window.marked) {
    const html = window.marked.parse(text || "", {
      breaks: true,
      mangle: false,
      headerIds: false,
    });
    el.innerHTML = window.DOMPurify ? window.DOMPurify.sanitize(html) : html;
  } else {
    el.textContent = text;
  }
}

// =============================================================================
// Stats
// =============================================================================

async function refreshStats() {
  if (!statsEl) return;
  try {
    const res = await fetch("/stats", { method: "GET" });
    if (!res.ok) {
      statsEl.textContent = "";
      return;
    }
    const data = await res.json();
    const docs = data?.documents;
    const chunks = data?.chunks;
    if (typeof docs === "number" && typeof chunks === "number") {
      statsEl.textContent = `Indexed: ${docs} documents • ${chunks} chunks`;
    } else if (typeof docs === "number") {
      statsEl.textContent = `Indexed: ${docs} documents`;
    } else {
      statsEl.textContent = "";
    }
  } catch (e) {
    statsEl.textContent = "";
  }
}

// =============================================================================
// Blog Job Tracking
// =============================================================================

async function pollBlogJobStatus(jobId, statusEl) {
  const maxPolls = 60;
  let polls = 0;

  const poll = async () => {
    if (polls >= maxPolls) {
      statusEl.innerHTML = `<span class="blog-status error">⚠️ Job timed out. <a href="/api/jobs/${jobId}" target="_blank">Check status</a></span>`;
      activeBlogJobs.delete(jobId);
      return;
    }

    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) {
        statusEl.innerHTML = `<span class="blog-status error">⚠️ Failed to get job status</span>`;
        activeBlogJobs.delete(jobId);
        return;
      }

      const job = await res.json();
      polls++;

      if (job.status === "running" || job.status === "queued") {
        statusEl.innerHTML = `<span class="blog-status pending">⏳ ${job.status === "running" ? "Generating" : "Queued"}... (${polls * 2}s)</span>`;
        setTimeout(poll, 2000);
      } else if (job.status === "complete") {
        const blogId = job.result?.blog_id;
        statusEl.innerHTML = `
          <span class="blog-status success">✅ Blog generated!</span>
          <div class="blog-actions">
            <a href="/api/blog/${blogId}" target="_blank" class="blog-btn">📄 View</a>
            <a href="/api/blog/${blogId}/download" class="blog-btn">⬇️ Download</a>
          </div>
        `;
        activeBlogJobs.delete(jobId);
      } else if (job.status === "failed") {
        statusEl.innerHTML = `<span class="blog-status error">❌ Failed: ${job.error || "Unknown error"}</span>`;
        activeBlogJobs.delete(jobId);
      } else if (job.status === "cancelled") {
        statusEl.innerHTML = `<span class="blog-status error">🚫 Cancelled</span>`;
        activeBlogJobs.delete(jobId);
      }
    } catch (e) {
      statusEl.innerHTML = `<span class="blog-status error">⚠️ Error: ${e.message}</span>`;
      activeBlogJobs.delete(jobId);
    }
  };

  poll();
}

function createBlogJobTracker(jobId, topic) {
  const container = document.createElement("div");
  container.className = "blog-job-tracker";
  container.innerHTML = `
    <div class="blog-job-header">📝 Blog: ${topic}</div>
    <div class="blog-job-status" id="blog-status-${jobId}">
      <span class="blog-status pending">⏳ Starting...</span>
    </div>
  `;

  const statusEl = container.querySelector(`#blog-status-${jobId}`);
  activeBlogJobs.set(jobId, statusEl);
  pollBlogJobStatus(jobId, statusEl);

  return container;
}

// =============================================================================
// Chat Functionality
// =============================================================================

// Auto-resize textarea
inputEl?.addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = this.scrollHeight + "px";
  if (this.value === "") {
    this.style.height = "auto";
  }
});

function append(role, text) {
  // Remove welcome message if it exists
  const welcome = document.querySelector(".welcome-message");
  if (welcome) welcome.remove();

  const div = document.createElement("div");
  div.className = role === "You" ? "msg user-msg" : "msg assistant-msg";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "You") {
    bubble.textContent = text;
  } else {
    bubble.innerHTML = `<div class="assistant-text">${text}</div>`;
  }

  div.appendChild(bubble);
  chatEl.appendChild(div);
  scrollToBottom();
}

function appendChunksCollapsible(chunks) {
  if (!chunks || !chunks.length) return;

  const msgs = document.querySelectorAll(".msg.assistant-msg");
  const lastMsg = msgs[msgs.length - 1];
  if (!lastMsg) return;

  const bubble = lastMsg.querySelector(".bubble");

  const wrap = document.createElement("details");
  wrap.className = "chunks";
  wrap.open = false;

  const listHtml = chunks
    .map(
      (c) =>
        `<details class="chunk-item">
          <summary>${c.loc} <span class="score">(${c.score ? c.score.toFixed(2) : "0.00"})</span></summary>
          <div class="chunk-content">${c.chunk}</div>
        </details>`
    )
    .join("");

  wrap.innerHTML = `<summary>Relevant sources (${chunks.length})</summary><div class="chunk-list">${listHtml}</div>`;

  bubble.appendChild(wrap);
  scrollToBottom();
}

// =============================================================================
// Image Display Functions
// =============================================================================

function appendImagesSection(images, bubble) {
  if (!images || !images.length) return;

  const imagesSection = document.createElement("div");
  imagesSection.className = "images-section";

  const header = document.createElement("div");
  header.className = "images-header";
  header.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
      <circle cx="8.5" cy="8.5" r="1.5"></circle>
      <polyline points="21 15 16 10 5 21"></polyline>
    </svg>
    <span>Related Images (${images.length})</span>
  `;

  const grid = document.createElement("div");
  grid.className = "images-grid";

  images.forEach((img) => {
    const imgContainer = document.createElement("div");
    imgContainer.className = "image-container";

    const imgEl = document.createElement("img");
    imgEl.className = "chat-image";
    imgEl.src = img.url;
    imgEl.alt = img.caption || "Related image";
    imgEl.loading = "lazy";

    // Handle image load error
    imgEl.onerror = function () {
      imgContainer.classList.add("image-error");
      imgContainer.innerHTML = `
        <div class="image-placeholder">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="9" y1="9" x2="15" y2="15"></line>
            <line x1="15" y1="9" x2="9" y2="15"></line>
          </svg>
          <span>Image unavailable</span>
        </div>
      `;
    };

    // Click to open lightbox
    imgEl.onclick = () => openImageLightbox(img);

    imgContainer.appendChild(imgEl);

    // Add caption/metadata if available
    if (img.caption || img.page_number) {
      const caption = document.createElement("div");
      caption.className = "image-caption";
      if (img.page_number) {
        caption.textContent = `Page ${img.page_number}`;
        if (img.caption) caption.textContent += ` - ${img.caption}`;
      } else if (img.caption) {
        caption.textContent = img.caption;
      }
      imgContainer.appendChild(caption);
    }

    grid.appendChild(imgContainer);
  });

  imagesSection.appendChild(header);
  imagesSection.appendChild(grid);
  bubble.appendChild(imagesSection);
  scrollToBottom();
}

function openImageLightbox(img) {
  // Create or get lightbox
  let lightbox = document.getElementById("image-lightbox");
  if (!lightbox) {
    lightbox = document.createElement("div");
    lightbox.id = "image-lightbox";
    lightbox.className = "image-lightbox";
    lightbox.innerHTML = `
      <div class="lightbox-overlay"></div>
      <div class="lightbox-content">
        <button class="lightbox-close" aria-label="Close">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
        <img class="lightbox-image" src="" alt="">
        <div class="lightbox-info"></div>
      </div>
    `;
    document.body.appendChild(lightbox);

    // Close handlers
    lightbox.querySelector(".lightbox-overlay").onclick = closeLightbox;
    lightbox.querySelector(".lightbox-close").onclick = closeLightbox;
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && lightbox.classList.contains("active")) {
        closeLightbox();
      }
    });
  }

  // Set image
  const lightboxImg = lightbox.querySelector(".lightbox-image");
  lightboxImg.src = img.url;
  lightboxImg.alt = img.caption || "Image";

  // Set info
  const infoEl = lightbox.querySelector(".lightbox-info");
  let infoHtml = "";
  if (img.page_number) infoHtml += `<span>Page ${img.page_number}</span>`;
  if (img.caption) infoHtml += `<span>${escapeHtml(img.caption)}</span>`;
  if (img.score) infoHtml += `<span>Relevance: ${(img.score * 100).toFixed(0)}%</span>`;
  infoEl.innerHTML = infoHtml;

  // Show lightbox
  lightbox.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  const lightbox = document.getElementById("image-lightbox");
  if (lightbox) {
    lightbox.classList.remove("active");
    document.body.style.overflow = "";
  }
}

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;

  // Ensure we have a conversation
  await ensureConversation();

  append("You", text);
  inputEl.value = "";
  inputEl.style.height = "auto";

  // Save user message to database
  if (state.currentConversationId) {
    try {
      await addMessage(state.currentConversationId, "user", text);
    } catch (e) {
      console.error("Failed to save user message:", e);
    }
  }

  // Create assistant message placeholder
  const assistantDiv = document.createElement("div");
  assistantDiv.className = "msg assistant-msg";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const meta = document.createElement("div");
  meta.className = "assistant-meta";

  const assistantText = document.createElement("div");
  assistantText.className = "assistant-text";
  assistantText.innerHTML =
    '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';

  bubble.appendChild(meta);
  bubble.appendChild(assistantText);
  assistantDiv.appendChild(bubble);
  chatEl.appendChild(assistantDiv);
  scrollToBottom();

  let assistantMd = "";

  try {
    const res = await fetch("/chat-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: state.conversationHistory,
        hits: parseInt(hitsInput?.value) || 5,
        k: parseInt(kInput?.value) || 3,
        query_k: parseInt(queryKInput?.value) || 3,
        include_images: true,
        max_images: 3,
      }),
    });

    if (!res.ok || !res.body) {
      const err = await res.text();
      assistantText.textContent = `Error: ${err}`;
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const chunksCache = [];
    const queriesCache = [];
    let thinkingContent = "";
    let thinkingEl = null;
    let thinkingBody = null;
    let statusEl = null;
    let isAnswerPhase = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        if (!part.startsWith("data:")) continue;
        const data = part.replace("data:", "").trim();
        if (!data) continue;
        try {
          const evt = JSON.parse(data);
          if (evt.type === "status") {
            if (statusEl) statusEl.remove();
            statusEl = document.createElement("div");
            statusEl.className = "status-line";
            statusEl.textContent = evt.payload;
            meta.appendChild(statusEl);
            scrollToBottom();
            if (evt.payload.includes("Generating answer")) {
              isAnswerPhase = true;
            }
          } else if (evt.type === "thinking") {
            if (!isAnswerPhase) continue;

            if (!thinkingEl) {
              thinkingEl = document.createElement("div");
              thinkingEl.className = "thinking-section";

              const header = document.createElement("div");
              header.className = "thinking-header";
              header.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                Thinking Process
              `;
              header.onclick = () => {
                thinkingBody.classList.toggle("collapsed");
              };

              thinkingBody = document.createElement("div");
              thinkingBody.className = "thinking-content";

              thinkingEl.appendChild(header);
              thinkingEl.appendChild(thinkingBody);

              bubble.insertBefore(thinkingEl, assistantText);
            }
            thinkingContent += evt.payload;
            thinkingBody.textContent = thinkingContent;
            scrollToBottom();
          } else if (evt.type === "queries") {
            queriesCache.splice(0, queriesCache.length, ...evt.payload);

            const details = document.createElement("details");
            details.className = "meta-details";
            const summary = document.createElement("summary");
            summary.textContent = `Queries (${evt.payload.length})`;
            details.appendChild(summary);

            const ul = document.createElement("ul");
            evt.payload.forEach((q) => {
              const li = document.createElement("li");
              li.textContent = q;
              ul.appendChild(li);
            });
            details.appendChild(ul);
            meta.appendChild(details);
            scrollToBottom();

            thinkingEl = null;
            thinkingContent = "";
          } else if (evt.type === "chunks") {
            chunksCache.push(...evt.payload);

            const locs = [...new Set(evt.payload.map((c) => c.loc))];

            const details = document.createElement("details");
            details.className = "meta-details";
            const summary = document.createElement("summary");
            summary.textContent = `Sources (${locs.length})`;
            details.appendChild(summary);

            const ul = document.createElement("ul");
            locs.forEach((loc) => {
              const li = document.createElement("li");
              li.textContent = loc;
              ul.appendChild(li);
            });
            details.appendChild(ul);
            meta.appendChild(details);
            scrollToBottom();

            thinkingEl = null;
            thinkingContent = "";
          } else if (evt.type === "images") {
            // Store images and display them
            if (evt.payload && evt.payload.length > 0) {
              appendImagesSection(evt.payload, bubble);
            }
          } else if (evt.type === "blog_job") {
            const { job_id, topic } = evt.payload;
            const tracker = createBlogJobTracker(job_id, topic);
            bubble.appendChild(tracker);
            scrollToBottom();
          } else if (evt.type === "token") {
            assistantMd += evt.payload;
            renderMarkdown(assistantText, assistantMd);
            scrollToBottom();
          } else if (evt.type === "done") {
            if (statusEl) statusEl.remove();
            if (evt.payload && typeof evt.payload === "string") {
              assistantMd = evt.payload;
            }
            renderMarkdown(assistantText, assistantMd);
            if (chunksCache.length) appendChunksCollapsible(chunksCache);
            scrollToBottom();

            // Add this exchange to conversation history
            state.conversationHistory.push({ role: "user", content: text });
            state.conversationHistory.push({ role: "assistant", content: assistantMd });

            // Save assistant message to database
            if (state.currentConversationId && assistantMd) {
              try {
                await addMessage(state.currentConversationId, "assistant", assistantMd);
                // Refresh conversation list to update title if auto-generated
                await loadConversations();
              } catch (e) {
                console.error("Failed to save assistant message:", e);
              }
            }

            refreshStats();
          }
        } catch (e) {
          continue;
        }
      }
    }
  } catch (e) {
    assistantText.textContent = e?.message || "Request failed";
  }
}

// =============================================================================
// Event Listeners
// =============================================================================

// Send button and Enter key
sendBtn?.addEventListener("click", send);
inputEl?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// New conversation button
newConversationBtn?.addEventListener("click", () => {
  startNewConversation();
});

// Settings modal
settingsBtn?.addEventListener("click", () => openModal(settingsModal));
saveSettingsBtn?.addEventListener("click", () => closeModal(settingsModal));

// Delete confirmation
confirmDeleteBtn?.addEventListener("click", confirmDelete);

// Modal close handlers
document.querySelectorAll(".modal").forEach((modal) => {
  modal.querySelector(".modal-overlay")?.addEventListener("click", () => closeModal(modal));
  modal.querySelector(".modal-close")?.addEventListener("click", () => closeModal(modal));
  modal.querySelector(".modal-cancel")?.addEventListener("click", () => closeModal(modal));
});

// Escape key to close modals
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    document.querySelectorAll(".modal.active").forEach(closeModal);
  }
});

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Load conversations
  loadConversations();

  // Refresh stats
  refreshStats();
});
