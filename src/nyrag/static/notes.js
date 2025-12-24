/**
 * Notes UI JavaScript for NyRAG
 *
 * Provides the client-side logic for the markdown notes editor including:
 * - EasyMDE markdown editor integration
 * - Image upload (drag-drop and paste)
 * - Note save/edit/delete operations
 * - Tags management
 *
 * Dependencies: EasyMDE (loaded via CDN in notes.html)
 */

// DOM Elements
let editor = null;
let titleInput = null;
let tagsContainer = null;
let tagInput = null;
let saveBtn = null;
let statusEl = null;
let notesListEl = null;
let newNoteBtn = null;
let deleteBtn = null;

// Current state
let currentNoteId = null;
let currentTags = [];
let isEditing = false;
let notesList = [];

/**
 * Initialize the notes page.
 */
function initNotesPage() {
  // Get DOM elements
  titleInput = document.getElementById("note-title");
  tagsContainer = document.getElementById("tags-container");
  tagInput = document.getElementById("tag-input");
  saveBtn = document.getElementById("save-btn");
  statusEl = document.getElementById("status-message");
  notesListEl = document.getElementById("notes-list");
  newNoteBtn = document.getElementById("new-note-btn");
  deleteBtn = document.getElementById("delete-btn");

  // Initialize EasyMDE editor
  initEditor();

  // Initialize event listeners
  initEventListeners();

  // Load notes list
  loadNotesList();

  console.log("Notes page initialized");
}

/**
 * Initialize the EasyMDE markdown editor.
 */
function initEditor() {
  const textArea = document.getElementById("note-content");
  if (!textArea) {
    console.error("Note content textarea not found");
    return;
  }

  // Check if EasyMDE is available
  if (typeof EasyMDE === "undefined") {
    console.error("EasyMDE not loaded");
    return;
  }

  editor = new EasyMDE({
    element: textArea,
    spellChecker: false,
    autosave: {
      enabled: false,
    },
    placeholder: "Write your note in markdown...",
    toolbar: [
      "bold",
      "italic",
      "heading",
      "|",
      "quote",
      "unordered-list",
      "ordered-list",
      "|",
      "link",
      "image",
      "|",
      "preview",
      "side-by-side",
      "fullscreen",
      "|",
      "guide",
    ],
    status: false,
    renderingConfig: {
      singleLineBreaks: false,
      codeSyntaxHighlighting: true,
    },
  });

  // Set up paste handler for images
  editor.codemirror.on("paste", handlePaste);
}

/**
 * Initialize event listeners.
 */
function initEventListeners() {
  // Save button
  if (saveBtn) {
    saveBtn.addEventListener("click", handleSave);
  }

  // New note button
  if (newNoteBtn) {
    newNoteBtn.addEventListener("click", () => {
      clearEditor();
      highlightSelectedNote(null);
    });
  }

  // Delete button
  if (deleteBtn) {
    deleteBtn.addEventListener("click", handleDelete);
  }

  // Tag input
  if (tagInput) {
    tagInput.addEventListener("keydown", handleTagInput);
  }

  // Set up drag-drop for images
  const editorEl = document.querySelector(".EasyMDEContainer");
  if (editorEl) {
    editorEl.addEventListener("dragover", handleDragOver);
    editorEl.addEventListener("dragleave", handleDragLeave);
    editorEl.addEventListener("drop", handleDrop);
  }
}

/**
 * Handle paste events for image upload.
 * @param {ClipboardEvent} e - The paste event.
 */
async function handlePaste(cm, e) {
  const items = e.clipboardData?.items;
  if (!items) return;

  for (const item of items) {
    if (item.type.startsWith("image/")) {
      e.preventDefault();
      const file = item.getAsFile();
      if (file) {
        await uploadImage(file);
      }
      break;
    }
  }
}

/**
 * Handle dragover event.
 * @param {DragEvent} e - The drag event.
 */
function handleDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add("drag-over");
}

/**
 * Handle dragleave event.
 * @param {DragEvent} e - The drag event.
 */
function handleDragLeave(e) {
  e.preventDefault();
  e.currentTarget.classList.remove("drag-over");
}

/**
 * Handle drop event for image upload.
 * @param {DragEvent} e - The drop event.
 */
async function handleDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove("drag-over");

  const files = e.dataTransfer?.files;
  if (!files) return;

  for (const file of files) {
    if (file.type.startsWith("image/")) {
      await uploadImage(file);
    }
  }
}

/**
 * Upload an image file.
 * @param {File} file - The image file to upload.
 */
async function uploadImage(file) {
  showStatus("Uploading image...", "info");

  try {
    const formData = new FormData();
    formData.append("file", file);

    // Use current note ID if editing, otherwise use a temp ID
    const noteId = currentNoteId || "temp-" + Date.now();
    formData.append("note_id", noteId);

    const response = await fetch("/api/notes/upload-image", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Upload failed");
    }

    const data = await response.json();

    // Insert markdown image link at cursor
    const imageMarkdown = `![${file.name}](${data.url})`;
    if (editor) {
      const cm = editor.codemirror;
      const cursor = cm.getCursor();
      cm.replaceRange(imageMarkdown + "\n", cursor);
    }

    showStatus("Image uploaded successfully", "success");
  } catch (error) {
    console.error("Image upload error:", error);
    showStatus("Failed to upload image: " + error.message, "error");
  }
}

/**
 * Handle tag input keydown.
 * @param {KeyboardEvent} e - The keyboard event.
 */
function handleTagInput(e) {
  if (e.key === "Enter" || e.key === ",") {
    e.preventDefault();
    const tag = tagInput.value.trim().toLowerCase();
    if (tag && !currentTags.includes(tag)) {
      addTag(tag);
    }
    tagInput.value = "";
  }
}

/**
 * Add a tag to the current note.
 * @param {string} tag - The tag to add.
 */
function addTag(tag) {
  currentTags.push(tag);
  renderTags();
}

/**
 * Remove a tag from the current note.
 * @param {string} tag - The tag to remove.
 */
function removeTag(tag) {
  currentTags = currentTags.filter((t) => t !== tag);
  renderTags();
}

/**
 * Render the tags in the UI.
 */
function renderTags() {
  if (!tagsContainer) return;

  const tagsHtml = currentTags
    .map(
      (tag) => `
    <span class="tag">
      ${escapeHtml(tag)}
      <span class="tag-remove" onclick="removeTag('${escapeHtml(tag)}')">&times;</span>
    </span>
  `
    )
    .join("");

  // Keep the input at the end
  tagsContainer.innerHTML =
    tagsHtml +
    '<input type="text" id="tag-input" class="tag-input" placeholder="Add tag...">';

  // Re-bind the input
  tagInput = document.getElementById("tag-input");
  if (tagInput) {
    tagInput.addEventListener("keydown", handleTagInput);
  }
}

/**
 * Handle save button click.
 */
async function handleSave() {
  if (!editor || !titleInput) return;

  const title = titleInput.value.trim();
  const content = editor.value();

  if (!title) {
    showStatus("Please enter a title", "error");
    return;
  }

  if (!content) {
    showStatus("Please enter some content", "error");
    return;
  }

  saveBtn.disabled = true;
  saveBtn.innerHTML = '<span class="spinner"></span> Saving...';

  try {
    const payload = {
      title: title,
      content: content,
      tags: currentTags,
    };

    const url = isEditing ? `/api/notes/${currentNoteId}` : "/api/notes";
    const method = isEditing ? "PUT" : "POST";

    const response = await fetch(url, {
      method: method,
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Save failed");
    }

    const savedNote = await response.json();
    currentNoteId = savedNote.id;
    isEditing = true;

    showStatus("Note saved successfully!", "success");
    await loadNotesList();
    highlightSelectedNote(currentNoteId);
  } catch (error) {
    console.error("Save error:", error);
    showStatus("Failed to save: " + error.message, "error");
  } finally {
    saveBtn.disabled = false;
    saveBtn.innerHTML = "Save Note";
  }
}

/**
 * Show a status message.
 * @param {string} message - The message to show.
 * @param {string} type - The message type (success, error, info).
 */
function showStatus(message, type) {
  if (!statusEl) return;

  statusEl.textContent = message;
  statusEl.className = "status-message " + type;
  statusEl.style.display = "block";

  // Auto-hide success messages
  if (type === "success") {
    setTimeout(() => {
      statusEl.style.display = "none";
    }, 3000);
  }
}

/**
 * Escape HTML to prevent XSS.
 * @param {string} str - The string to escape.
 * @returns {string} The escaped string.
 */
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Clear the editor for a new note.
 */
function clearEditor() {
  if (editor) {
    editor.value("");
  }
  if (titleInput) {
    titleInput.value = "";
  }
  currentNoteId = null;
  currentTags = [];
  isEditing = false;
  renderTags();
  if (statusEl) {
    statusEl.style.display = "none";
  }
}

/**
 * Load a note for editing.
 * @param {string} noteId - The ID of the note to load.
 */
async function loadNote(noteId) {
  try {
    const response = await fetch(`/api/notes/${noteId}`);
    if (!response.ok) {
      throw new Error("Failed to load note");
    }

    const note = await response.json();

    if (editor) {
      editor.value(note.content || "");
    }
    if (titleInput) {
      titleInput.value = note.title || "";
    }

    currentNoteId = note.id;
    currentTags = note.tags || [];
    isEditing = true;
    renderTags();
    highlightSelectedNote(noteId);

    showStatus("Note loaded", "info");
  } catch (error) {
    console.error("Load error:", error);
    showStatus("Failed to load note: " + error.message, "error");
  }
}

/**
 * Load the list of notes from the API.
 */
async function loadNotesList() {
  try {
    const response = await fetch("/api/notes");
    if (!response.ok) {
      throw new Error("Failed to load notes");
    }

    notesList = await response.json();
    renderNotesList();
  } catch (error) {
    console.error("Load notes list error:", error);
    showStatus("Failed to load notes list: " + error.message, "error");
  }
}

/**
 * Render the notes list in the sidebar.
 */
function renderNotesList() {
  if (!notesListEl) return;

  if (notesList.length === 0) {
    notesListEl.innerHTML = '<div class="notes-list-empty">No notes yet. Create your first note!</div>';
    return;
  }

  const listHtml = notesList
    .map((note) => {
      const date = new Date(note.created_at).toLocaleDateString();
      const isSelected = note.id === currentNoteId;
      const tagsHtml = (note.tags || [])
        .slice(0, 3)
        .map((t) => `<span class="note-item-tag">${escapeHtml(t)}</span>`)
        .join("");
      const moreTags = (note.tags || []).length > 3 ? `<span class="note-item-tag">+${(note.tags || []).length - 3}</span>` : "";

      return `
        <div class="note-item ${isSelected ? 'selected' : ''}" data-note-id="${note.id}" onclick="loadNote('${note.id}')">
          <div class="note-item-title">${escapeHtml(note.title)}</div>
          <div class="note-item-date">${date}</div>
          <div class="note-item-tags">${tagsHtml}${moreTags}</div>
        </div>
      `;
    })
    .join("");

  notesListEl.innerHTML = listHtml;
}

/**
 * Highlight the selected note in the list.
 * @param {string|null} noteId - The ID of the note to highlight, or null to clear.
 */
function highlightSelectedNote(noteId) {
  if (!notesListEl) return;

  // Remove selection from all items
  notesListEl.querySelectorAll(".note-item").forEach((item) => {
    item.classList.remove("selected");
  });

  // Add selection to the current item
  if (noteId) {
    const selected = notesListEl.querySelector(`[data-note-id="${noteId}"]`);
    if (selected) {
      selected.classList.add("selected");
    }
  }
}

/**
 * Handle delete button click.
 */
async function handleDelete() {
  if (!currentNoteId) {
    showStatus("No note selected to delete", "error");
    return;
  }

  // Find the current note's title
  const currentNote = notesList.find((n) => n.id === currentNoteId);
  const noteTitle = currentNote ? currentNote.title : "this note";

  const confirmed = confirm(`Are you sure you want to delete "${noteTitle}"?\n\nThis action cannot be undone.`);
  if (!confirmed) {
    return;
  }

  deleteBtn.disabled = true;
  deleteBtn.innerHTML = '<span class="spinner"></span> Deleting...';

  try {
    const response = await fetch(`/api/notes/${currentNoteId}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Delete failed");
    }

    showStatus("Note deleted successfully", "success");
    clearEditor();
    await loadNotesList();
  } catch (error) {
    console.error("Delete error:", error);
    showStatus("Failed to delete: " + error.message, "error");
  } finally {
    deleteBtn.disabled = false;
    deleteBtn.innerHTML = "Delete";
  }
}

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", initNotesPage);
