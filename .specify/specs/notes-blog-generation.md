# Feature Specification: Notes & Blog Generation

**Feature Branch**: `feat/notes-blog-generation`  
**Created**: 2025-12-23  
**Status**: Draft  
**Input**: User description: "Add notes via UI and generate blog posts for website/Substack using a background agent"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Capture Notes via Web UI (Priority: P1)

As a user, I want to add notes through a web interface so I can quickly capture ideas, research snippets, and thoughts in markdown format with images.

**Why this priority**: Notes capture is the foundation—without notes, there's nothing to generate blogs from. This enables the core data input workflow.

**Independent Test**: Can be fully tested by opening the notes UI, adding a markdown note with an image, and verifying it's saved locally and searchable in Vespa.

**Acceptance Scenarios**:

1. **Given** I'm on the NyRAG UI, **When** I navigate to the notes section, **Then** I see a markdown editor with image upload capability
2. **Given** I'm in the notes editor, **When** I write markdown content and click save, **Then** the note is persisted to local storage AND indexed in Vespa
3. **Given** I have saved notes, **When** I search in the chat UI, **Then** my notes appear in search results alongside crawled content
4. **Given** I'm adding a note, **When** I upload/paste an image, **Then** the image is stored and rendered in the note preview

---

### User Story 2 - Generate Blog Post via Background Agent (Priority: P2)

As a user, I want to delegate blog generation to a background agent from the chat UI so I can request a blog post and continue working while it generates.

**Why this priority**: This is the primary value proposition—turning accumulated notes into publishable content. Depends on P1 for source material.

**Independent Test**: Can be tested by having at least one note saved, then requesting "Generate a blog post about X" in chat, and receiving a Substack-compatible markdown file.

**Acceptance Scenarios**:

1. **Given** I have notes indexed in Vespa, **When** I ask the chat "Generate a blog post about [topic]", **Then** a background agent is spawned to handle generation
2. **Given** a blog generation is in progress, **When** I check status, **Then** I see progress indication (queued/generating/complete)
3. **Given** the agent completes, **When** I view the result, **Then** I receive a Substack-compatible markdown blog post
4. **Given** I request a blog, **When** the agent generates content, **Then** it uses my notes + docs + crawled web content as RAG context

---

### User Story 3 - Choose Blog Template/Style (Priority: P3)

As a user, I want to select a blog template (tutorial, opinion piece, roundup) so the generated content matches my intended style and format.

**Why this priority**: Enhances quality and customization but not required for MVP. Users can get value from P1+P2 alone.

**Independent Test**: Can be tested by requesting a blog with a specific template and verifying the output structure matches the template expectations.

**Acceptance Scenarios**:

1. **Given** I'm requesting a blog generation, **When** I specify a template (e.g., "tutorial"), **Then** the output follows tutorial structure (intro, steps, conclusion)
2. **Given** templates exist, **When** I list available templates, **Then** I see options like: tutorial, opinion, roundup, technical-deep-dive
3. **Given** I have custom requirements, **When** I provide style instructions in my prompt, **Then** the agent incorporates them into generation

---

### User Story 4 - Manage and Browse Notes (Priority: P4)

As a user, I want to browse, edit, and delete my saved notes so I can maintain my knowledge base over time.

**Why this priority**: Important for long-term usability but not blocking for initial value delivery.

**Independent Test**: Can be tested by viewing a list of notes, editing one, deleting another, and verifying changes persist.

**Acceptance Scenarios**:

1. **Given** I have saved notes, **When** I open the notes browser, **Then** I see a list of all notes with titles and dates
2. **Given** I'm viewing a note, **When** I click edit, **Then** I can modify content and save changes
3. **Given** I'm viewing a note, **When** I delete it, **Then** it's removed from local storage AND Vespa index

---

### Edge Cases

- What happens when generating a blog with no relevant notes? → Use crawled/doc content only, warn user if context is thin
- How does system handle image storage limits? → Configurable max image size, reject with clear error if exceeded
- What if Vespa is unavailable during note save? → Save locally, queue for Vespa indexing when available
- What if blog generation times out? → Configurable timeout, partial results saved, user notified
- How are duplicate notes handled? → Allow duplicates, user manages their own content

## Requirements *(mandatory)*

### Functional Requirements

**Notes Capture:**
- **FR-001**: System MUST provide a web UI for adding notes in markdown format
- **FR-002**: System MUST support image upload/paste in notes (stored locally)
- **FR-003**: System MUST persist notes to local JSONL files in `output/<project>/notes/`
- **FR-004**: System MUST index notes in Vespa with embeddings for RAG retrieval
- **FR-005**: System MUST render markdown preview in the notes editor

**Blog Generation:**
- **FR-006**: System MUST spawn a background agent when blog generation is requested
- **FR-007**: System MUST use all RAG sources (notes + docs + web) for blog context
- **FR-008**: System MUST output Substack-compatible markdown format
- **FR-009**: System MUST support configurable blog templates (tutorial, opinion, roundup, etc.)
- **FR-010**: System MUST provide generation status feedback (queued/in-progress/complete/failed)

**Notes Management:**
- **FR-011**: System MUST allow browsing all saved notes with title/date
- **FR-012**: System MUST allow editing existing notes
- **FR-013**: System MUST allow deleting notes (from local + Vespa)

**Integration:**
- **FR-014**: Notes MUST appear in regular chat search results
- **FR-015**: Blog output MUST be saved to `output/<project>/blogs/`

### Key Entities

- **Note**: Markdown content, title (auto-extracted or user-provided), created_at, updated_at, image attachments, tags (optional)
- **BlogPost**: Generated markdown content, source_notes (references), template_used, topic, created_at, status
- **BlogTemplate**: Name, structure definition, example prompts, output format hints

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can create and save a note in under 30 seconds
- **SC-002**: Notes are searchable in Vespa within 5 seconds of saving
- **SC-003**: Blog generation completes within 2 minutes for typical requests
- **SC-004**: Generated blogs require less than 20% manual editing for Substack publishing, measured by:
  - Correct markdown heading hierarchy (H1 title, H2 sections)
  - No placeholder text (e.g., "[INSERT]", "TODO", "...")
  - All source citations properly attributed
  - Word count within ±10% of template target
- **SC-005**: System handles 100+ notes without performance degradation

## Technical Boundaries

### In Scope
- Notes UI as new route in existing FastAPI app (`/notes`)
- Background task queue for blog generation (using asyncio or simple job queue)
- New `mode: notes` in config or extension of existing modes
- Local image storage in `output/<project>/assets/`
- New Vespa document type for notes (separate from crawled content)

### Out of Scope (for this feature)
- Direct Substack API integration (copy/paste workflow is acceptable)
- Real-time collaboration on notes
- Version history for notes
- Mobile-optimized notes UI
- Audio/video attachments

## Decisions

1. **Vespa Schema**: ✅ Separate schema `nyrag<project>notes` for clean separation from crawled/doc content
2. **Max Image Size**: ✅ 5MB default, configurable via `notes_params.max_image_size_mb`
3. **Blog Templates**: ✅ YAML-configurable from the start via `blog_templates/` directory
