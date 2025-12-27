# Quickstart: UI Document Manager

**Feature**: UI Document Manager  
**Estimated Implementation Time**: 22-28 hours

## Overview

This feature adds a web-based document management UI to NyRAG, enabling users to:
- Add URLs for crawling and files for processing via the browser
- Monitor background ingestion jobs with real-time progress
- View and manage all indexed data sources
- Access chat conversation history

## Prerequisites

1. **Existing NyRAG Installation**
   ```bash
   pip install -e .  # or pip install nyrag
   ```

2. **Running Vespa Instance**
   ```bash
   # Local development
   docker-compose up -d
   
   # Or set VESPA_URL for cloud deployment
   ```

3. **New Dependencies** (add to pyproject.toml)
   ```bash
   pip install aiosqlite aiofiles
   ```

## Quick Start

### 1. Start the API Server

```bash
# With config file
nyrag --config configs/example.yml

# The API starts at http://localhost:8000
```

### 2. Access the UI

Open your browser to:
- **Chat**: http://localhost:8000/chat
- **Data Sources**: http://localhost:8000/sources
- **Agents/Jobs**: http://localhost:8000/agents

### 3. Add Your First Data Source

**Option A: Add a URL**
1. Go to `/sources`
2. Click "+ Add Source"
3. Enter a URL (e.g., `https://example.com/docs`)
4. Click "Start Ingestion"

**Option B: Upload Files**
1. Go to `/sources`
2. Click "+ Add Source"
3. Switch to "Upload Files" tab
4. Drag & drop PDF, Markdown, or text files
5. Click "Start Ingestion"

### 4. Monitor Progress

1. Go to `/agents`
2. Watch real-time progress bars
3. View logs or stop jobs as needed

### 5. Query Your Content

1. Go to `/chat`
2. Ask questions about your indexed content
3. Previous conversations are saved in the sidebar

## File Structure (New Files)

```
src/nyrag/
├── database.py          # SQLite connection and models
├── sources.py           # Data source management
├── static/
│   ├── app.css          # Shared UI styles
│   ├── app.js           # Navigation/sidebar JS
│   ├── sources.js       # Data Sources page
│   ├── agents.js        # Agents page
│   └── history.js       # Chat history
└── templates/
    ├── base.html        # Base layout with sidebar
    ├── sources.html     # Data Sources page
    └── agents.html      # Agents page
```

## API Endpoints Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sources` | GET | List all data sources |
| `/api/sources/url` | POST | Add URL for crawling |
| `/api/sources/files` | POST | Upload files |
| `/api/sources/{id}` | DELETE | Delete a source |
| `/api/sources/{id}/sync` | POST | Re-index a source |
| `/api/jobs` | GET | List all jobs |
| `/api/jobs/{id}/cancel` | POST | Cancel running job |
| `/api/jobs/{id}/retry` | POST | Retry failed job |
| `/api/conversations` | GET | List chat history |
| `/api/conversations/{id}` | GET | Get conversation with messages |

## Configuration

The UI respects existing YAML configuration:
- Uses configured embedding model and chunk sizes
- Connects to same Vespa instance (local or cloud)
- SQLite database stored in project's output directory

## Troubleshooting

### "Vespa not available" Error

The banner appears when Vespa is unreachable:
```bash
# Start local Vespa
docker-compose up -d

# Verify it's running
curl http://localhost:8080/state/v1/health
```

### File Upload Fails

- Check file size (max 50MB per file)
- Verify file type is supported (.pdf, .md, .txt)
- Check available disk space in uploads directory

### Jobs Stuck in "Queued"

Maximum 2 concurrent jobs allowed. Wait for running jobs to complete or cancel them.

## Development Tips

1. **Run Quality Checks**: `make quality`
2. **Run Tests**: `make test`
3. **Format Code**: `make style`

## Next Steps

After implementing, verify:
1. ✅ Add URL → appears in sources table → job runs → chunks indexed
2. ✅ Upload file → appears in sources table → job runs → chunks indexed
3. ✅ Delete source → chunks removed from Vespa
4. ✅ Chat queries return content from indexed sources
5. ✅ Conversation history persists across sessions
