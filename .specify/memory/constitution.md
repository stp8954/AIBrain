# NyRAG Constitution

A Python-based RAG (Retrieval-Augmented Generation) tool for building search applications by crawling websites or processing documents, deployed to Vespa for hybrid search with an integrated chat UI.

## Core Principles

### I. Configuration-Driven Design

All behavior is controlled through YAML configuration files. No hardcoded business logic.
- Web crawling (`mode: web`) and document processing (`mode: docs`) share unified config structure
- RAG parameters (embedding model, chunk size, overlap) are configurable per project
- Environment variables control deployment mode (`NYRAG_LOCAL`) and credentials

### II. CLI-First Interface

Primary interaction is via the `nyrag` CLI command.
- Single entry point: `nyrag --config <path>` for all operations
- Resume capability: `--resume` flag to skip already processed items
- Clear logging with success/error states via loguru
- Exit codes: 0 for success, 1 for errors

### III. Dual Deployment Modes

Support both local development and cloud production with minimal config changes.
- **Local Mode** (`NYRAG_LOCAL=1`): Docker-based Vespa for development/testing
- **Cloud Mode** (`NYRAG_LOCAL=0`): Vespa Cloud with mTLS authentication
- Same codebase, configuration toggles the target

### IV. Modular Pipeline Architecture

Processing follows a clear, composable pipeline:
1. **Crawl/Ingest**: `crawly/` for web, `markitdown` for documents
2. **Schema**: Vespa schema generation (`schema.py`)
3. **Deploy**: Application package deployment (`deploy.py`)
4. **Feed**: Document chunking and embedding (`feed.py`)
5. **API**: FastAPI server with chat UI (`api.py`)

### V. Observability & Debugging

All operations are logged and traceable.
- Structured logging via `loguru` with configurable levels
- Progress tracking with `tqdm` for long-running operations
- JSONL output format for inspectable intermediate data
- Clear error messages with actionable guidance

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.10+ | Core runtime |
| Web Framework | FastAPI + Uvicorn | Chat API server |
| Vector DB | Vespa (local Docker or Cloud) | Hybrid search & retrieval |
| Embeddings | SentenceTransformers | Text vectorization |
| Document Parsing | markitdown | Multi-format document conversion |
| Web Crawling | Scrapy + BeautifulSoup | Website content extraction |
| LLM Integration | OpenAI SDK via OpenRouter | Answer generation |
| Configuration | Pydantic + PyYAML | Type-safe config validation |
| Logging | Loguru + Rich | Structured logging with formatting |

## Code Quality Standards

### Style & Formatting
- **Line length**: 119 characters max
- **Formatter**: `black` with Python 3.10 target
- **Import sorting**: `isort` with trailing commas
- **Linting**: `flake8` (ignoring E203, W503)
- Run `make style` before commits, `make quality` for CI checks

### Type Safety
- Pydantic models for all configuration and API contracts
- Type hints on all public functions
- Literal types for enum-like fields (e.g., `mode`, `user_agent_type`)

### Testing
- Test directory: `tests/`
- Framework: `pytest` with `--maxfail=1`
- Run via `make test` or `pytest -sv ./src/`

## Project Structure

```
src/nyrag/
├── __init__.py      # Package exports, version
├── cli.py           # Entry point, argument parsing
├── config.py        # Pydantic configuration models
├── process.py       # Main processing orchestration
├── schema.py        # Vespa schema generation
├── deploy.py        # Vespa deployment logic
├── feed.py          # Document chunking and feeding
├── api.py           # FastAPI application
├── utils.py         # Shared utilities
├── logger.py        # Logging configuration
├── crawly/          # Web crawling module
│   ├── crawly.py    # Scrapy spider
│   └── user_agents.py
├── static/          # Chat UI assets
└── templates/       # Jinja2 templates
```

## Configuration Contract

All configs must include:
- `name`: Project identifier (used for Vespa schema naming)
- `mode`: Either `web` or `docs`
- `start_loc`: Starting URL or directory path
- `exclude`: Optional list of patterns to skip

Mode-specific params:
- `crawl_params`: Web crawling settings (robots.txt, user agent, etc.)
- `doc_params`: Document processing settings (recursive, extensions, etc.)
- `rag_params`: Embedding model, chunk size, distance metric

## Governance

- This constitution defines the architectural boundaries for NyRAG
- New features must align with the modular pipeline architecture
- Breaking changes to config schema require version bump and migration docs
- All PRs must pass `make quality` checks

**Version**: 1.0.0 | **Ratified**: 2025-12-23 | **Last Amended**: 2025-12-23
