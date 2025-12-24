"""Notes module for NyRAG.

This module provides functionality for capturing, storing, and retrieving
markdown notes with image support. Notes are stored locally in JSONL format
and indexed in Vespa for semantic search integration with the RAG pipeline.

Key features:
- Create, read, update, delete notes
- Image attachment support with local storage
- Vespa indexing with embeddings for semantic search
- Integration with existing chat search results
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from nyrag.config import Config
from nyrag.logger import get_logger
from nyrag.utils import DEFAULT_EMBEDDING_MODEL, get_vespa_tls_config, make_vespa_client, resolve_vespa_port


logger = get_logger("notes")

# Module-level embedding model (lazy loaded)
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Get or create the embedding model singleton."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


class Note(BaseModel):
    """Pydantic model representing a note."""

    id: str = Field(..., description="Unique identifier for the note")
    title: str = Field(..., description="Note title")
    content: str = Field(..., description="Markdown content of the note")
    images: List[str] = Field(default_factory=list, description="List of image paths")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Optional tags for categorization")


# Placeholder functions - to be implemented in Phase 3 (US1)


def _get_notes_path(config: Config) -> Path:
    """Get the path to the notes JSONL file."""
    output_path = config.get_output_path()
    notes_storage = config.notes_params.storage_path if config.notes_params else "notes"
    return output_path / notes_storage / "notes.jsonl"


def _get_assets_path(config: Config, note_id: str) -> Path:
    """Get the path to store assets for a note."""
    output_path = config.get_output_path()
    return output_path / "assets" / note_id


def save_note_local(note: Note, config: Config) -> None:
    """Save a note to local JSONL storage.

    Args:
        note: The note to save.
        config: NyRAG configuration.
    """
    notes_path = _get_notes_path(config)
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize note to JSON line
    note_data = note.model_dump()
    # Convert datetime to ISO format strings
    note_data["created_at"] = note.created_at.isoformat()
    note_data["updated_at"] = note.updated_at.isoformat()

    with open(notes_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(note_data) + "\n")

    logger.info(f"Saved note {note.id} to {notes_path}")


def save_image(image_data: bytes, note_id: str, filename: str, config: Config) -> str:
    """Save an image attachment for a note.

    Args:
        image_data: Raw image bytes.
        note_id: ID of the note this image belongs to.
        filename: Original filename of the image.
        config: NyRAG configuration.

    Returns:
        Relative path to the saved image.

    Raises:
        ValueError: If image exceeds max size limit.
    """
    # Validate image size
    max_size_mb = config.notes_params.max_image_size_mb if config.notes_params else 5.0
    size_mb = len(image_data) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Image size {size_mb:.2f}MB exceeds limit of {max_size_mb}MB")

    # Create assets directory
    assets_path = _get_assets_path(config, note_id)
    assets_path.mkdir(parents=True, exist_ok=True)

    # Sanitize filename and save
    safe_filename = "".join(c for c in filename if c.isalnum() or c in ".-_")
    if not safe_filename:
        safe_filename = f"image_{uuid.uuid4().hex[:8]}"

    # Add extension if missing
    if "." not in safe_filename:
        safe_filename += ".png"

    file_path = assets_path / safe_filename
    with open(file_path, "wb") as f:
        f.write(image_data)

    # Return relative URL path
    relative_path = f"/assets/{note_id}/{safe_filename}"
    logger.info(f"Saved image {filename} to {file_path}")
    return relative_path


def index_note_vespa(note: Note, config: Config) -> bool:
    """Index a note in Vespa for semantic search.

    Args:
        note: The note to index.
        config: NyRAG configuration.

    Returns:
        True if indexing succeeded, False otherwise.
    """
    try:
        # Get embedding model
        rag_params = config.rag_params or {}
        model_name = rag_params.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        model = _get_embedding_model(model_name)

        # Generate embedding from title + content
        text_to_embed = f"{note.title}\n\n{note.content}"
        embedding = model.encode(text_to_embed, convert_to_numpy=True).tolist()

        # Connect to Vespa
        vespa_url = os.getenv("VESPA_URL", "http://localhost")
        vespa_port = resolve_vespa_port(vespa_url)
        cert_path, key_path, ca_cert, verify = get_vespa_tls_config()
        vespa_app = make_vespa_client(vespa_url, vespa_port, cert_path, key_path, ca_cert, verify)

        # Prepare document fields for notes schema
        schema_name = f"nyrag{config.name.replace('-', '').replace('_', '').lower()}notes"
        fields: Dict[str, Any] = {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "tags": note.tags,
            "created_at": int(note.created_at.timestamp() * 1000),  # ms epoch
            "updated_at": int(note.updated_at.timestamp() * 1000),
            "content_embedding": {"values": embedding},
        }

        # Feed to Vespa
        response = vespa_app.feed_data_point(
            schema=schema_name,
            data_id=note.id,
            fields=fields,
        )

        status_code = getattr(response, "status_code", 200)
        if status_code >= 400:
            logger.error(f"Vespa indexing failed for note {note.id}: {getattr(response, 'json', response)}")
            return False

        logger.success(f"Indexed note {note.id} in Vespa schema {schema_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to index note {note.id} in Vespa: {e}")
        # Note: In production, should queue for retry
        return False


def save_note(note: Note, config: Config) -> Note:
    """Orchestrator function to save a note locally and index in Vespa.

    Args:
        note: The note to save.
        config: NyRAG configuration.

    Returns:
        The saved note with generated ID.
    """
    # Generate ID if not set
    if not note.id or note.id == "":
        note.id = str(uuid.uuid4())

    # Ensure timestamps
    now = datetime.utcnow()
    if not note.created_at:
        note.created_at = now
    note.updated_at = now

    # Save locally first
    save_note_local(note, config)

    # Index in Vespa (async, don't block on failure)
    indexed = index_note_vespa(note, config)
    if not indexed:
        logger.warning(f"Note {note.id} saved locally but Vespa indexing failed")

    return note


def get_note(note_id: str, config: Config) -> Optional[Note]:
    """Retrieve a single note by ID.

    Args:
        note_id: The ID of the note to retrieve.
        config: NyRAG configuration.

    Returns:
        The note if found, None otherwise.
    """
    notes_path = _get_notes_path(config)
    if not notes_path.exists():
        return None

    with open(notes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                note_data = json.loads(line)
                if note_data.get("id") == note_id:
                    # Parse datetime strings back to datetime objects
                    if isinstance(note_data.get("created_at"), str):
                        note_data["created_at"] = datetime.fromisoformat(note_data["created_at"])
                    if isinstance(note_data.get("updated_at"), str):
                        note_data["updated_at"] = datetime.fromisoformat(note_data["updated_at"])
                    return Note(**note_data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON line in notes file: {line[:50]}...")
                continue

    return None


def search_notes(query: str, config: Config, limit: int = 10) -> List[Note]:
    """Search notes using Vespa semantic search.

    Args:
        query: Search query string.
        config: NyRAG configuration.
        limit: Maximum number of results to return.

    Returns:
        List of matching notes.
    """
    try:
        # Get embedding model
        rag_params = config.rag_params or {}
        model_name = rag_params.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        model = _get_embedding_model(model_name)

        # Generate query embedding
        embedding = model.encode(query, convert_to_numpy=True).tolist()

        # Connect to Vespa
        vespa_url = os.getenv("VESPA_URL", "http://localhost")
        vespa_port = resolve_vespa_port(vespa_url)
        cert_path, key_path, ca_cert, verify = get_vespa_tls_config()
        vespa_app = make_vespa_client(vespa_url, vespa_port, cert_path, key_path, ca_cert, verify)

        # Query notes schema
        schema_name = f"nyrag{config.name.replace('-', '').replace('_', '').lower()}notes"
        body = {
            "yql": "select * from sources * where true",
            "hits": limit,
            "ranking.profile": "hybrid",
            "input.query(embedding)": embedding,
        }

        response = vespa_app.query(body=body, schema=schema_name)
        status_code = getattr(response, "status_code", 200)
        if status_code >= 400:
            logger.error(f"Vespa notes search failed: {getattr(response, 'json', response)}")
            return []

        # Parse results
        notes: List[Note] = []
        hits = response.json.get("root", {}).get("children", []) or []
        for hit in hits:
            fields = hit.get("fields", {})
            try:
                # Convert epoch ms back to datetime
                created_at = datetime.fromtimestamp(fields.get("created_at", 0) / 1000)
                updated_at = datetime.fromtimestamp(fields.get("updated_at", 0) / 1000)

                note = Note(
                    id=fields.get("id", ""),
                    title=fields.get("title", ""),
                    content=fields.get("content", ""),
                    tags=fields.get("tags", []),
                    created_at=created_at,
                    updated_at=updated_at,
                    images=[],  # Images not stored in Vespa
                )
                notes.append(note)
            except Exception as e:
                logger.warning(f"Failed to parse note from Vespa hit: {e}")
                continue

        logger.info(f"Found {len(notes)} notes matching query: {query[:50]}...")
        return notes

    except Exception as e:
        logger.error(f"Notes search failed: {e}")
        return []


def list_notes(config: Config) -> List[Note]:
    """List all notes sorted by creation date (newest first).

    Args:
        config: NyRAG configuration.

    Returns:
        List of all notes.
    """
    notes_path = _get_notes_path(config)
    if not notes_path.exists():
        return []

    notes: List[Note] = []
    with open(notes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                note_data = json.loads(line)
                # Parse datetime strings back to datetime objects
                if isinstance(note_data.get("created_at"), str):
                    note_data["created_at"] = datetime.fromisoformat(note_data["created_at"])
                if isinstance(note_data.get("updated_at"), str):
                    note_data["updated_at"] = datetime.fromisoformat(note_data["updated_at"])
                notes.append(Note(**note_data))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse note: {e}")
                continue

    # Sort by created_at descending (newest first)
    notes.sort(key=lambda n: n.created_at, reverse=True)
    logger.info(f"Listed {len(notes)} notes")
    return notes


def update_note(note: Note, config: Config) -> Note:
    """Update an existing note.

    Args:
        note: The note with updated content.
        config: NyRAG configuration.

    Returns:
        The updated note.

    Raises:
        ValueError: If the note doesn't exist.
    """
    notes_path = _get_notes_path(config)
    if not notes_path.exists():
        raise ValueError(f"Note not found: {note.id}")

    # Read all notes
    all_notes: List[Dict[str, Any]] = []
    found = False

    with open(notes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                note_data = json.loads(line)
                if note_data.get("id") == note.id:
                    # Update this note
                    found = True
                    note.updated_at = datetime.utcnow()
                    updated_data = note.model_dump()
                    updated_data["created_at"] = note.created_at.isoformat()
                    updated_data["updated_at"] = note.updated_at.isoformat()
                    all_notes.append(updated_data)
                else:
                    all_notes.append(note_data)
            except json.JSONDecodeError:
                continue

    if not found:
        raise ValueError(f"Note not found: {note.id}")

    # Rewrite the file with updated notes
    with open(notes_path, "w", encoding="utf-8") as f:
        for note_data in all_notes:
            f.write(json.dumps(note_data) + "\n")

    logger.info(f"Updated note {note.id}")

    # Re-index in Vespa with new embedding
    indexed = index_note_vespa(note, config)
    if not indexed:
        logger.warning(f"Note {note.id} updated locally but Vespa re-indexing failed")

    return note


def delete_note(note_id: str, config: Config) -> bool:
    """Delete a note and its associated images.

    Args:
        note_id: The ID of the note to delete.
        config: NyRAG configuration.

    Returns:
        True if deletion succeeded, False otherwise.
    """
    notes_path = _get_notes_path(config)
    if not notes_path.exists():
        return False

    # Read all notes, excluding the one to delete
    all_notes: List[Dict[str, Any]] = []
    found = False

    with open(notes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                note_data = json.loads(line)
                if note_data.get("id") == note_id:
                    found = True
                else:
                    all_notes.append(note_data)
            except json.JSONDecodeError:
                continue

    if not found:
        logger.warning(f"Note not found for deletion: {note_id}")
        return False

    # Rewrite the file without the deleted note
    with open(notes_path, "w", encoding="utf-8") as f:
        for note_data in all_notes:
            f.write(json.dumps(note_data) + "\n")

    logger.info(f"Deleted note {note_id} from local storage")

    # Delete from Vespa
    try:
        vespa_url = os.getenv("VESPA_URL", "http://localhost")
        vespa_port = resolve_vespa_port(vespa_url)
        cert_path, key_path, ca_cert, verify = get_vespa_tls_config()
        vespa_app = make_vespa_client(vespa_url, vespa_port, cert_path, key_path, ca_cert, verify)

        schema_name = f"nyrag{config.name.replace('-', '').replace('_', '').lower()}notes"
        vespa_app.delete_data(schema=schema_name, data_id=note_id)
        logger.info(f"Deleted note {note_id} from Vespa")
    except Exception as e:
        logger.warning(f"Failed to delete note {note_id} from Vespa: {e}")

    # Delete associated images
    assets_path = _get_assets_path(config, note_id)
    if assets_path.exists():
        import shutil

        try:
            shutil.rmtree(assets_path)
            logger.info(f"Deleted assets for note {note_id}")
        except Exception as e:
            logger.warning(f"Failed to delete assets for note {note_id}: {e}")

    return True
