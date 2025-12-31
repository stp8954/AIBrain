"""
LangChain-based ingestion pipeline for NyRAG.

This module provides pluggable chunking strategies, embedding models,
and unified document loaders for the NyRAG ingestion pipeline.

Modules:
    - base: Pipeline orchestrator (IngestionPipeline)
    - chunking: Text splitting strategies (fixed, recursive, semantic)
    - embeddings: Text and image embedding wrappers
    - loaders: Unified document loaders (URL, PDF, markdown, text)
    - validation: Configuration validation utilities
"""

# Chunking exports
from .chunking import (
    ChunkingStrategy,
    ChunkMetadata,
    DocumentChunk,
    FixedChunkingStrategy,
    RecursiveChunkingStrategy,
    SemanticChunkingStrategy,
    chunk_document,
    get_chunking_strategy,
)

# Embeddings exports
from .embeddings import (
    BaseEmbedder,
    ImageEmbedder,
    TextEmbedder,
    get_image_embedder,
    get_text_embedder,
)

# Base pipeline exports
from .base import ImageDocument, IngestionPipeline

# Loader exports
from .loaders import DocumentLoader, ExtractedImage, LoadedDocument, get_loader_for_source

# Validation exports
from .validation import (
    PipelineValidationError,
    ValidationResult,
    validate_chunking_strategy,
    validate_embedding_dimensions,
    validate_embedding_model,
    validate_image_embedding,
    validate_pipeline_config,
    validate_and_raise,
)

__all__ = [
    # Chunking
    "ChunkingStrategy",
    "FixedChunkingStrategy",
    "RecursiveChunkingStrategy",
    "SemanticChunkingStrategy",
    "DocumentChunk",
    "ChunkMetadata",
    "get_chunking_strategy",
    "chunk_document",
    # Embeddings
    "BaseEmbedder",
    "TextEmbedder",
    "ImageEmbedder",
    "get_text_embedder",
    "get_image_embedder",
    # Base pipeline
    "ImageDocument",
    "IngestionPipeline",
    # Loaders
    "DocumentLoader",
    "ExtractedImage",
    "LoadedDocument",
    "get_loader_for_source",
    # Validation
    "PipelineValidationError",
    "ValidationResult",
    "validate_chunking_strategy",
    "validate_embedding_dimensions",
    "validate_embedding_model",
    "validate_image_embedding",
    "validate_pipeline_config",
    "validate_and_raise",
]
