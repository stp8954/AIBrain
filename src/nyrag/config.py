from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import os

import yaml
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Pipeline Configuration Models (LangChain-based ingestion pipeline)
# =============================================================================


class TextEmbeddingConfig(BaseModel):
    """Configuration for text embedding model."""

    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name or local path",
    )
    batch_size: int = Field(default=32, ge=1, le=512)
    device: Literal["cpu", "cuda", "mps"] = "cpu"


class ImageEmbeddingConfig(BaseModel):
    """Configuration for image embedding model."""

    enabled: bool = True
    model: str = Field(
        default="sentence-transformers/clip-ViT-B-32",
        description="Vision embedding model",
    )
    batch_size: int = Field(default=16, ge=1, le=128)
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    max_dimension: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Resize images to max dimension",
    )


class BackgroundProcessingConfig(BaseModel):
    """Configuration for background job processing."""

    max_concurrent_jobs: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum number of concurrent ingestion jobs",
    )
    enable_image_indexing: bool = Field(
        default=True,
        description="Enable image extraction and CLIP embedding during ingestion",
    )
    max_images_per_doc: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of images to extract per document",
    )


class DoclingConfig(BaseModel):
    """Configuration for Docling document parser.

    Docling provides superior PDF/DOCX parsing with layout understanding,
    table structure recognition, and intelligent chunking.
    """

    ocr_enabled: bool = Field(
        default=False,
        description="Enable OCR for scanned documents (adds latency)",
    )
    table_structure: bool = Field(
        default=True,
        description="Use TableFormer model for table structure recognition",
    )
    export_tables_as: Literal["markdown", "html", "json"] = Field(
        default="markdown",
        description="Format for table export in chunks",
    )
    image_extraction: bool = Field(
        default=True,
        description="Extract images from documents",
    )


class PipelineConfig(BaseModel):
    """Configuration for the LangChain-based ingestion pipeline."""

    # Document parser selection: "docling" (recommended) or "legacy" (PyPDF)
    document_parser: Literal["docling", "legacy"] = Field(
        default="docling",
        description="Document parser: 'docling' for superior parsing, 'legacy' for PyPDF",
    )

    # Chunking strategy: "docling" uses native HybridChunker
    chunking_strategy: Literal["fixed", "recursive", "semantic", "docling"] = Field(
        default="docling",
        description="Chunking strategy: 'docling' for context-aware chunks",
    )
    chunk_size: int = Field(default=512, ge=1, le=8192)
    chunk_overlap: int = Field(default=50, ge=0)

    # Docling-specific configuration
    docling: DoclingConfig = Field(default_factory=DoclingConfig)

    text_embedding: TextEmbeddingConfig = Field(default_factory=TextEmbeddingConfig)
    image_embedding: ImageEmbeddingConfig = Field(default_factory=ImageEmbeddingConfig)

    # Background processing configuration
    background: BackgroundProcessingConfig = Field(default_factory=BackgroundProcessingConfig)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure chunk_overlap is less than chunk_size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    def uses_docling(self) -> bool:
        """Check if Docling parser is enabled."""
        return self.document_parser == "docling"

    def uses_docling_chunking(self) -> bool:
        """Check if Docling native chunking is enabled."""
        return self.chunking_strategy == "docling"


# =============================================================================
# Existing Configuration Models
# =============================================================================


class CrawlParams(BaseModel):
    """Parameters specific to web crawling."""

    respect_robots_txt: bool = True
    aggressive_crawl: bool = False
    follow_subdomains: bool = True
    strict_mode: bool = False
    user_agent_type: Literal["chrome", "firefox", "safari", "mobile", "bot"] = "chrome"
    custom_user_agent: Optional[str] = None
    allowed_domains: Optional[List[str]] = None


class DocParams(BaseModel):
    """Parameters specific to document processing."""

    recursive: bool = True
    include_hidden: bool = False
    follow_symlinks: bool = False
    max_file_size_mb: Optional[float] = None
    file_extensions: Optional[List[str]] = None


class NotesParams(BaseModel):
    """Parameters for notes capture and storage."""

    max_image_size_mb: float = 5.0
    storage_path: str = "notes"


class BlogParams(BaseModel):
    """Parameters for blog generation."""

    templates_path: str = "blog_templates"
    output_path: str = "blogs"
    timeout_seconds: int = 120


class Config(BaseModel):
    """Configuration model for nyrag."""

    name: str
    mode: Literal["web", "docs"]
    start_loc: str
    exclude: Optional[List[str]] = None
    rag_params: Optional[Dict[str, Any]] = None
    crawl_params: Optional[CrawlParams] = None
    doc_params: Optional[DocParams] = None
    notes_params: Optional[NotesParams] = None
    blog_params: Optional[BlogParams] = None

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate and normalize mode."""
        if v.lower() in ["web", "docs", "doc"]:
            return "docs" if v.lower() in ["docs", "doc"] else "web"
        raise ValueError("mode must be 'web' or 'docs'")

    def model_post_init(self, __context):
        """Initialize params with defaults if None."""
        if self.crawl_params is None:
            self.crawl_params = CrawlParams()
        if self.doc_params is None:
            self.doc_params = DocParams()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_output_path(self) -> Path:
        """Get the output directory path."""
        # Use NYRAG_DATA_DIR if set (for Docker), otherwise use relative 'output'
        data_dir = os.getenv("NYRAG_DATA_DIR", "")
        base_path = Path(data_dir) / "output" if data_dir else Path("output")
        # Use schema name format for consistency (lowercase alphanumeric only)
        schema_name = self.get_schema_name()
        return base_path / schema_name

    def get_app_path(self) -> Path:
        """Get the app directory path for Vespa schema."""
        return self.get_output_path() / "app"

    def get_schema_name(self) -> str:
        """Get the schema name in format nyragPROJECTNAME (lowercase alphanumeric only)."""
        # Remove hyphens, underscores, and convert to lowercase for valid Vespa schema name
        clean_name = self.name.replace("-", "").replace("_", "").lower()
        return f"nyrag{clean_name}"

    def get_app_package_name(self) -> str:
        """Get a valid application package name (lowercase, no hyphens, max 20 chars)."""
        # Remove hyphens and convert to lowercase
        clean_name = self.name.replace("-", "").replace("_", "").lower()
        # Prefix with nyrag and limit to 20 characters
        app_name = f"nyrag{clean_name}"[:20]
        return app_name

    def get_schema_params(self) -> Dict[str, Any]:
        """Get schema parameters from rag_params."""
        if self.rag_params is None:
            return {}
        return {
            "embedding_dim": self.rag_params.get("embedding_dim", 384),
            "chunk_size": self.rag_params.get("chunk_size", 1024),
            "distance_metric": self.rag_params.get("distance_metric", "angular"),
        }

    def get_pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration with backward compatibility.

        If rag_params.pipeline is defined, parse it into a PipelineConfig.
        Otherwise, return defaults that match legacy behavior.

        Returns:
            PipelineConfig instance with either explicit or default settings.
        """
        if self.rag_params is None:
            return PipelineConfig()

        pipeline_data = self.rag_params.get("pipeline")
        if pipeline_data is None:
            # Backward compatibility: use legacy chunk_size/chunk_overlap if present
            legacy_chunk_size = self.rag_params.get("chunk_size", 512)
            legacy_chunk_overlap = self.rag_params.get("chunk_overlap", 50)
            return PipelineConfig(
                chunk_size=legacy_chunk_size,
                chunk_overlap=legacy_chunk_overlap,
            )

        # Parse nested pipeline config
        return PipelineConfig(**pipeline_data)

    def is_web_mode(self) -> bool:
        """Check if config is for web crawling."""
        return self.mode == "web"

    def is_docs_mode(self) -> bool:
        """Check if config is for document processing."""
        return self.mode == "docs"
