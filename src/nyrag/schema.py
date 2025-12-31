import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel
from pydantic import Field as PydanticField
from vespa.package import (
    ApplicationPackage,
    Document,
    DocumentSummary,
    Field,
    FieldSet,
    Function,
    RankProfile,
    Schema,
    SecondPhaseRanking,
    Summary,
)


@dataclass
class VespaSchema:
    schema_name: str
    app_package_name: str
    embedding_dim: int = 384
    chunk_size: int = 1024
    distance_metric: str = "angular"

    def create_schema_fields(self) -> Schema:
        """Create a Vespa schema with predefined fields and rank profile."""
        # Create document with basic fields
        document = Document(
            fields=[
                Field(
                    name="id",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="content",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="loc",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="title",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="chunks",
                    type="array<string>",
                    indexing=["summary", "index"],
                    index="enable-bm25",
                ),
                Field(
                    name="chunk_count",
                    type="int",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="content_embedding",
                    type=f"tensor<float>(x[{self.embedding_dim}])",
                    indexing=["attribute", "index"],
                    attribute=[f"distance-metric: {self.distance_metric}"],
                ),
                Field(
                    name="chunk_embeddings",
                    type=f"tensor<float>(chunk{{}}, x[{self.embedding_dim}])",
                    indexing=["attribute", "index"],
                    attribute=[f"distance-metric: {self.distance_metric}"],
                ),
                # Pipeline tracking fields (added for LangChain ingestion pipeline)
                Field(
                    name="source_type",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="chunking_strategy",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="embedding_model",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="indexed_at",
                    type="long",
                    indexing=["attribute", "summary"],
                ),
            ]
        )

        # Create schema with document
        schema = Schema(name=self.schema_name, document=document)

        # Define field set
        schema.add_field_set(FieldSet(name="default", fields=["content", "chunks"]))

        # Add document summaries
        self._add_document_summaries(schema)

        return schema

    def _add_document_summaries(self, schema: Schema) -> None:
        """Add document summaries to the schema."""
        # Document summary: no-chunks (basic fields without chunks content)
        schema.add_document_summary(
            DocumentSummary(
                name="no-chunks",
                summary_fields=[
                    Summary(name="id"),
                    Summary(name="content"),
                    Summary(name="loc"),
                    Summary(name="title"),
                ],
            )
        )

        # Document summary: top_n_chunks (select top N chunks based on similarity)
        schema.add_document_summary(
            DocumentSummary(
                name="top_k_chunks",
                from_disk=True,
                summary_fields=[
                    Summary(name="id"),
                    Summary(name="loc"),
                    Summary(name="title"),
                    Summary(
                        name="chunks_topk",
                        fields=[
                            ("source", "chunks"),
                            (
                                "select-elements-by",
                                "top_k_chunk_combined_scores",
                            ),
                        ],
                    ),
                ],
            )
        )

    def add_rank_profile(self, schema: Schema) -> None:
        # Common inputs
        common_inputs = [
            ("query(embedding)", f"tensor<float>(x[{self.embedding_dim}])"),
            ("query(k)", "double"),
        ]

        # Functions for layered ranking: coarse content relevance, then best chunks
        chunk_functions = [
            Function(
                name="chunk_text_scores",
                expression=("map(bm25(chunks), f(x)((2 / 3.141592653589793) * atan(x)))"),
            ),
            Function(
                name="chunk_emb_vecs",
                expression="attribute(chunk_embeddings)",
            ),
            Function(name="content_vec", expression="attribute(content_embedding)"),
            Function(
                name="vector_norms",
                expression="sqrt(sum(pow(t, 2), x))",
                args=["t"],
            ),
            Function(
                name="chunk_sim_scores",
                expression=(
                    "reduce(query(embedding) * chunk_emb_vecs(), sum, x) / "
                    "(vector_norms(chunk_emb_vecs()) * vector_norms(query(embedding)))"
                ),
            ),
            Function(
                name="content_sim_score",
                expression=(
                    "reduce(query(embedding) * content_vec(), sum, x) / "
                    "(vector_norms(content_vec()) * vector_norms(query(embedding)))"
                ),
            ),
            Function(
                name="content_text_score",
                expression="(2 / 3.141592653589793) * atan(bm25(content))",
            ),
            Function(
                name="chunk_combined_scores",
                expression=("0.5 * chunk_sim_scores() + " "0.3 * chunk_text_scores() + " "0.2 * content_sim_score()"),
            ),
            Function(
                name="chunk_count",
                expression="if(query(k) > 0, query(k), 3)",
            ),
            Function(
                name="top_k_chunk_combined_scores",
                expression="top(chunk_count(), chunk_combined_scores())",
            ),
            Function(
                name="best_chunk_score",
                expression="reduce(top_k_chunk_combined_scores(), max, chunk)",
            ),
        ]

        summary_features = [
            "top_k_chunk_combined_scores",
            "best_chunk_score",
            "content_sim_score",
            "content_text_score",
        ]

        schema.add_rank_profile(
            RankProfile(
                name="default",
                inputs=common_inputs,
                functions=chunk_functions,
                summary_features=summary_features,
                rank_properties=[("query(chunks).elementGap", "1")],
                first_phase="sum(chunk_combined_scores())",
                second_phase=SecondPhaseRanking(expression="best_chunk_score()", rerank_count=100),
            )
        )

    def get_package(self) -> ApplicationPackage:
        """Get the Vespa application package with schema and rank profile."""
        schema = self.create_schema_fields()
        self.add_rank_profile(schema)
        app_package = ApplicationPackage(name=self.app_package_name, schema=[schema])
        return app_package

    def save_package(self, output_dir: str) -> None:
        """Save the Vespa application package to the specified directory."""
        app_package = self.get_package()
        app_package.to_files(output_dir)


@dataclass
class NotesVespaSchema:
    """Vespa schema specifically for notes storage and retrieval.

    This schema is simpler than the main VespaSchema as notes don't need
    chunk-level retrieval - each note is treated as a single document.
    """

    schema_name: str
    app_package_name: str
    embedding_dim: int = 384
    distance_metric: str = "angular"

    def create_schema_fields(self) -> Schema:
        """Create a Vespa schema for notes with basic fields and embedding."""
        document = Document(
            fields=[
                Field(
                    name="id",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="title",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="content",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="tags",
                    type="array<string>",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="created_at",
                    type="long",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="updated_at",
                    type="long",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="content_embedding",
                    type=f"tensor<float>(x[{self.embedding_dim}])",
                    indexing=["attribute", "index"],
                    attribute=[f"distance-metric: {self.distance_metric}"],
                ),
            ]
        )

        schema = Schema(name=self.schema_name, document=document)
        schema.add_field_set(FieldSet(name="default", fields=["title", "content", "tags"]))

        # Add document summary
        schema.add_document_summary(
            DocumentSummary(
                name="default",
                summary_fields=[
                    Summary(name="id"),
                    Summary(name="title"),
                    Summary(name="content"),
                    Summary(name="tags"),
                    Summary(name="created_at"),
                    Summary(name="updated_at"),
                ],
            )
        )

        return schema

    def add_rank_profile(self, schema: Schema) -> None:
        """Add a simple hybrid ranking profile for notes."""
        common_inputs = [
            ("query(embedding)", f"tensor<float>(x[{self.embedding_dim}])"),
        ]

        functions = [
            Function(
                name="content_vec",
                expression="attribute(content_embedding)",
            ),
            Function(
                name="vector_norms",
                expression="sqrt(sum(pow(t, 2), x))",
                args=["t"],
            ),
            Function(
                name="sim_score",
                expression=(
                    "reduce(query(embedding) * content_vec(), sum, x) / "
                    "(vector_norms(content_vec()) * vector_norms(query(embedding)))"
                ),
            ),
            Function(
                name="text_score",
                expression="(2 / 3.141592653589793) * atan(bm25(content) + bm25(title))",
            ),
            Function(
                name="combined_score",
                expression="0.6 * sim_score() + 0.4 * text_score()",
            ),
        ]

        schema.add_rank_profile(
            RankProfile(
                name="default",
                inputs=common_inputs,
                functions=functions,
                summary_features=["sim_score", "text_score", "combined_score"],
                first_phase="combined_score()",
            )
        )

    def get_package(self) -> ApplicationPackage:
        """Get the Vespa application package with notes schema."""
        schema = self.create_schema_fields()
        self.add_rank_profile(schema)
        app_package = ApplicationPackage(name=self.app_package_name, schema=[schema])
        return app_package


def generate_notes_schema(project_name: str, embedding_dim: int = 384) -> NotesVespaSchema:
    """Generate a Vespa schema for notes storage.

    Args:
        project_name: The project name (used for schema naming).
        embedding_dim: Dimension of the embedding vectors.

    Returns:
        A NotesVespaSchema instance configured for the project.
    """
    # Clean the project name for valid Vespa naming
    clean_name = project_name.replace("-", "").replace("_", "").lower()
    schema_name = f"nyrag{clean_name}notes"
    app_package_name = f"nyrag{clean_name}"[:20]

    return NotesVespaSchema(
        schema_name=schema_name,
        app_package_name=app_package_name,
        embedding_dim=embedding_dim,
    )


# =============================================================================
# UI Document Manager Models
# =============================================================================

# Type literals for status enums
DataSourceType = Literal["url", "pdf", "markdown", "txt"]
DataSourceStatus = Literal["pending", "processing", "indexed", "failed"]
JobType = Literal["crawl", "process_file", "sync"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
MessageRole = Literal["user", "assistant"]


def generate_id() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


# -----------------------------------------------------------------------------
# Data Source Models
# -----------------------------------------------------------------------------


class DataSourceCreate(BaseModel):
    """Request model for creating a data source."""

    source_type: DataSourceType
    name: str = PydanticField(..., min_length=1, max_length=255)
    url: Optional[str] = PydanticField(None, max_length=2048)
    file_path: Optional[str] = PydanticField(None, max_length=1024)


class DataSourceUpdate(BaseModel):
    """Request model for updating a data source."""

    name: Optional[str] = None
    status: Optional[DataSourceStatus] = PydanticField(None)
    error_message: Optional[str] = None
    document_count: Optional[int] = None


class DataSourceResponse(BaseModel):
    """Response model for a data source."""

    id: str
    source_type: DataSourceType
    name: str
    status: DataSourceStatus
    url: Optional[str] = None
    file_path: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = PydanticField(None)
    document_count: int = 0
    error_message: Optional[str] = None


class DataSourceListResponse(BaseModel):
    """Response model for listing data sources."""

    sources: List[DataSourceResponse]
    total: int


# -----------------------------------------------------------------------------
# Job Models
# -----------------------------------------------------------------------------


class JobCreate(BaseModel):
    """Request model for creating a job."""

    job_type: JobType
    data_source_id: str = PydanticField(..., min_length=1)


class JobUpdate(BaseModel):
    """Request model for updating a job."""

    status: Optional[JobStatus] = None
    progress: Optional[int] = PydanticField(None, ge=0, le=100)
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


class JobResponse(BaseModel):
    """Response model for a job."""

    id: str
    job_type: JobType
    data_source_id: str
    status: JobStatus
    progress: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobListResponse(BaseModel):
    """Response model for listing jobs."""

    jobs: List[JobResponse]
    total: int


# -----------------------------------------------------------------------------
# Conversation Models
# -----------------------------------------------------------------------------


class ConversationCreate(BaseModel):
    """Request model for creating a conversation."""

    title: Optional[str] = PydanticField(None, max_length=255)


class ConversationResponse(BaseModel):
    """Response model for a conversation."""

    id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Response model for listing conversations."""

    conversations: List[ConversationResponse]
    total: int


# -----------------------------------------------------------------------------
# Message Models
# -----------------------------------------------------------------------------


class MessageCreate(BaseModel):
    """Request model for creating a message."""

    role: MessageRole
    content: str = PydanticField(..., min_length=1)


class MessageResponse(BaseModel):
    """Response model for a message."""

    id: str
    conversation_id: str
    role: MessageRole
    content: str
    created_at: datetime


class MessageListResponse(BaseModel):
    """Response model for listing messages."""

    messages: List[MessageResponse]
    total: int


# -----------------------------------------------------------------------------
# System Status Models
# -----------------------------------------------------------------------------


class SystemStatusResponse(BaseModel):
    """Response model for system status."""

    vespa_available: bool = PydanticField(..., description="Whether Vespa is reachable")
    database_available: bool = PydanticField(..., description="Whether SQLite database is accessible")
    running_jobs: int = PydanticField(..., description="Number of currently running jobs")
    queued_jobs: int = PydanticField(..., description="Number of queued jobs")
    total_data_sources: int = PydanticField(..., description="Total number of data sources")
    total_conversations: int = PydanticField(..., description="Total number of conversations")


# =============================================================================
# Image Document Schema (for future multimodal search - P3)
# =============================================================================


@dataclass
class ImageVespaSchema:
    """Vespa schema for image documents with embeddings.

    This schema stores image embeddings separately from parent documents,
    enabling future multimodal search capabilities.
    """

    schema_name: str
    app_package_name: str
    embedding_dim: int = 512  # CLIP default dimension
    distance_metric: str = "angular"

    def create_schema_fields(self) -> Schema:
        """Create a Vespa schema for image documents."""
        document = Document(
            fields=[
                Field(
                    name="id",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="parent_document_id",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="image_hash",
                    type="string",
                    indexing=["attribute"],
                ),
                Field(
                    name="position_in_parent",
                    type="int",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="alt_text",
                    type="string",
                    indexing=["index", "summary"],
                    index="enable-bm25",
                ),
                Field(
                    name="width",
                    type="int",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="height",
                    type="int",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="thumbnail_path",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="image_embedding",
                    type=f"tensor<float>(x[{self.embedding_dim}])",
                    indexing=["attribute", "index"],
                    attribute=[f"distance-metric: {self.distance_metric}"],
                ),
                Field(
                    name="embedding_model",
                    type="string",
                    indexing=["attribute", "summary"],
                ),
                Field(
                    name="indexed_at",
                    type="long",
                    indexing=["attribute", "summary"],
                ),
            ]
        )

        schema = Schema(name=self.schema_name, document=document)
        schema.add_field_set(FieldSet(name="default", fields=["alt_text"]))

        # Add document summary
        schema.add_document_summary(
            DocumentSummary(
                name="default",
                summary_fields=[
                    Summary(name="id"),
                    Summary(name="parent_document_id"),
                    Summary(name="position_in_parent"),
                    Summary(name="alt_text"),
                    Summary(name="width"),
                    Summary(name="height"),
                    Summary(name="thumbnail_path"),
                ],
            )
        )

        return schema

    def add_rank_profile(self, schema: Schema) -> None:
        """Add a simple vector similarity ranking for image search."""
        common_inputs = [
            ("query(image_embedding)", f"tensor<float>(x[{self.embedding_dim}])"),
        ]

        functions = [
            Function(
                name="image_vec",
                expression="attribute(image_embedding)",
            ),
            Function(
                name="vector_norms",
                expression="sqrt(sum(pow(t, 2), x))",
                args=["t"],
            ),
            Function(
                name="sim_score",
                expression=(
                    "reduce(query(image_embedding) * image_vec(), sum, x) / "
                    "(vector_norms(image_vec()) * vector_norms(query(image_embedding)))"
                ),
            ),
            Function(
                name="text_score",
                expression="(2 / 3.141592653589793) * atan(bm25(alt_text))",
            ),
            Function(
                name="combined_score",
                expression="0.7 * sim_score() + 0.3 * text_score()",
            ),
        ]

        schema.add_rank_profile(
            RankProfile(
                name="default",
                inputs=common_inputs,
                functions=functions,
                summary_features=["sim_score", "text_score", "combined_score"],
                first_phase="combined_score()",
            )
        )

    def get_package(self) -> ApplicationPackage:
        """Get the Vespa application package with image schema."""
        schema = self.create_schema_fields()
        self.add_rank_profile(schema)
        app_package = ApplicationPackage(name=self.app_package_name, schema=[schema])
        return app_package


def generate_image_schema(
    project_name: str, embedding_dim: int = 512, distance_metric: str = "angular"
) -> ImageVespaSchema:
    """Generate a Vespa schema for image document storage.

    Args:
        project_name: The project name (used for schema naming).
        embedding_dim: Dimension of the image embedding vectors (CLIP default: 512).
        distance_metric: Distance metric for vector similarity.

    Returns:
        An ImageVespaSchema instance configured for the project.
    """
    clean_name = project_name.replace("-", "").replace("_", "").lower()
    schema_name = f"nyrag{clean_name}images"
    app_package_name = f"nyrag{clean_name}"[:20]

    return ImageVespaSchema(
        schema_name=schema_name,
        app_package_name=app_package_name,
        embedding_dim=embedding_dim,
        distance_metric=distance_metric,
    )
