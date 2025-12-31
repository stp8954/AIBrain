"""
Ingestion pipeline orchestrator for NyRAG.

This module provides the IngestionPipeline class that orchestrates the complete
ingestion flow: document loading → chunking → embedding → feeding to Vespa.
Includes support for image extraction and embedding for multimodal documents.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyrag.config import Config, PipelineConfig
from nyrag.feed import VespaFeeder
from nyrag.logger import logger
from nyrag.pipeline.chunking import ChunkMetadata, chunk_document, get_chunking_strategy
from nyrag.pipeline.embeddings import get_text_embedder, get_image_embedder
from nyrag.pipeline.loaders import ExtractedImage, LoadedDocument, get_loader_for_source


@dataclass
class ImageDocument:
    """Represents an image document for embedding and indexing.

    This is the output format for processed images that will be fed to Vespa.
    """

    id: str  # Unique identifier for the image document
    source_doc_loc: str  # Location of the source document containing this image
    page: Optional[int] = None  # Page number for PDFs
    alt_text: Optional[str] = None  # Alt text or caption
    embedding: List[float] = field(default_factory=list)  # CLIP embedding
    metadata: Dict = field(default_factory=dict)


class IngestionPipeline:
    """Orchestrates the complete ingestion pipeline.

    This class coordinates:
    1. Document loading (via unified loaders)
    2. Text chunking (via configurable strategies)
    3. Text embedding (via configurable models)
    4. Image embedding (via CLIP models, optional)
    5. Feeding to Vespa (via VespaFeeder)

    Example:
        >>> config = Config.from_yaml("config.yml")
        >>> pipeline = IngestionPipeline(config)
        >>> pipeline.process_document(source="https://example.com", source_type="url")
    """

    def __init__(
        self,
        config: Config,
        redeploy: bool = False,
        vespa_url: str = "http://localhost",
        vespa_port: int = 8080,
    ):
        """Initialize the ingestion pipeline.

        Args:
            config: NyRAG configuration object.
            redeploy: Whether to redeploy Vespa schema before feeding.
            vespa_url: Vespa endpoint URL.
            vespa_port: Vespa endpoint port.
        """
        self.config = config
        self.pipeline_config: PipelineConfig = config.get_pipeline_config()

        # Initialize components
        self.chunking_strategy = get_chunking_strategy(self.pipeline_config)
        self.text_embedder = get_text_embedder(self.pipeline_config)

        # Initialize image embedder if enabled
        self.image_embedder = None
        if self.pipeline_config.image_embedding.enabled:
            self.image_embedder = get_image_embedder(self.pipeline_config)
            logger.info(f"Image embedding enabled with model: {self.image_embedder.model_name}")

        # Initialize Vespa feeder
        self.feeder = VespaFeeder(
            config=config,
            redeploy=redeploy,
            vespa_url=vespa_url,
            vespa_port=vespa_port,
        )

        logger.info(
            f"Initialized IngestionPipeline with {self.chunking_strategy.strategy_name} chunking "
            f"and {self.text_embedder.model_name} embeddings"
        )

    def process_document(
        self,
        source: str,
        source_type: str,
        title: Optional[str] = None,
    ) -> bool:
        """Process a single document through the complete pipeline.

        This method:
        1. Loads the document using the appropriate loader for source_type
        2. Chunks the content using the configured chunking strategy
        3. Embeds the content and chunks using the configured embedding model
        4. Feeds the processed document to Vespa

        Graceful error handling (FR-011): This method catches all exceptions
        and returns False, allowing batch processing to continue with remaining
        documents.

        Args:
            source: Source location (URL, file path, etc.)
            source_type: Type of source ("url", "pdf", "markdown", "text", "note")
            title: Optional override for document title

        Returns:
            True if document was successfully processed and fed, False otherwise.
        """
        logger.info(f"Processing {source_type} document: {source}")

        try:
            # Step 1: Load document
            loaded_doc = self._load_document(source, source_type)

            # Override title if provided
            if title:
                loaded_doc.title = title

            logger.debug(
                f"Loaded document: {loaded_doc.source_loc} "
                f"({len(loaded_doc.content)} chars, title='{loaded_doc.title}')"
            )

            # Step 2: Prepare record for Vespa feeder
            # The feeder will handle chunking and embedding internally
            record = {
                "content": loaded_doc.content,
                "loc": loaded_doc.source_loc,
                "title": loaded_doc.title or "",
                "source_type": loaded_doc.source_type,
            }

            # Step 3: Feed to Vespa (includes chunking and embedding)
            success = self.feeder.feed(record)

            if success:
                logger.success(f"Successfully processed document: {source}")
            else:
                logger.error(f"Failed to feed document to Vespa: {source}")

            return success

        except FileNotFoundError as e:
            logger.error(f"Document not found: {source} - {e}")
            return False
        except ValueError as e:
            logger.error(f"Invalid document content: {source} - {e}")
            return False
        except ConnectionError as e:
            logger.error(f"Network error processing document: {source} - {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error processing document {source}: {type(e).__name__}: {e}")
            return False

    def _load_document(self, source: str, source_type: str) -> LoadedDocument:
        """Load a document using the appropriate loader.

        Args:
            source: Source location (URL, file path, etc.)
            source_type: Type of source ("url", "pdf", "markdown", "text", "note")

        Returns:
            LoadedDocument with content and metadata.

        Raises:
            ValueError: If source_type is not supported.
            FileNotFoundError: If file source does not exist.
        """
        loader = get_loader_for_source(source_type)
        return loader.load(source)

    def process_batch(
        self,
        sources: list[tuple[str, str]],
        title: Optional[str] = None,
    ) -> tuple[int, int]:
        """Process multiple documents through the pipeline.

        Includes progress tracking with periodic status updates.

        Args:
            sources: List of (source, source_type) tuples.
            title: Optional override for document titles.

        Returns:
            Tuple of (successful_count, failed_count).
        """
        total = len(sources)
        success_count = 0
        failed_count = 0

        logger.info(f"Starting batch processing of {total} documents")

        for idx, (source, source_type) in enumerate(sources, 1):
            # Log progress every 10 documents or at the start
            if idx == 1 or idx % 10 == 0 or idx == total:
                progress_pct = (idx / total) * 100
                logger.info(f"Progress: {idx}/{total} ({progress_pct:.1f}%) - Processing: {source}")

            if self.process_document(source, source_type, title):
                success_count += 1
            else:
                failed_count += 1

        # Final summary
        success_pct = (success_count / total * 100) if total > 0 else 0
        logger.info(
            f"Batch processing complete: {success_count}/{total} succeeded ({success_pct:.1f}%), "
            f"{failed_count} failed"
        )

        return success_count, failed_count

    def process_images(
        self,
        images: List[ExtractedImage],
        source_doc_loc: str,
    ) -> List[ImageDocument]:
        """Process extracted images through the image embedding pipeline.

        This method:
        1. Converts ExtractedImage objects to PIL Images
        2. Embeds images using the configured CLIP model
        3. Returns ImageDocument objects ready for Vespa feeding

        Includes progress tracking for image processing.

        Args:
            images: List of ExtractedImage objects from document loading.
            source_doc_loc: Location of the source document.

        Returns:
            List of ImageDocument objects with embeddings.

        Raises:
            RuntimeError: If image embedding is not enabled.
        """
        if not self.image_embedder:
            raise RuntimeError(
                "Image embedding is not enabled. "
                "Set image_embedding.enabled=true in pipeline config."
            )

        if not images:
            return []

        total_images = len(images)
        logger.info(f"Processing {total_images} images from {source_doc_loc}")
        image_docs: List[ImageDocument] = []

        try:
            # Convert to PIL images
            pil_images = []
            valid_indices = []

            for idx, img in enumerate(images):
                try:
                    pil_img = img.to_pil_image()
                    pil_images.append(pil_img)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Could not convert image {idx} to PIL: {e}")

            if not pil_images:
                logger.warning("No valid images to embed")
                return []

            # Batch embed images
            embeddings = self.image_embedder.embed_images(pil_images)

            # Create ImageDocument objects
            for embed_idx, (original_idx, embedding) in enumerate(zip(valid_indices, embeddings)):
                img = images[original_idx]

                # Generate unique ID for the image
                import hashlib

                image_hash = hashlib.md5(img.data).hexdigest()[:12]
                image_id = f"{source_doc_loc}:img:{image_hash}"

                image_doc = ImageDocument(
                    id=image_id,
                    source_doc_loc=source_doc_loc,
                    page=img.page,
                    alt_text=img.alt_text,
                    embedding=embedding,
                    metadata={
                        "format": img.format,
                        "source_metadata": img.metadata,
                    },
                )
                image_docs.append(image_doc)

            logger.success(f"Embedded {len(image_docs)} images from {source_doc_loc}")

        except Exception as e:
            logger.error(f"Failed to process images: {e}")

        return image_docs

    def process_document_with_images(
        self,
        source: str,
        source_type: str,
        title: Optional[str] = None,
    ) -> tuple[bool, List[ImageDocument]]:
        """Process a document and its images through the complete pipeline.

        This method:
        1. Loads the document with image extraction
        2. Processes text content (chunk, embed, feed)
        3. Processes extracted images if image embedding is enabled

        Graceful error handling (FR-011): This method catches all exceptions
        and returns (False, []), allowing batch processing to continue with
        remaining documents.

        Args:
            source: Source location (URL, file path, etc.)
            source_type: Type of source ("url", "pdf", "markdown", "text", "note")
            title: Optional override for document title

        Returns:
            Tuple of (text_success, image_documents).
        """
        logger.info(f"Processing {source_type} document with images: {source}")

        try:
            # Load document with image extraction
            extract_images = self.image_embedder is not None
            loader = get_loader_for_source(source_type, extract_images=extract_images)
            loaded_doc = loader.load(source)

            # Override title if provided
            if title:
                loaded_doc.title = title

            # Process text content
            record = {
                "content": loaded_doc.content,
                "loc": loaded_doc.source_loc,
                "title": loaded_doc.title or "",
                "source_type": loaded_doc.source_type,
            }
            text_success = self.feeder.feed(record)

            # Process images if we have any and image embedding is enabled
            image_docs: List[ImageDocument] = []
            if loaded_doc.images and self.image_embedder:
                image_docs = self.process_images(loaded_doc.images, loaded_doc.source_loc)

            return text_success, image_docs

        except FileNotFoundError as e:
            logger.error(f"Document not found: {source} - {e}")
            return False, []
        except ValueError as e:
            logger.error(f"Invalid document content: {source} - {e}")
            return False, []
        except ConnectionError as e:
            logger.error(f"Network error processing document: {source} - {e}")
            return False, []
        except Exception as e:
            logger.error(f"Unexpected error processing document with images {source}: {type(e).__name__}: {e}")
            return False, []


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    "ImageDocument",
    "IngestionPipeline",
]
