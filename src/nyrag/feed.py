import re
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from vespa.io import VespaResponse


def _sanitize_text(text: str) -> str:
    """Remove illegal characters that Vespa doesn't accept.

    Vespa string fields don't accept:
    - Control characters (0x00-0x1F except tab, newline, carriage return)
    - Unicode noncharacters (U+FDD0-U+FDEF, U+FFFE, U+FFFF, and end-of-plane noncharacters)
    - Surrogate code points (U+D800-U+DFFF)

    Args:
        text: Input text that may contain illegal characters.

    Returns:
        Sanitized text with illegal characters removed.
    """
    # Build a comprehensive pattern for all illegal characters:
    # 1. Control chars 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F (but keep tab, newline, CR)
    # 2. Unicode noncharacters U+FDD0-U+FDEF
    # 3. BMP noncharacters U+FFFE, U+FFFF
    # 4. Surrogate code points U+D800-U+DFFF (shouldn't appear in valid UTF-8 but can in some sources)
    # 5. End-of-plane noncharacters (U+1FFFE, U+1FFFF, U+2FFFE, etc.)

    # Pattern for control characters (excluding tab \x09, LF \x0a, CR \x0d)
    control_chars = r'[\x00-\x08\x0b\x0c\x0e-\x1f]'

    # Pattern for Unicode noncharacters and surrogates
    # \uFDD0-\uFDEF: noncharacter range
    # \uFFFE-\uFFFF: BMP noncharacters
    # \uD800-\uDFFF: surrogate pairs (invalid in isolation)
    unicode_illegal = r'[\uFDD0-\uFDEF\uFFFE\uFFFF\uD800-\uDFFF]'

    # Combined pattern
    pattern = f'{control_chars}|{unicode_illegal}'

    result = re.sub(pattern, '', text)

    # Also handle supplementary plane noncharacters (U+1FFFE, U+1FFFF, U+2FFFE, etc.)
    # These are at the end of each plane from 1-16
    def remove_supplementary_nonchars(s: str) -> str:
        chars = []
        for char in s:
            code = ord(char)
            # Check if it's an end-of-plane noncharacter (xFFFE or xFFFF where x is 1-16)
            if code > 0xFFFF:
                plane_offset = code & 0xFFFF
                if plane_offset in (0xFFFE, 0xFFFF):
                    continue  # Skip this noncharacter
            chars.append(char)
        return ''.join(chars)

    return remove_supplementary_nonchars(result)

from nyrag.config import Config, PipelineConfig
from nyrag.deploy import deploy_app_package
from nyrag.logger import logger
from nyrag.pipeline.chunking import ChunkingStrategy, get_chunking_strategy
from nyrag.pipeline.embeddings import ImageEmbedder, TextEmbedder, get_image_embedder, get_text_embedder
from nyrag.schema import VespaSchema, generate_image_schema
from nyrag.utils import DEFAULT_EMBEDDING_MODEL, get_vespa_tls_config, make_vespa_client

if TYPE_CHECKING:
    from nyrag.pipeline.base import ImageDocument


DEFAULT_HOST = "http://localhost"
DEFAULT_PORT = 8080


class VespaFeeder:
    """Feed processed data (content, chunks, embeddings) into Vespa."""

    def __init__(
        self,
        config: Config,
        redeploy: bool = False,
        vespa_url: str = DEFAULT_HOST,
        vespa_port: int = DEFAULT_PORT,
    ):
        self.config = config
        self.schema_name = config.get_schema_name()
        self.app_package_name = config.get_app_package_name()

        rag_params = config.rag_params or {}
        schema_params = config.get_schema_params()

        # Initialize pipeline configuration
        self.pipeline_config: PipelineConfig = config.get_pipeline_config()

        # Initialize chunking strategy
        self.chunking_strategy: ChunkingStrategy = get_chunking_strategy(self.pipeline_config)

        # Initialize text embedder (uses pipeline config or falls back to legacy config)
        self.text_embedder: TextEmbedder = get_text_embedder(self.pipeline_config)

        # Store model info for Vespa feed fields
        self.embedding_model_name = self.text_embedder.model_name
        self.embedding_dim = self.text_embedder.embedding_dim

        logger.info(
            f"Using embedding model: '{self.embedding_model_name}' "
            f"(dim={self.embedding_dim}, device='{self.text_embedder.device}')"
        )
        logger.info(
            f"Using chunking strategy: {self.chunking_strategy.strategy_name} "
            f"(size={self.chunking_strategy.chunk_size}, overlap={self.chunking_strategy.chunk_overlap})"
        )

        # Update schema_params with auto-detected embedding dimension
        schema_params["embedding_dim"] = self.embedding_dim

        self.app = self._connect_vespa(redeploy, vespa_url, vespa_port, schema_params)

    def feed(self, record: Dict[str, Any]) -> bool:
        """Feed a single record into Vespa.

        Args:
            record: A dictionary with at least 'content' and optional 'loc' fields.

        Returns:
            True if feed succeeded, False otherwise.
        """
        prepared = self._prepare_record(record)
        try:
            response = self.app.feed_data_point(
                schema=self.schema_name,
                data_id=prepared["id"],
                fields=prepared["fields"],
            )
        except Exception as e:
            msg = str(e)
            if "401" in msg and "Unauthorized" in msg:
                logger.error(
                    "Vespa feed returned 401 Unauthorized. "
                    "For Vespa Cloud, set VESPA_CLIENT_CERT and VESPA_CLIENT_KEY (mTLS) before feeding."
                )
            logger.error(f"Feed request failed for id={prepared['id']}: {e}")
            return False

        if self._is_success(response):
            logger.success(f"Fed document id={prepared['id']}")
            return True

        logger.error(f"Feed failed for id={prepared['id']}: {getattr(response, 'json', response)}")
        return False

    def _connect_vespa(
        self,
        redeploy: bool,
        vespa_url: str,
        vespa_port: int,
        schema_params: Dict[str, Any],
    ):
        cert_path, key_path, ca_cert, verify = get_vespa_tls_config()

        if redeploy:
            logger.info("Redeploying Vespa application before feeding")
            vespa_schema = VespaSchema(
                schema_name=self.schema_name,
                app_package_name=self.app_package_name,
                **schema_params,
            )
            app_package = vespa_schema.get_package()
            deploy_app_package(None, app_package=app_package)

        logger.info(f"Connecting to Vespa at {vespa_url}:{vespa_port}")
        return make_vespa_client(vespa_url, vespa_port, cert_path, key_path, ca_cert, verify)

    def _prepare_record(self, record: Dict[str, str]) -> Dict[str, Dict[str, object]]:
        content = record.get("content", "").strip()
        if not content:
            raise ValueError("Record is missing content")

        # Sanitize content to remove illegal control characters (e.g., from PDFs)
        content = _sanitize_text(content)

        loc = record.get("loc", "").strip()
        title = _sanitize_text(record.get("title", "").strip())
        source_type = record.get("source_type", "url").strip()
        data_id = self._make_id(loc)

        # Log document being indexed
        logger.info(f"[INDEX] Processing document for indexing:")
        logger.info(f"  - ID: {data_id}")
        logger.info(f"  - Location: {loc}")
        logger.info(f"  - Title: {title[:80] if title else '(no title)'}")
        logger.info(f"  - Source type: {source_type}")
        logger.info(f"  - Content length: {len(content)} chars")

        # Use the new LangChain-based chunking strategy
        chunk_texts = self.chunking_strategy.split(content)
        # Sanitize each chunk as well
        chunk_texts = [_sanitize_text(chunk) for chunk in chunk_texts]

        logger.info(f"[INDEX] Chunking complete:")
        logger.info(f"  - Strategy: {self.chunking_strategy.strategy_name}")
        logger.info(f"  - Total chunks: {len(chunk_texts)}")

        # Log individual chunk details (first 5 and last 1)
        for i, chunk in enumerate(chunk_texts[:5]):
            preview = chunk[:100].replace('\n', ' ')
            logger.info(f"  - Chunk {i+1}: {len(chunk)} chars - '{preview}...'")
        if len(chunk_texts) > 6:
            logger.info(f"  - ... ({len(chunk_texts) - 6} more chunks) ...")
            last_chunk = chunk_texts[-1]
            preview = last_chunk[:100].replace('\n', ' ')
            logger.info(f"  - Chunk {len(chunk_texts)}: {len(last_chunk)} chars - '{preview}...'")

        # Batch encode content + chunks to minimize model calls
        logger.info(f"[INDEX] Generating embeddings:")
        logger.info(f"  - Model: {self.embedding_model_name}")
        logger.info(f"  - Dimensions: {self.embedding_dim}")
        logger.info(f"  - Texts to embed: {1 + len(chunk_texts)} (1 content + {len(chunk_texts)} chunks)")

        embeddings = self._encode_texts([content, *chunk_texts])
        content_embedding, chunk_embeddings = embeddings[0], embeddings[1:]
        if len(chunk_embeddings) != len(chunk_texts):
            raise ValueError(
                "Chunk embeddings count mismatch. " f"Expected {len(chunk_texts)}, got {len(chunk_embeddings)}"
            )
        logger.info(f"[INDEX] Embeddings generated successfully")

        fields = {
            "id": data_id,
            "loc": loc,
            "title": title,
            "content": content,
            "chunks": chunk_texts,
            "chunk_count": len(chunk_texts),
            "content_embedding": self._dense_tensor(content_embedding),
            "chunk_embeddings": self._chunk_tensor(chunk_embeddings),
            # Pipeline tracking fields
            "source_type": source_type,
            "chunking_strategy": self.chunking_strategy.strategy_name,
            "embedding_model": self.embedding_model_name,
            "indexed_at": int(time.time() * 1000),  # milliseconds since epoch
        }

        return {"id": data_id, "fields": fields}

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using the TextEmbedder.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self.text_embedder.embed_texts(texts)

    def _dense_tensor(self, values: List[float]) -> Dict[str, List[float]]:
        target_dim = self.embedding_dim
        if len(values) != target_dim:
            raise ValueError(f"Tensor length mismatch. Expected {target_dim}, got {len(values)}")
        return {"values": values}

    def _chunk_tensor(self, chunk_vectors: List[List[float]]) -> Dict[str, List[Dict[str, object]]]:
        target_dim = self.embedding_dim
        tensor: Dict[str, List[float]] = {}
        for chunk_idx, vector in enumerate(chunk_vectors):
            if len(vector) != target_dim:
                raise ValueError(
                    f"Chunk {chunk_idx} embedding dim mismatch. " f"Expected {target_dim}, got {len(vector)}"
                )
            tensor[str(chunk_idx)] = vector
        return tensor

    def _make_id(self, loc: str) -> str:
        base = loc if loc else f"nyrag-{uuid.uuid4()}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

    def _is_success(self, response: VespaResponse) -> bool:
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            return False
        if status_code >= 400:
            return False
        return True

    # =========================================================================
    # Image Document Feeding (P3 - Future multimodal search)
    # =========================================================================

    def feed_image(self, image_doc: "ImageDocument") -> bool:
        """Feed a single image document into Vespa.

        This method feeds an image document (with CLIP embedding) to Vespa
        for future multimodal search capabilities.

        Args:
            image_doc: ImageDocument with embedding.

        Returns:
            True if feed succeeded, False otherwise.
        """
        # Check if image embedding is enabled
        if not self.pipeline_config.image_embedding.enabled:
            logger.warning("[IMAGE] Image embedding is disabled. Skipping image feed.")
            return False

        logger.info(f"[IMAGE] Indexing image document:")
        logger.info(f"  - Image ID: {image_doc.id}")
        logger.info(f"  - Source document: {image_doc.source_doc_loc}")
        logger.info(f"  - Page: {image_doc.page or 'N/A'}")
        logger.info(f"  - Alt text: {image_doc.alt_text[:50] if image_doc.alt_text else '(none)'}")
        logger.info(f"  - Embedding dimensions: {len(image_doc.embedding)}")

        prepared = self._prepare_image_record(image_doc)
        image_schema_name = self._get_image_schema_name()

        try:
            response = self.app.feed_data_point(
                schema=image_schema_name,
                data_id=prepared["id"],
                fields=prepared["fields"],
            )
        except Exception as e:
            logger.error(f"[IMAGE] Image feed request failed for id={prepared['id']}: {e}")
            return False

        if self._is_success(response):
            logger.success(f"[IMAGE] Successfully indexed image id={prepared['id']}")
            return True

        logger.error(f"[IMAGE] Image feed failed for id={prepared['id']}: {getattr(response, 'json', response)}")
        return False

    def feed_images(self, image_docs: List["ImageDocument"]) -> tuple:
        """Feed multiple image documents into Vespa.

        Args:
            image_docs: List of ImageDocument objects.

        Returns:
            Tuple of (success_count, failed_count).
        """
        success_count = 0
        failed_count = 0

        if not image_docs:
            logger.info("[IMAGE] No images to index")
            return 0, 0

        logger.info(f"[IMAGE] Starting batch indexing of {len(image_docs)} images")

        for idx, image_doc in enumerate(image_docs, 1):
            logger.info(f"[IMAGE] Processing image {idx}/{len(image_docs)}")
            if self.feed_image(image_doc):
                success_count += 1
            else:
                failed_count += 1

        logger.info(f"[IMAGE] Batch indexing complete: {success_count} succeeded, {failed_count} failed")
        return success_count, failed_count

    def _prepare_image_record(self, image_doc: "ImageDocument") -> Dict[str, Any]:
        """Prepare an image document for Vespa feeding.

        Args:
            image_doc: ImageDocument with embedding.

        Returns:
            Dictionary with id and fields for Vespa.
        """
        # Get image embedder for model name
        image_embedder = get_image_embedder(self.pipeline_config)

        data_id = str(uuid.uuid5(uuid.NAMESPACE_URL, image_doc.id))

        fields = {
            "id": data_id,
            "parent_document_id": image_doc.source_doc_loc,
            "position_in_parent": image_doc.page or 0,
            "alt_text": image_doc.alt_text or "",
            "image_embedding": {"values": image_doc.embedding},
            "embedding_model": image_embedder.model_name,
            "indexed_at": int(time.time() * 1000),
        }

        return {"id": data_id, "fields": fields}

    def _get_image_schema_name(self) -> str:
        """Get the image schema name based on config.

        Returns:
            Vespa schema name for images.
        """
        # Clean the project name for valid Vespa naming
        clean_name = self.config.name.replace("-", "").replace("_", "").lower()
        return f"nyrag{clean_name}images"


def feed_from_config(
    config_path: str,
    record: Dict[str, Any],
    redeploy: bool = False,
) -> bool:
    """Convenience helper to feed a single record using a YAML config path."""
    config = Config.from_yaml(config_path)
    feeder = VespaFeeder(config=config, redeploy=redeploy)
    return feeder.feed(record)
