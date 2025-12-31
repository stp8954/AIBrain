"""
Text chunking strategies for the NyRAG ingestion pipeline.

This module provides pluggable chunking strategies using LangChain text splitters.
Strategies can be configured via YAML configuration without code changes.

Supported strategies:
    - fixed: Character-based fixed-size splitting (CharacterTextSplitter)
    - recursive: Smart splitting by separators (RecursiveCharacterTextSplitter)
    - semantic: Sentence-based splitting using NLP (SpacyTextSplitter)
    - docling: Context-aware chunking using Docling's HybridChunker [default]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from nyrag.config import PipelineConfig
from nyrag.logger import logger

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DocumentChunk:
    """Represents a chunk of content after splitting.

    Attributes:
        id: Unique chunk identifier (generated).
        content: The chunk text content.
        source_id: Parent document identifier.
        source_type: Source type (url, pdf, markdown, text, note).
        source_loc: Original location (URL or file path).
        parent_title: Title of the parent document.
        chunk_index: Position in parent document (0-based).
        chunking_strategy: Strategy used to create this chunk.
        metadata: Additional metadata from the loader.
    """

    id: str
    content: str
    source_id: str
    source_type: Literal["url", "pdf", "markdown", "text", "note"]
    source_loc: str
    chunk_index: int
    chunking_strategy: str
    parent_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkMetadata:
    """Metadata for chunking a document.

    Attributes:
        source_id: Parent document identifier.
        source_type: Source type (url, pdf, markdown, text, note).
        source_loc: Original location (URL or file path).
        parent_title: Title of the parent document.
        extra: Additional metadata from the loader.
    """

    source_id: str
    source_type: Literal["url", "pdf", "markdown", "text", "note"]
    source_loc: str
    parent_title: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Chunking Strategy Base Class
# =============================================================================


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies."""

    strategy_name: str = "base"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize the chunking strategy.

        Args:
            chunk_size: Target chunk size (interpretation depends on strategy).
            chunk_overlap: Overlap between chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: The input text to split.

        Returns:
            List of text chunks.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"


# =============================================================================
# Concrete Strategy Implementations
# =============================================================================


class FixedChunkingStrategy(ChunkingStrategy):
    """Fixed-size character-based splitting.

    Uses LangChain's CharacterTextSplitter for simple fixed-size chunks.
    Best for: Code, structured data, when exact chunk sizes are needed.
    """

    strategy_name: str = "fixed"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self._splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> List[str]:
        """Split text into fixed-size character chunks."""
        if not text or not text.strip():
            return []
        chunks = self._splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


class RecursiveChunkingStrategy(ChunkingStrategy):
    """Smart recursive splitting by separators.

    Uses LangChain's RecursiveCharacterTextSplitter which tries to split
    by paragraphs first, then sentences, then words, then characters.
    Best for: General content, prose, documentation.
    """

    strategy_name: str = "recursive"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self._splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> List[str]:
        """Split text recursively by natural boundaries."""
        if not text or not text.strip():
            return []
        chunks = self._splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


class SemanticChunkingStrategy(ChunkingStrategy):
    """Sentence-based semantic splitting using NLP.

    Uses LangChain's SpacyTextSplitter for sentence-aware chunking.
    Requires spacy and a language model (en_core_web_sm).
    Best for: Articles, documentation, when sentence boundaries matter.

    Note: Falls back to RecursiveChunkingStrategy if spacy is unavailable.
    """

    strategy_name: str = "semantic"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self._splitter = None
        self._fallback = None
        self._init_splitter()

    def _init_splitter(self) -> None:
        """Initialize the spacy-based splitter with fallback."""
        try:
            from langchain_text_splitters import SpacyTextSplitter

            self._splitter = SpacyTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                pipeline="en_core_web_sm",
            )
            logger.debug("Initialized SpacyTextSplitter for semantic chunking")
        except ImportError:
            logger.warning(
                "spacy not available for semantic chunking. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
            self._fallback = RecursiveChunkingStrategy(self.chunk_size, self.chunk_overlap)
        except OSError as e:
            # Spacy model not downloaded
            if "en_core_web_sm" in str(e):
                logger.warning(
                    "Spacy model 'en_core_web_sm' not found. "
                    "Download with: python -m spacy download en_core_web_sm"
                )
            else:
                logger.warning(f"Spacy initialization failed: {e}")
            self._fallback = RecursiveChunkingStrategy(self.chunk_size, self.chunk_overlap)

    def split(self, text: str) -> List[str]:
        """Split text by sentence boundaries."""
        if not text or not text.strip():
            return []

        if self._fallback is not None:
            return self._fallback.split(text)

        chunks = self._splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


class DoclingChunkingStrategy(ChunkingStrategy):
    """Context-aware chunking using Docling's HybridChunker.

    Uses Docling's native HybridChunker which understands document structure
    and provides context-enriched chunks with headings and metadata.

    Best for: PDFs, academic papers, technical documents with complex structure.

    Features:
    - Respects document structure (headings, sections)
    - Preserves table context
    - Adds heading context to chunks for better retrieval
    - Uses tokenizer-aware chunk sizing

    Note: Falls back to RecursiveChunkingStrategy if docling is unavailable.
    """

    strategy_name: str = "docling"

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize Docling chunking strategy.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks (not used by HybridChunker).
            embedding_model: HuggingFace model for tokenizer.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.embedding_model = embedding_model
        self._chunker = None
        self._fallback = None
        self._init_chunker()

    def _init_chunker(self) -> None:
        """Initialize the Docling HybridChunker with fallback."""
        try:
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            from transformers import AutoTokenizer

            # Create tokenizer matching the embedding model
            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(self.embedding_model),
                max_tokens=self.chunk_size,
            )

            self._chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,
            )
            logger.debug(
                f"Initialized Docling HybridChunker (model={self.embedding_model}, "
                f"max_tokens={self.chunk_size})"
            )
        except ImportError as e:
            logger.warning(
                f"Docling not available for chunking: {e}. "
                "Falling back to recursive chunking. "
                "Install with: pip install docling"
            )
            self._fallback = RecursiveChunkingStrategy(self.chunk_size, self.chunk_overlap)
        except Exception as e:
            logger.warning(f"Docling chunker initialization failed: {e}. Using fallback.")
            self._fallback = RecursiveChunkingStrategy(self.chunk_size, self.chunk_overlap)

    def split(self, text: str) -> List[str]:
        """Split text using Docling HybridChunker.

        Note: For optimal results, use chunk_docling_document() with
        the actual DoclingDocument object for structure-aware chunking.
        This method converts text to a simple document first.

        Args:
            text: The input text to split.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        if self._fallback is not None:
            return self._fallback.split(text)

        try:
            # For plain text, we need to create a DoclingDocument
            # This is a simplified approach - for best results use chunk_docling_document
            from docling_core.types.doc.document import DoclingDocument

            # Create a minimal document from text
            doc = DoclingDocument(name="text_document")
            doc.add_text(text=text)

            # Chunk the document
            chunks = list(self._chunker.chunk(dl_doc=doc))

            # Extract text with context from chunks
            result = []
            for chunk in chunks:
                # Use contextualize() for enriched text with headings
                chunk_text = self._chunker.contextualize(chunk)
                if chunk_text and chunk_text.strip():
                    result.append(chunk_text.strip())

            return result

        except Exception as e:
            logger.warning(f"Docling chunking failed: {e}. Using fallback.")
            if self._fallback is None:
                self._fallback = RecursiveChunkingStrategy(self.chunk_size, self.chunk_overlap)
            return self._fallback.split(text)

    def chunk_docling_document(self, doc: "DoclingDocument") -> List[str]:
        """Chunk a DoclingDocument with full structure awareness.

        This method provides optimal chunking when you have the actual
        DoclingDocument object from the Docling parser.

        Args:
            doc: DoclingDocument from Docling parser.

        Returns:
            List of context-enriched text chunks.
        """
        if self._fallback is not None:
            # Fallback: export to markdown and use recursive chunking
            content = doc.export_to_markdown()
            return self._fallback.split(content)

        try:
            chunks = list(self._chunker.chunk(dl_doc=doc))

            result = []
            for chunk in chunks:
                # Use contextualize() for enriched text with headings
                chunk_text = self._chunker.contextualize(chunk)
                if chunk_text and chunk_text.strip():
                    result.append(chunk_text.strip())

            logger.debug(f"Docling chunked document into {len(result)} context-aware chunks")
            return result

        except Exception as e:
            logger.warning(f"Docling document chunking failed: {e}. Using markdown fallback.")
            content = doc.export_to_markdown()
            if self._fallback is None:
                self._fallback = RecursiveChunkingStrategy(self.chunk_size, self.chunk_overlap)
            return self._fallback.split(content)


# =============================================================================
# Factory Functions
# =============================================================================

# Strategy registry
_STRATEGY_REGISTRY: Dict[str, type] = {
    "fixed": FixedChunkingStrategy,
    "recursive": RecursiveChunkingStrategy,
    "semantic": SemanticChunkingStrategy,
    "docling": DoclingChunkingStrategy,
}


def get_chunking_strategy(config: Optional[PipelineConfig] = None) -> ChunkingStrategy:
    """Get a chunking strategy instance from configuration.

    Args:
        config: Pipeline configuration. If None, uses defaults.

    Returns:
        A ChunkingStrategy instance.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    if config is None:
        config = PipelineConfig()

    strategy_name = config.chunking_strategy
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    if strategy_name not in _STRATEGY_REGISTRY:
        valid = ", ".join(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Invalid chunking_strategy: '{strategy_name}'. Must be one of: {valid}")

    strategy_class = _STRATEGY_REGISTRY[strategy_name]
    logger.debug(f"Creating {strategy_name} chunking strategy (size={chunk_size}, overlap={chunk_overlap})")

    # DoclingChunkingStrategy needs the embedding model from config
    if strategy_name == "docling":
        embedding_model = config.text_embedding.model
        return DoclingChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )

    return strategy_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_document(
    content: str,
    strategy: ChunkingStrategy,
    metadata: ChunkMetadata,
) -> List[DocumentChunk]:
    """Chunk a document using the specified strategy.

    Args:
        content: The document content to chunk.
        strategy: The chunking strategy to use.
        metadata: Metadata about the source document.

    Returns:
        List of DocumentChunk objects with full metadata.
    """
    import uuid

    chunk_texts = strategy.split(content)

    chunks = []
    for idx, chunk_text in enumerate(chunk_texts):
        # Generate unique chunk ID based on source and index
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{metadata.source_id}:chunk:{idx}"))

        chunk = DocumentChunk(
            id=chunk_id,
            content=chunk_text,
            source_id=metadata.source_id,
            source_type=metadata.source_type,
            source_loc=metadata.source_loc,
            parent_title=metadata.parent_title,
            chunk_index=idx,
            chunking_strategy=strategy.strategy_name,
            metadata=metadata.extra,
        )
        chunks.append(chunk)

    logger.debug(
        f"Chunked document '{metadata.source_loc}' into {len(chunks)} chunks "
        f"using {strategy.strategy_name} strategy"
    )

    return chunks


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    "ChunkingStrategy",
    "FixedChunkingStrategy",
    "RecursiveChunkingStrategy",
    "SemanticChunkingStrategy",
    "DoclingChunkingStrategy",
    "DocumentChunk",
    "ChunkMetadata",
    "get_chunking_strategy",
    "chunk_document",
]
