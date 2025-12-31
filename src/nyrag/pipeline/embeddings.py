"""
Text and image embedding wrappers for the NyRAG ingestion pipeline.

This module provides pluggable embedding models using LangChain and SentenceTransformers.
Models can be configured via YAML configuration without code changes.

Supported embedding types:
    - Text embeddings: HuggingFace/SentenceTransformer models (local only)
    - Image embeddings: CLIP models via SentenceTransformers (future/P3)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from nyrag.config import ImageEmbeddingConfig, PipelineConfig, TextEmbeddingConfig
from nyrag.logger import logger


# =============================================================================
# Embedding Base Class
# =============================================================================


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""

    model_name: str = "base"
    embedding_dim: int = 0

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        pass

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension for this model.

        Returns:
            Integer dimension of embedding vectors.
        """
        return self.embedding_dim


# =============================================================================
# Text Embedder
# =============================================================================


class TextEmbedder(BaseEmbedder):
    """Text embedding using HuggingFace/SentenceTransformer models.

    This wrapper provides a consistent interface for text embeddings with:
    - Automatic dimension detection
    - Device selection (cpu/cuda/mps)
    - Configurable batch sizes
    - Local-only operation (no API dependencies)

    Example:
        >>> config = TextEmbeddingConfig(model="all-MiniLM-L6-v2", device="cuda")
        >>> embedder = TextEmbedder(config)
        >>> vectors = embedder.embed_texts(["Hello world", "Test text"])
    """

    def __init__(self, config: Optional[TextEmbeddingConfig] = None):
        """Initialize the text embedder.

        Args:
            config: Text embedding configuration. If None, uses defaults.
        """
        if config is None:
            config = TextEmbeddingConfig()

        self.config = config
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.device = config.device

        # Load the model
        self._model = self._load_model()

        # Auto-detect embedding dimension
        self.embedding_dim = self._detect_dimension()

        logger.info(
            f"Initialized TextEmbedder: model='{self.model_name}', "
            f"dim={self.embedding_dim}, device='{self.device}', batch_size={self.batch_size}"
        )

    def _load_model(self) -> Any:
        """Load the SentenceTransformer model.

        Returns:
            Loaded model instance.

        Raises:
            RuntimeError: If model cannot be loaded.
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Map device string to appropriate value
            device = self._resolve_device()

            model = SentenceTransformer(self.model_name, device=device)
            logger.debug(f"Loaded SentenceTransformer model '{self.model_name}' on device '{device}'")
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}") from e

    def _resolve_device(self) -> str:
        """Resolve the device string to a valid device.

        Returns:
            Device string for SentenceTransformer.
        """
        import torch

        if self.device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif self.device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        else:
            return "cpu"

    def _detect_dimension(self) -> int:
        """Auto-detect the embedding dimension from the model.

        Returns:
            Embedding dimension.
        """
        try:
            # SentenceTransformer provides this method
            dim = self._model.get_sentence_embedding_dimension()
            logger.debug(f"Auto-detected embedding dimension: {dim}")
            return dim
        except Exception as e:
            logger.warning(f"Could not auto-detect embedding dimension: {e}. Using default 384.")
            return 384

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts with batching.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        encode_kwargs: Dict[str, Any] = {
            "convert_to_numpy": True,
            "show_progress_bar": False,
            "batch_size": self.batch_size,
        }

        embeddings = self._model.encode(texts, **encode_kwargs)
        return [vec.tolist() for vec in embeddings]

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        result = self.embed_texts([text])
        return result[0] if result else []

    def embed_with_content(self, content: str, chunks: List[str]) -> tuple:
        """Embed content and its chunks in a single batch call.

        This is optimized for the common pattern of embedding both
        the full content and its chunks together.

        Args:
            content: Full document content.
            chunks: List of chunk texts.

        Returns:
            Tuple of (content_embedding, chunk_embeddings).
        """
        all_texts = [content, *chunks]
        all_embeddings = self.embed_texts(all_texts)

        content_embedding = all_embeddings[0]
        chunk_embeddings = all_embeddings[1:]

        return content_embedding, chunk_embeddings

    def __repr__(self) -> str:
        return (
            f"TextEmbedder(model='{self.model_name}', dim={self.embedding_dim}, "
            f"device='{self.device}', batch_size={self.batch_size})"
        )


# =============================================================================
# Image Embedder (P3 - Infrastructure for future multimodal search)
# =============================================================================


class ImageEmbedder(BaseEmbedder):
    """Image embedding using CLIP models via SentenceTransformers.

    This wrapper provides image embeddings for future multimodal search:
    - CLIP-based image-text alignment
    - Automatic resizing to max dimension
    - Local-only operation

    Note: This is P3 priority infrastructure. Full image search is a future feature.

    Example:
        >>> config = ImageEmbeddingConfig(enabled=True, model="clip-ViT-B-32")
        >>> embedder = ImageEmbedder(config)
        >>> vectors = embedder.embed_images([image1, image2])
    """

    def __init__(self, config: Optional[ImageEmbeddingConfig] = None):
        """Initialize the image embedder.

        Args:
            config: Image embedding configuration. If None, uses defaults.
        """
        if config is None:
            config = ImageEmbeddingConfig()

        self.config = config
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.device = config.device
        self.max_dimension = config.max_dimension
        self.enabled = config.enabled

        if not self.enabled:
            self._model = None
            self.embedding_dim = 512  # CLIP default
            logger.debug("ImageEmbedder initialized but disabled")
            return

        # Load the model
        self._model = self._load_model()

        # Auto-detect embedding dimension
        self.embedding_dim = self._detect_dimension()

        logger.info(
            f"Initialized ImageEmbedder: model='{self.model_name}', "
            f"dim={self.embedding_dim}, device='{self.device}'"
        )

    def _load_model(self) -> Any:
        """Load the CLIP model via SentenceTransformers.

        Returns:
            Loaded model instance.

        Raises:
            RuntimeError: If model cannot be loaded.
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Resolve device
            device = self._resolve_device()

            model = SentenceTransformer(self.model_name, device=device)
            logger.debug(f"Loaded CLIP model '{self.model_name}' on device '{device}'")
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load image embedding model '{self.model_name}': {e}") from e

    def _resolve_device(self) -> str:
        """Resolve the device string to a valid device."""
        import torch

        if self.device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif self.device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        else:
            return "cpu"

    def _detect_dimension(self) -> int:
        """Auto-detect the embedding dimension from the model."""
        try:
            dim = self._model.get_sentence_embedding_dimension()
            logger.debug(f"Auto-detected image embedding dimension: {dim}")
            return dim
        except Exception as e:
            logger.warning(f"Could not auto-detect image embedding dimension: {e}. Using default 512.")
            return 512

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts for text-to-image search (CLIP text encoder).

        Args:
            texts: List of text queries for image search.

        Returns:
            List of embedding vectors.
        """
        if not self.enabled or self._model is None:
            raise RuntimeError("ImageEmbedder is not enabled. Set image_embedding.enabled=true in config.")

        if not texts:
            return []

        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text for image search."""
        result = self.embed_texts([text])
        return result[0] if result else []

    def embed_images(self, images: List[Any]) -> List[List[float]]:
        """Embed a list of images.

        Args:
            images: List of PIL Image objects.

        Returns:
            List of embedding vectors.

        Note: Images should be PIL Image objects. Preprocessing (resize) is done internally.
        """
        if not self.enabled or self._model is None:
            raise RuntimeError("ImageEmbedder is not enabled. Set image_embedding.enabled=true in config.")

        if not images:
            return []

        # Resize images if needed
        processed_images = [self._preprocess_image(img) for img in images]

        embeddings = self._model.encode(
            processed_images,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return [vec.tolist() for vec in embeddings]

    def embed_image(self, image: Any) -> List[float]:
        """Embed a single image.

        Args:
            image: PIL Image object.

        Returns:
            Embedding vector.
        """
        result = self.embed_images([image])
        return result[0] if result else []

    def _preprocess_image(self, image: Any) -> Any:
        """Preprocess an image (resize to max dimension).

        Args:
            image: PIL Image object.

        Returns:
            Preprocessed PIL Image.
        """
        try:
            from PIL import Image

            if not isinstance(image, Image.Image):
                raise ValueError("Expected PIL Image object")

            # Resize if larger than max dimension
            width, height = image.size
            max_dim = max(width, height)

            if max_dim > self.max_dimension:
                scale = self.max_dimension / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

            return image

        except ImportError:
            logger.warning("PIL not available for image preprocessing")
            return image

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"ImageEmbedder(model='{self.model_name}', dim={self.embedding_dim}, status={status})"


# =============================================================================
# Factory Functions
# =============================================================================


def get_text_embedder(config: Optional[Union[PipelineConfig, TextEmbeddingConfig]] = None) -> TextEmbedder:
    """Get a text embedder instance from configuration.

    Args:
        config: Either a PipelineConfig (extracts text_embedding) or
                TextEmbeddingConfig directly. If None, uses defaults.

    Returns:
        A TextEmbedder instance.
    """
    if config is None:
        text_config = TextEmbeddingConfig()
    elif isinstance(config, PipelineConfig):
        text_config = config.text_embedding
    elif isinstance(config, TextEmbeddingConfig):
        text_config = config
    else:
        raise TypeError(f"Expected PipelineConfig or TextEmbeddingConfig, got {type(config)}")

    return TextEmbedder(text_config)


def get_image_embedder(config: Optional[Union[PipelineConfig, ImageEmbeddingConfig]] = None) -> ImageEmbedder:
    """Get an image embedder instance from configuration.

    Args:
        config: Either a PipelineConfig (extracts image_embedding) or
                ImageEmbeddingConfig directly. If None, uses defaults.

    Returns:
        An ImageEmbedder instance.
    """
    if config is None:
        image_config = ImageEmbeddingConfig()
    elif isinstance(config, PipelineConfig):
        image_config = config.image_embedding
    elif isinstance(config, ImageEmbeddingConfig):
        image_config = config
    else:
        raise TypeError(f"Expected PipelineConfig or ImageEmbeddingConfig, got {type(config)}")

    return ImageEmbedder(image_config)


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    "BaseEmbedder",
    "TextEmbedder",
    "ImageEmbedder",
    "get_text_embedder",
    "get_image_embedder",
]
