"""
Pipeline configuration validation for NyRAG.

This module provides validation functions for the ingestion pipeline configuration,
catching errors at startup before processing begins.

Validators:
    - validate_chunking_strategy: Verify chunking strategy is valid
    - validate_embedding_model: Test model loading and availability
    - validate_embedding_dimensions: Check Vespa schema compatibility
    - validate_pipeline_config: Run all validators
"""

from dataclasses import dataclass
from typing import List, Optional

from nyrag.config import Config, PipelineConfig
from nyrag.logger import logger


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    message: str
    warnings: List[str]

    @classmethod
    def success(cls, message: str = "Validation passed", warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(valid=True, message=message, warnings=warnings or [])

    @classmethod
    def failure(cls, message: str, warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(valid=False, message=message, warnings=warnings or [])


class PipelineValidationError(Exception):
    """Raised when pipeline configuration validation fails."""

    def __init__(self, message: str, errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or [message]
        self.warnings = warnings or []


# =============================================================================
# Individual Validators
# =============================================================================


def validate_chunking_strategy(config: PipelineConfig) -> ValidationResult:
    """Validate that the chunking strategy is valid and available.

    Args:
        config: Pipeline configuration to validate.

    Returns:
        ValidationResult with success/failure status.
    """
    valid_strategies = {"fixed", "recursive", "semantic"}
    strategy = config.chunking_strategy

    if strategy not in valid_strategies:
        return ValidationResult.failure(
            f"Invalid chunking_strategy: '{strategy}'. "
            f"Must be one of: {', '.join(sorted(valid_strategies))}"
        )

    # Check for semantic strategy dependencies
    warnings = []
    if strategy == "semantic":
        try:
            import spacy

            try:
                spacy.load("en_core_web_sm")
            except OSError:
                warnings.append(
                    "Semantic chunking requires spacy model 'en_core_web_sm'. "
                    "Install with: python -m spacy download en_core_web_sm. "
                    "Falling back to recursive chunking."
                )
        except ImportError:
            warnings.append(
                "Semantic chunking requires spacy. "
                "Install with: pip install spacy. "
                "Falling back to recursive chunking."
            )

    # Validate chunk_size and chunk_overlap
    if config.chunk_size <= 0:
        return ValidationResult.failure("chunk_size must be greater than 0")

    if config.chunk_overlap < 0:
        return ValidationResult.failure("chunk_overlap must be non-negative")

    if config.chunk_overlap >= config.chunk_size:
        return ValidationResult.failure(
            f"chunk_overlap ({config.chunk_overlap}) must be less than chunk_size ({config.chunk_size})"
        )

    return ValidationResult.success(
        f"Chunking strategy '{strategy}' is valid (size={config.chunk_size}, overlap={config.chunk_overlap})",
        warnings=warnings,
    )


def validate_embedding_model(config: PipelineConfig) -> ValidationResult:
    """Validate that the embedding model can be loaded.

    This performs a lightweight test to verify the model exists and can be initialized.

    Args:
        config: Pipeline configuration to validate.

    Returns:
        ValidationResult with success/failure status.
    """
    model_name = config.text_embedding.model
    device = config.text_embedding.device
    warnings = []

    try:
        from sentence_transformers import SentenceTransformer

        # Check device availability
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            warnings.append(f"CUDA requested but not available. Will fall back to CPU.")
        elif device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                warnings.append(f"MPS requested but not available. Will fall back to CPU.")

        # Try to load the model (this validates the model exists)
        logger.debug(f"Validating embedding model: {model_name}")
        model = SentenceTransformer(model_name, device="cpu")

        # Get embedding dimension for reference
        dim = model.get_sentence_embedding_dimension()

        return ValidationResult.success(
            f"Embedding model '{model_name}' loaded successfully (dim={dim})",
            warnings=warnings,
        )

    except Exception as e:
        return ValidationResult.failure(
            f"Failed to load embedding model '{model_name}': {str(e)}. "
            f"Ensure the model name is valid on HuggingFace Hub or the local path exists.",
            warnings=warnings,
        )


def validate_embedding_dimensions(config: Config, expected_dim: Optional[int] = None) -> ValidationResult:
    """Validate that embedding dimensions match Vespa schema expectations.

    Args:
        config: Full NyRAG configuration.
        expected_dim: Optional expected dimension from Vespa schema. If None, extracted from config.

    Returns:
        ValidationResult with success/failure status.
    """
    pipeline_config = config.get_pipeline_config()
    model_name = pipeline_config.text_embedding.model
    warnings = []

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        actual_dim = model.get_sentence_embedding_dimension()

        # Get expected dimension from config
        if expected_dim is None:
            rag_params = config.rag_params or {}
            expected_dim = rag_params.get("embedding_dim", 384)

        if actual_dim != expected_dim:
            return ValidationResult.failure(
                f"Embedding dimension mismatch: model '{model_name}' produces {actual_dim}-dim vectors, "
                f"but Vespa schema expects {expected_dim}-dim. "
                f"Either update rag_params.embedding_dim to {actual_dim}, or use a model with {expected_dim} dimensions."
            )

        return ValidationResult.success(
            f"Embedding dimensions match: model produces {actual_dim}-dim vectors, schema expects {expected_dim}-dim"
        )

    except Exception as e:
        return ValidationResult.failure(f"Could not validate embedding dimensions: {str(e)}")


def validate_image_embedding(config: PipelineConfig) -> ValidationResult:
    """Validate image embedding configuration if enabled.

    Args:
        config: Pipeline configuration to validate.

    Returns:
        ValidationResult with success/failure status.
    """
    if not config.image_embedding.enabled:
        return ValidationResult.success("Image embedding is disabled")

    model_name = config.image_embedding.model
    device = config.image_embedding.device
    warnings = []

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            warnings.append(f"CUDA requested for image embedding but not available. Will fall back to CPU.")
        elif device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                warnings.append(f"MPS requested for image embedding but not available. Will fall back to CPU.")

        # Try to load the model
        logger.debug(f"Validating image embedding model: {model_name}")
        model = SentenceTransformer(model_name, device="cpu")
        dim = model.get_sentence_embedding_dimension()

        return ValidationResult.success(
            f"Image embedding model '{model_name}' loaded successfully (dim={dim})",
            warnings=warnings,
        )

    except Exception as e:
        return ValidationResult.failure(
            f"Failed to load image embedding model '{model_name}': {str(e)}",
            warnings=warnings,
        )


# =============================================================================
# Main Validation Function
# =============================================================================


def validate_pipeline_config(
    config: Config,
    check_embedding_model: bool = True,
    check_dimensions: bool = True,
) -> ValidationResult:
    """Run all pipeline configuration validators.

    This is the main entry point for validating pipeline configuration.
    It runs all validators and collects results.

    Args:
        config: Full NyRAG configuration.
        check_embedding_model: Whether to validate embedding model loading (can be slow).
        check_dimensions: Whether to validate embedding dimensions match Vespa schema.

    Returns:
        ValidationResult with overall success/failure and aggregated messages.

    Raises:
        PipelineValidationError: If validation fails with raise_on_error=True.
    """
    pipeline_config = config.get_pipeline_config()
    errors = []
    warnings = []

    logger.info("Validating pipeline configuration...")

    # 1. Validate chunking strategy
    chunking_result = validate_chunking_strategy(pipeline_config)
    if not chunking_result.valid:
        errors.append(chunking_result.message)
    warnings.extend(chunking_result.warnings)

    # 2. Validate embedding model (if requested)
    if check_embedding_model:
        embedding_result = validate_embedding_model(pipeline_config)
        if not embedding_result.valid:
            errors.append(embedding_result.message)
        warnings.extend(embedding_result.warnings)

    # 3. Validate embedding dimensions (if requested and embedding model validation passed)
    if check_dimensions and check_embedding_model and not errors:
        dim_result = validate_embedding_dimensions(config)
        if not dim_result.valid:
            errors.append(dim_result.message)
        warnings.extend(dim_result.warnings)

    # 4. Validate image embedding (if enabled)
    if pipeline_config.image_embedding.enabled:
        image_result = validate_image_embedding(pipeline_config)
        if not image_result.valid:
            errors.append(image_result.message)
        warnings.extend(image_result.warnings)

    # Log warnings
    for warning in warnings:
        logger.warning(f"Pipeline validation warning: {warning}")

    # Return result
    if errors:
        error_msg = "Pipeline configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        return ValidationResult.failure(error_msg, warnings=warnings)

    logger.success("Pipeline configuration validation passed")
    return ValidationResult.success("All pipeline configuration checks passed", warnings=warnings)


def validate_and_raise(config: Config, check_embedding_model: bool = True, check_dimensions: bool = True) -> None:
    """Validate pipeline configuration and raise an exception if invalid.

    This is a convenience function for use in startup code that should fail fast.

    Args:
        config: Full NyRAG configuration.
        check_embedding_model: Whether to validate embedding model loading.
        check_dimensions: Whether to validate embedding dimensions.

    Raises:
        PipelineValidationError: If validation fails.
    """
    result = validate_pipeline_config(config, check_embedding_model, check_dimensions)
    if not result.valid:
        raise PipelineValidationError(
            result.message,
            errors=[result.message],
            warnings=result.warnings,
        )


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    "ValidationResult",
    "PipelineValidationError",
    "validate_chunking_strategy",
    "validate_embedding_model",
    "validate_embedding_dimensions",
    "validate_image_embedding",
    "validate_pipeline_config",
    "validate_and_raise",
]
