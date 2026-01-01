"""Document loaders for unified ingestion pipeline.

This module provides document loaders for URLs, PDFs, markdown, and text files.
Supports two backends:
- Docling (default): Superior PDF/DOCX parsing with layout understanding and table structure
- Legacy: LangChain-based loaders (PyPDFLoader, UnstructuredMarkdownLoader)

Includes image extraction support for multimodal document processing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    from nyrag.config import DoclingConfig, PipelineConfig

# LangChain loaders for legacy mode
from langchain_community.document_loaders import (
    PyPDFLoader as LangChainPyPDFLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)


@dataclass
class ExtractedImage:
    """Represents an image extracted from a document."""

    data: bytes  # Raw image bytes
    format: str  # Image format (png, jpg, etc.)
    page: Optional[int] = None  # Page number for PDFs
    alt_text: Optional[str] = None  # Alt text if available
    source_loc: str = ""  # Source document location
    metadata: Dict = field(default_factory=dict)

    def to_pil_image(self) -> Any:
        """Convert to PIL Image object.

        Returns:
            PIL Image object.

        Raises:
            ImportError: If PIL is not installed.
        """
        try:
            from PIL import Image
            import io

            return Image.open(io.BytesIO(self.data))
        except ImportError:
            raise ImportError("PIL is required for image processing. Install with: pip install Pillow")


@dataclass
class LoadedDocument:
    """Represents a loaded document with metadata."""

    content: str
    source_loc: str
    source_type: str
    title: Optional[str] = None
    metadata: Optional[Dict] = None
    images: List[ExtractedImage] = field(default_factory=list)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentLoader(ABC):
    """Base class for document loaders."""

    @abstractmethod
    def load(self, source: str) -> LoadedDocument:
        """Load a document from the given source.

        Args:
            source: Source location (URL, file path, etc.)

        Returns:
            LoadedDocument with content and metadata
        """
        pass


class URLLoader(DocumentLoader):
    """Loader for web URLs using LangChain WebBaseLoader.

    Automatically detects PDF URLs and uses appropriate PDF parsing.
    """

    def __init__(self, extract_images: bool = True):
        """Initialize URL loader.

        Args:
            extract_images: Whether to extract images from PDFs.
        """
        self.extract_images = extract_images

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file.

        Args:
            url: URL to check.

        Returns:
            True if URL appears to be a PDF.
        """
        # Check URL path for .pdf extension
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.path.lower().endswith('.pdf'):
            return True

        # Check Content-Type header
        try:
            import requests
            response = requests.head(url, allow_redirects=True, timeout=10)
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type:
                return True
        except Exception:
            pass

        return False

    def _load_pdf_from_url(self, url: str) -> LoadedDocument:
        """Download and parse PDF from URL.

        Args:
            url: URL to PDF file.

        Returns:
            LoadedDocument with parsed PDF content.
        """
        import tempfile
        import requests

        logger.info(f"[LOADER] Detected PDF URL, downloading: {url}")

        # Download PDF to temp file
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            # Use PDFLoader to parse
            pdf_loader = PDFLoader(extract_images=self.extract_images)
            loaded_doc = pdf_loader.load(tmp_path)

            # Update source location to original URL
            loaded_doc.source_loc = url
            loaded_doc.source_type = "url"

            return loaded_doc
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def load(self, source: str) -> LoadedDocument:
        """Load content from a URL.

        Args:
            source: URL to load

        Returns:
            LoadedDocument with web content
        """
        logger.info(f"[LOADER] Loading URL: {source}")

        # Check if URL points to a PDF
        if self._is_pdf_url(source):
            return self._load_pdf_from_url(source)

        # Standard web page loading
        loader = WebBaseLoader(source)
        docs = loader.load()

        # WebBaseLoader returns a list of LangChain documents
        if not docs:
            raise ValueError(f"No content loaded from URL: {source}")

        # Combine all documents (usually just one for a single URL)
        content = "\n\n".join(doc.page_content for doc in docs)

        # Extract metadata
        metadata = docs[0].metadata if docs else {}
        title = metadata.get("title", "")

        # Log content details
        logger.info(f"[LOADER] URL loaded successfully:")
        logger.info(f"  - Title: {title[:100] if title else '(no title)'}")
        logger.info(f"  - Content length: {len(content)} chars")
        logger.debug(f"  - Content preview: {content[:200]}...")

        return LoadedDocument(
            content=content,
            source_loc=source,
            source_type="url",
            title=title,
            metadata=metadata,
        )


class PDFLoader(DocumentLoader):
    """Loader for PDF files using LangChain PyPDFLoader."""

    def __init__(self, extract_images: bool = True):
        """Initialize PDF loader.

        Args:
            extract_images: Whether to extract images from PDF.
        """
        self.extract_images = extract_images

    def load(self, source: str) -> LoadedDocument:
        """Load content from a PDF file.

        Args:
            source: Path to PDF file

        Returns:
            LoadedDocument with PDF text content and extracted images
        """
        logger.info(f"[LOADER] Loading PDF: {source}")

        # Verify file exists
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        loader = LangChainPyPDFLoader(source)
        docs = loader.load()

        if not docs:
            raise ValueError(f"No content extracted from PDF: {source}")

        # Combine all pages
        content = "\n\n".join(doc.page_content for doc in docs)

        # Extract metadata from first page
        metadata = docs[0].metadata if docs else {}
        metadata["page_count"] = len(docs)

        # Try to extract title (if available in PDF metadata)
        title = metadata.get("title", path.stem)

        # Log content details
        logger.info(f"[LOADER] PDF loaded successfully:")
        logger.info(f"  - Title: {title[:100] if title else '(no title)'}")
        logger.info(f"  - Pages: {len(docs)}")
        logger.info(f"  - Content length: {len(content)} chars")
        logger.debug(f"  - Content preview: {content[:300]}...")

        # Extract images if enabled
        images: List[ExtractedImage] = []
        if self.extract_images:
            logger.info(f"[LOADER] Extracting images from PDF...")
            images = self._extract_images(source)
            if images:
                logger.info(f"[LOADER] Extracted {len(images)} images from PDF")
                for idx, img in enumerate(images):
                    logger.info(f"  - Image {idx+1}: page {img.page}, format={img.format}, size={len(img.data)} bytes")
            else:
                logger.info(f"[LOADER] No images found in PDF")

        return LoadedDocument(
            content=content,
            source_loc=source,
            source_type="pdf",
            title=title,
            metadata=metadata,
            images=images,
        )

    def _extract_images(self, source: str) -> List[ExtractedImage]:
        """Extract images from PDF using PyMuPDF (fitz).

        Args:
            source: Path to PDF file.

        Returns:
            List of ExtractedImage objects.
        """
        images = []
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(source)

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()

                for img_idx, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        images.append(
                            ExtractedImage(
                                data=image_bytes,
                                format=image_ext,
                                page=page_num + 1,
                                source_loc=source,
                                metadata={"xref": xref, "image_index": img_idx},
                            )
                        )
                    except Exception as e:
                        logger.debug(f"Could not extract image {img_idx} from page {page_num}: {e}")

            doc.close()

        except ImportError:
            logger.debug("PyMuPDF (fitz) not installed. PDF image extraction disabled.")
        except Exception as e:
            logger.debug(f"Could not extract images from PDF: {e}")

        return images


class MarkdownLoader(DocumentLoader):
    """Loader for markdown files using LangChain UnstructuredMarkdownLoader."""

    def load(self, source: str) -> LoadedDocument:
        """Load content from a markdown file.

        Args:
            source: Path to markdown file

        Returns:
            LoadedDocument with markdown content
        """
        logger.info(f"[LOADER] Loading Markdown: {source}")

        # Verify file exists
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {source}")

        loader = UnstructuredMarkdownLoader(source)
        docs = loader.load()

        if not docs:
            raise ValueError(f"No content extracted from markdown: {source}")

        # Combine all document elements
        content = "\n\n".join(doc.page_content for doc in docs)

        # Extract metadata
        metadata = docs[0].metadata if docs else {}
        title = path.stem

        # Log content details
        logger.info(f"[LOADER] Markdown loaded successfully:")
        logger.info(f"  - Title: {title[:100] if title else '(no title)'}")
        logger.info(f"  - Content length: {len(content)} chars")
        logger.debug(f"  - Content preview: {content[:200]}...")

        return LoadedDocument(
            content=content,
            source_loc=source,
            source_type="markdown",
            title=title,
            metadata=metadata,
        )


class TextLoader(DocumentLoader):
    """Loader for plain text files."""

    def load(self, source: str) -> LoadedDocument:
        """Load content from a plain text file.

        Args:
            source: Path to text file

        Returns:
            LoadedDocument with text content
        """
        logger.info(f"[LOADER] Loading Text file: {source}")

        # Verify file exists
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {source}")

        # Read file content
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            raise ValueError(f"Empty text file: {source}")

        # Log content details
        logger.info(f"[LOADER] Text file loaded successfully:")
        logger.info(f"  - Title: {path.stem}")
        logger.info(f"  - Content length: {len(content)} chars")
        logger.debug(f"  - Content preview: {content[:200]}...")

        return LoadedDocument(
            content=content,
            source_loc=source,
            source_type="text",
            title=path.stem,
            metadata={},
        )


# =============================================================================
# Docling Document Loader (Superior PDF/DOCX parsing)
# =============================================================================


class DoclingDocumentLoader(DocumentLoader):
    """Document loader using Docling for superior PDF/DOCX parsing.

    Docling provides:
    - Layout-aware document understanding
    - Table structure recognition (97.9% accuracy)
    - Reading order detection
    - Formula and equation preservation
    - Multi-format support (PDF, DOCX, PPTX, HTML, images)

    This loader wraps the langchain-docling integration.
    """

    # Supported file extensions for Docling
    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".pptx", ".ppt",
        ".xlsx", ".xls", ".html", ".htm", ".md",
        ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
    }

    def __init__(
        self,
        docling_config: Optional["DoclingConfig"] = None,
        extract_images: bool = True,
    ):
        """Initialize Docling document loader.

        Args:
            docling_config: Docling-specific configuration.
            extract_images: Whether to extract images from documents.
        """
        self.docling_config = docling_config
        self.extract_images = extract_images
        self._converter = None

    def _get_converter(self):
        """Lazy-load the Docling converter.

        Returns:
            Configured DocumentConverter instance.
        """
        if self._converter is not None:
            return self._converter

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
        except ImportError:
            raise ImportError(
                "Docling is required for this loader. "
                "Install with: pip install docling langchain-docling"
            )

        # Configure pipeline options based on docling_config
        pipeline_options = PdfPipelineOptions()

        if self.docling_config:
            pipeline_options.do_ocr = self.docling_config.ocr_enabled
            pipeline_options.do_table_structure = self.docling_config.table_structure
            pipeline_options.images_scale = 2.0 if self.extract_images else 1.0

        # Enable image generation for pictures and tables
        if self.extract_images:
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True

        # Create converter with configured options
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )

        return self._converter

    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        return source.startswith(("http://", "https://"))

    def _get_source_type(self, source: str) -> str:
        """Determine source type from path or URL."""
        if self._is_url(source):
            from urllib.parse import urlparse
            parsed = urlparse(source)
            path = parsed.path.lower()
        else:
            path = source.lower()

        if path.endswith(".pdf"):
            return "pdf"
        elif path.endswith((".docx", ".doc")):
            return "docx"
        elif path.endswith((".pptx", ".ppt")):
            return "pptx"
        elif path.endswith((".xlsx", ".xls")):
            return "xlsx"
        elif path.endswith((".md", ".markdown")):
            return "markdown"
        elif path.endswith((".html", ".htm")):
            return "html"
        elif path.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            return "image"
        else:
            return "url" if self._is_url(source) else "text"

    def load(self, source: str) -> LoadedDocument:
        """Load a document using Docling.

        Args:
            source: URL or file path to load.

        Returns:
            LoadedDocument with parsed content.
        """
        source_type = self._get_source_type(source)
        logger.info(f"[DOCLING] Loading {source_type}: {source}")

        try:
            converter = self._get_converter()

            # Convert document
            result = converter.convert(source)
            doc = result.document

            # Export to markdown for content
            content = doc.export_to_markdown()

            # Extract title from document
            title = None
            if hasattr(doc, 'title') and doc.title:
                title = doc.title
            elif not self._is_url(source):
                title = Path(source).stem

            # Build metadata
            metadata = {
                "source": source,
                "source_type": source_type,
                "page_count": len(doc.pages) if hasattr(doc, 'pages') else 1,
            }

            # Extract images if enabled
            images: List[ExtractedImage] = []
            if self.extract_images and hasattr(doc, 'pictures'):
                images = self._extract_images_from_doc(doc, source)

            logger.info(f"[DOCLING] Loaded successfully:")
            logger.info(f"  - Title: {title[:100] if title else '(no title)'}")
            logger.info(f"  - Content length: {len(content)} chars")
            logger.info(f"  - Pages: {metadata.get('page_count', 'N/A')}")
            logger.info(f"  - Images: {len(images)}")
            logger.debug(f"  - Content preview: {content[:300]}...")

            return LoadedDocument(
                content=content,
                source_loc=source,
                source_type=source_type,
                title=title,
                metadata=metadata,
                images=images,
            )

        except Exception as e:
            logger.error(f"[DOCLING] Failed to load {source}: {e}")
            raise RuntimeError(f"Docling failed to parse document: {e}")

    def _extract_images_from_doc(self, doc, source: str) -> List[ExtractedImage]:
        """Extract images from Docling document.

        Args:
            doc: Docling Document object.
            source: Source location for metadata.

        Returns:
            List of ExtractedImage objects.
        """
        images = []

        try:
            if not hasattr(doc, 'pictures'):
                logger.debug("Document has no 'pictures' attribute")
                return images

            logger.info(f"Found {len(doc.pictures)} pictures in document")

            for idx, picture in enumerate(doc.pictures):
                try:
                    # Try multiple ways to get image data from Docling picture
                    image_bytes = None

                    # Method 0: Try get_image() method (Docling 2.x API)
                    if hasattr(picture, 'get_image'):
                        try:
                            from io import BytesIO
                            pil_img = picture.get_image(doc)  # Pass document object
                            if pil_img:
                                buffer = BytesIO()
                                pil_img.save(buffer, format='PNG')
                                image_bytes = buffer.getvalue()
                                logger.debug(f"Extracted image {idx} using get_image() method")
                        except Exception as e:
                            logger.debug(f"get_image() failed for picture {idx}: {e}")

                    # Method 1: Direct image attribute
                    if image_bytes is None and hasattr(picture, 'image') and picture.image is not None:
                        img_data = picture.image
                        if hasattr(img_data, 'tobytes'):
                            image_bytes = img_data.tobytes()
                        elif isinstance(img_data, bytes):
                            image_bytes = img_data
                        elif hasattr(img_data, 'read'):  # File-like object
                            image_bytes = img_data.read()

                    # Method 2: Try to get from PIL Image if available
                    if image_bytes is None and hasattr(picture, 'pil_image'):
                        from io import BytesIO
                        buffer = BytesIO()
                        picture.pil_image.save(buffer, format='PNG')
                        image_bytes = buffer.getvalue()

                    # Method 3: Try data attribute
                    if image_bytes is None and hasattr(picture, 'data'):
                        image_bytes = picture.data

                    if image_bytes:
                        images.append(
                            ExtractedImage(
                                data=image_bytes,
                                format="png",
                                page=getattr(picture, 'page_no', None) or getattr(picture, 'page', None),
                                alt_text=getattr(picture, 'caption', None) or getattr(picture, 'text', None),
                                source_loc=source,
                                metadata={"index": idx},
                            )
                        )
                        logger.debug(f"Successfully extracted image {idx} ({len(image_bytes)} bytes)")
                    else:
                        logger.warning(f"Could not extract bytes from picture {idx} - available attrs: {dir(picture)}")

                except Exception as e:
                    logger.warning(f"Could not extract image {idx}: {e}")

        except Exception as e:
            logger.error(f"Could not extract images from document: {e}")

        logger.info(f"Extracted {len(images)} images total")
        return images

    @classmethod
    def is_supported(cls, source: str) -> bool:
        """Check if source format is supported by Docling.

        Args:
            source: File path or URL.

        Returns:
            True if format is supported.
        """
        if source.startswith(("http://", "https://")):
            return True  # Docling can handle URLs

        path = Path(source)
        return path.suffix.lower() in cls.SUPPORTED_EXTENSIONS


# =============================================================================
# Factory Functions
# =============================================================================


def get_loader_for_source(
    source_type: str,
    extract_images: bool = True,
    pipeline_config: Optional["PipelineConfig"] = None,
) -> DocumentLoader:
    """Factory function to get appropriate loader for source type.

    Args:
        source_type: Type of source ("url", "pdf", "markdown", "text", "note", "docx")
        extract_images: Whether to extract images from documents.
        pipeline_config: Pipeline configuration (determines Docling vs legacy).

    Returns:
        DocumentLoader instance for the given source type

    Raises:
        ValueError: If source_type is not supported
    """
    # Check if we should use Docling
    use_docling = True  # Default to Docling
    docling_config = None

    if pipeline_config is not None:
        use_docling = pipeline_config.uses_docling()
        docling_config = pipeline_config.docling if use_docling else None

    # Docling-supported types
    docling_types = {"pdf", "docx", "pptx", "xlsx", "html", "image"}

    if use_docling and source_type in docling_types:
        logger.debug(f"[LOADER] Using Docling for {source_type}")
        return DoclingDocumentLoader(
            docling_config=docling_config,
            extract_images=extract_images,
        )

    # URL handling - use Docling if it's a document URL
    if source_type == "url":
        if use_docling:
            # DoclingDocumentLoader handles URLs natively
            return DoclingDocumentLoader(
                docling_config=docling_config,
                extract_images=extract_images,
            )
        else:
            return URLLoader(extract_images=extract_images)

    # Legacy loaders for remaining types
    if source_type == "pdf":
        return PDFLoader(extract_images=extract_images)

    loaders = {
        "markdown": MarkdownLoader,
        "text": TextLoader,
        "note": TextLoader,  # Notes are treated as plain text
    }

    loader_class = loaders.get(source_type)
    if loader_class is None:
        raise ValueError(
            f"Unsupported source type: {source_type}. "
            f"Supported types: url, pdf, docx, pptx, markdown, text, note"
        )

    return loader_class()


def get_docling_loader(
    docling_config: Optional["DoclingConfig"] = None,
    extract_images: bool = True,
) -> DoclingDocumentLoader:
    """Get a Docling document loader directly.

    Args:
        docling_config: Docling-specific configuration.
        extract_images: Whether to extract images.

    Returns:
        DoclingDocumentLoader instance.
    """
    return DoclingDocumentLoader(
        docling_config=docling_config,
        extract_images=extract_images,
    )
