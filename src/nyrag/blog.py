"""Blog generation module for NyRAG.

This module provides functionality for generating Substack-compatible blog posts
using RAG (Retrieval-Augmented Generation) from multiple sources including notes,
documents, and crawled web content.

Key features:
- Multi-source RAG context retrieval
- YAML-configurable blog templates
- Substack-compatible markdown output
- Background async generation via job queue
"""

import asyncio
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from nyrag.config import Config
from nyrag.logger import get_logger
from nyrag.utils import DEFAULT_EMBEDDING_MODEL, get_vespa_tls_config, make_vespa_client, resolve_vespa_port


logger = get_logger("blog")

# Module-level embedding model (lazy loaded)
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Get or create the embedding model singleton."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _get_openrouter_client() -> AsyncOpenAI:
    """Get AsyncOpenAI client for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    default_headers = {}
    referer = os.getenv("OPENROUTER_REFERRER")
    if referer:
        default_headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_TITLE")
    if title:
        default_headers["X-Title"] = title
    return AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=default_headers or None)


class BlogStatus(str, Enum):
    """Status of a blog post generation."""

    DRAFT = "draft"
    GENERATING = "generating"
    COMPLETE = "complete"
    FAILED = "failed"


class BlogPost(BaseModel):
    """Pydantic model representing a generated blog post."""

    id: str = Field(..., description="Unique identifier for the blog post")
    topic: str = Field(..., description="Topic or title of the blog")
    template: Optional[str] = Field(None, description="Template used for generation")
    content: str = Field(default="", description="Generated markdown content")
    source_notes: List[str] = Field(default_factory=list, description="IDs of source notes used")
    status: BlogStatus = Field(default=BlogStatus.DRAFT, description="Generation status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    error: Optional[str] = Field(None, description="Error message if generation failed")


class BlogTemplate(BaseModel):
    """Pydantic model representing a blog template configuration."""

    name: str = Field(..., description="Template name (e.g., 'tutorial', 'opinion')")
    description: str = Field(..., description="Human-readable description of the template")
    structure: List[str] = Field(..., description="Section structure (e.g., ['intro', 'steps', 'conclusion'])")
    system_prompt: str = Field(..., description="System prompt for LLM generation")
    example_output: Optional[str] = Field(None, description="Example of expected output format")


# Placeholder functions - to be implemented in Phase 5 (US3)


def retrieve_rag_context(topic: str, config: Config, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve RAG context from all sources for blog generation.

    Args:
        topic: The blog topic to search for.
        config: NyRAG configuration.
        limit: Maximum number of context chunks to retrieve.

    Returns:
        List of context chunks with source attribution.
    """
    try:
        # Get embedding model
        rag_params = config.rag_params or {}
        model_name = rag_params.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        model = _get_embedding_model(model_name)

        # Generate query embedding
        embedding = model.encode(topic, convert_to_numpy=True).tolist()

        # Connect to Vespa
        vespa_url = os.getenv("VESPA_URL", "http://localhost")
        vespa_port = resolve_vespa_port(vespa_url)
        cert_path, key_path, ca_cert, verify = get_vespa_tls_config()
        vespa_app = make_vespa_client(vespa_url, vespa_port, cert_path, key_path, ca_cert, verify)

        chunks: List[Dict[str, Any]] = []

        # Query main schema (docs/web)
        main_schema = config.get_schema_name()
        try:
            body = {
                "yql": "select * from sources * where userInput(@query)",
                "query": topic,
                "hits": limit,
                "summary": "top_k_chunks",
                "ranking.profile": "default",
                "input.query(embedding)": embedding,
                "input.query(k)": 3,
            }
            response = vespa_app.query(body=body, schema=main_schema)
            hits = response.json.get("root", {}).get("children", []) or []
            for hit in hits:
                fields = hit.get("fields", {})
                loc = fields.get("loc") or fields.get("id") or ""
                chunk_texts = fields.get("chunks_topk") or []
                hit_score = float(hit.get("relevance", 0.0) or 0.0)
                for chunk in chunk_texts:
                    chunks.append(
                        {
                            "loc": loc,
                            "chunk": chunk,
                            "score": hit_score,
                            "source_type": "document",
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to query main schema: {e}")

        # Query notes schema
        notes_schema = f"{main_schema}notes"
        try:
            body = {
                "yql": "select * from sources * where true",
                "hits": min(limit, 5),
                "ranking.profile": "default",
                "input.query(embedding)": embedding,
            }
            response = vespa_app.query(body=body, schema=notes_schema)
            hits = response.json.get("root", {}).get("children", []) or []
            for hit in hits:
                fields = hit.get("fields", {})
                note_id = fields.get("id", "")
                title = fields.get("title", "")
                content = fields.get("content", "")
                hit_score = float(hit.get("relevance", 0.0) or 0.0)
                chunks.append(
                    {
                        "loc": f"note:{note_id}:{title}",
                        "chunk": content[:2000],  # Limit chunk size
                        "score": hit_score,
                        "source_type": "note",
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to query notes schema: {e}")

        # Sort by score and limit
        chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        chunks = chunks[:limit]

        logger.info(f"Retrieved {len(chunks)} RAG context chunks for topic: {topic[:50]}...")
        return chunks

    except Exception as e:
        logger.error(f"Failed to retrieve RAG context: {e}")
        return []


def build_blog_prompt(
    topic: str,
    context: List[Dict[str, Any]],
    template: Optional["BlogTemplate"] = None,
    instructions: Optional[str] = None,
) -> str:
    """Build the LLM prompt for blog generation.

    Args:
        topic: The blog topic.
        context: RAG context chunks.
        template: Optional blog template to use.
        instructions: Optional user instructions.

    Returns:
        Complete prompt string for the LLM.
    """
    # Build context section
    context_text = "\n\n".join([f"[Source: {c.get('loc', 'unknown')}]\n{c.get('chunk', '')}" for c in context])

    # Default Substack-style system prompt
    system_prompt = """You are an expert blog writer creating engaging, Substack-compatible blog posts.

Your writing style should be:
- Conversational and engaging, as if speaking to a friend
- Well-structured with clear sections and headings
- Informative and backed by the provided context
- Personal with occasional anecdotes or opinions
- Formatted in clean markdown suitable for Substack

Structure your blog post with:
1. A compelling hook/introduction
2. Clear main sections with ## headings
3. Supporting points with evidence from context
4. A memorable conclusion with a call-to-action or takeaway

Use markdown formatting: headers, bullet points, bold/italic for emphasis.
Include attribution to sources when referencing specific information."""

    # Override with template system prompt if provided
    if template:
        system_prompt = template.system_prompt
        # Add template structure guidance
        structure_hint = "\n\nFollow this structure:\n" + "\n".join([f"- {section}" for section in template.structure])
        system_prompt += structure_hint

    # Build user prompt
    user_prompt = f"""Write a blog post about: {topic}

Use the following context to inform your writing:

{context_text or "(No context available - write based on general knowledge)"}
"""

    if instructions:
        user_prompt += f"\n\nAdditional instructions: {instructions}"

    # Combine into full prompt structure for the LLM
    full_prompt = f"""SYSTEM: {system_prompt}

USER: {user_prompt}

Write the complete blog post in markdown format. Start with a title using # heading."""

    return full_prompt


async def generate_blog_content(prompt: str, config: Config) -> str:
    """Generate blog content using the LLM.

    Args:
        prompt: The complete prompt for generation.
        config: NyRAG configuration.

    Returns:
        Generated markdown content.
    """
    model_id = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    timeout_seconds = config.blog_params.timeout_seconds if config.blog_params else 120

    # Parse prompt into system and user parts
    parts = prompt.split("\n\nUSER: ", 1)
    system_content = parts[0].replace("SYSTEM: ", "") if parts else ""
    user_content = parts[1] if len(parts) > 1 else prompt

    # Remove the final instruction line if present
    if "\n\nWrite the complete blog post" in user_content:
        user_parts = user_content.rsplit("\n\nWrite the complete blog post", 1)
        user_content = (
            user_parts[0] + "\n\nWrite the complete blog post in markdown format. Start with a title using # heading."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    try:
        client = _get_openrouter_client()

        # Use streaming for progress, collect full response
        full_content = ""
        stream = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=True,
                max_tokens=4000,
            ),
            timeout=timeout_seconds,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        logger.success(f"Generated blog content: {len(full_content)} characters")
        return full_content

    except asyncio.TimeoutError:
        logger.error(f"Blog generation timed out after {timeout_seconds}s")
        raise RuntimeError(f"Blog generation timed out after {timeout_seconds} seconds")
    except Exception as e:
        logger.error(f"Blog generation failed: {e}")
        raise RuntimeError(f"Blog generation failed: {str(e)}")


def save_blog(blog: BlogPost, config: Config) -> Path:
    """Save a generated blog post to the output directory.

    Args:
        blog: The blog post to save.
        config: NyRAG configuration.

    Returns:
        Path to the saved blog file.
    """
    # Get output path
    output_path = config.get_output_path()
    blogs_dir = config.blog_params.output_path if config.blog_params else "blogs"
    blog_dir = output_path / blogs_dir
    blog_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from blog id
    filename = f"{blog.id}.md"
    file_path = blog_dir / filename

    # Build frontmatter
    frontmatter = f"""---
title: "{blog.topic}"
date: {blog.created_at.isoformat()}
status: {blog.status.value}
template: {blog.template or "default"}
sources:
{chr(10).join([f'  - "{src}"' for src in blog.source_notes]) if blog.source_notes else "  []"}
---

"""

    # Write file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(frontmatter)
        f.write(blog.content)

    logger.success(f"Saved blog {blog.id} to {file_path}")
    return file_path


async def generate_blog_task(
    topic: str,
    config: Config,
    template: Optional[str] = None,
    instructions: Optional[str] = None,
) -> BlogPost:
    """Full orchestration task for blog generation.

    This is the main entry point for async blog generation via the job queue.

    Args:
        topic: The blog topic.
        config: NyRAG configuration.
        template: Optional template name.
        instructions: Optional user instructions.

    Returns:
        The generated blog post.
    """
    blog_id = str(uuid.uuid4())
    blog = BlogPost(
        id=blog_id,
        topic=topic,
        template=template,
        status=BlogStatus.GENERATING,
    )

    try:
        # Step 1: Retrieve RAG context
        logger.info(f"[{blog_id}] Retrieving RAG context for: {topic[:50]}...")
        context = retrieve_rag_context(topic, config, limit=10)

        # Extract source IDs for attribution
        source_notes = [c.get("loc", "") for c in context if c.get("source_type") == "note"]
        blog.source_notes = source_notes[:10]  # Limit sources

        # Step 2: Load template if specified
        template_obj = None
        if template:
            try:
                template_obj = load_template(template, config)
            except NotImplementedError:
                logger.warning("Template loading not implemented, using default prompt")
            except Exception as e:
                logger.warning(f"Failed to load template {template}: {e}")

        # Step 3: Build prompt
        logger.info(f"[{blog_id}] Building blog prompt...")
        prompt = build_blog_prompt(topic, context, template_obj, instructions)

        # Step 4: Generate content
        logger.info(f"[{blog_id}] Generating blog content...")
        content = await generate_blog_content(prompt, config)
        blog.content = content
        blog.status = BlogStatus.COMPLETE

        # Step 5: Save blog
        logger.info(f"[{blog_id}] Saving blog...")
        save_blog(blog, config)

        logger.success(f"[{blog_id}] Blog generation complete!")
        return blog

    except Exception as e:
        blog.status = BlogStatus.FAILED
        blog.error = str(e)
        logger.error(f"[{blog_id}] Blog generation failed: {e}")
        # Still try to save failed blog for debugging
        try:
            save_blog(blog, config)
        except Exception:
            pass
        raise


# Placeholder functions - to be implemented in Phase 5 (US3)


def load_template(template_name: str, config: Config) -> BlogTemplate:
    """Load a blog template from YAML configuration.

    Args:
        template_name: Name of the template to load.
        config: NyRAG configuration.

    Returns:
        The loaded template.
    """
    raise NotImplementedError("To be implemented in Phase 5 (T048)")


def list_templates(config: Config) -> List[str]:
    """List all available blog templates.

    Args:
        config: NyRAG configuration.

    Returns:
        List of template names.
    """
    raise NotImplementedError("To be implemented in Phase 5 (T049)")


def get_blog(blog_id: str, config: Config) -> Optional[BlogPost]:
    """Retrieve a blog post by ID from the output directory.

    Args:
        blog_id: The blog post ID.
        config: NyRAG configuration.

    Returns:
        The blog post if found, None otherwise.
    """
    output_path = config.get_output_path()
    blogs_dir = config.blog_params.output_path if config.blog_params else "blogs"
    blog_path = output_path / blogs_dir / f"{blog_id}.md"

    if not blog_path.exists():
        return None

    try:
        with open(blog_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                import yaml

                frontmatter = yaml.safe_load(parts[1])
                markdown_content = parts[2].strip()

                return BlogPost(
                    id=blog_id,
                    topic=frontmatter.get("title", ""),
                    template=frontmatter.get("template"),
                    content=markdown_content,
                    source_notes=frontmatter.get("sources", []),
                    status=BlogStatus(frontmatter.get("status", "complete")),
                    created_at=datetime.fromisoformat(frontmatter.get("date", datetime.utcnow().isoformat())),
                )

        # No frontmatter, return raw content
        return BlogPost(
            id=blog_id,
            topic="Untitled",
            content=content,
            status=BlogStatus.COMPLETE,
        )

    except Exception as e:
        logger.error(f"Failed to load blog {blog_id}: {e}")
        return None


def get_blog_path(blog_id: str, config: Config) -> Optional[Path]:
    """Get the file path for a blog post.

    Args:
        blog_id: The blog post ID.
        config: NyRAG configuration.

    Returns:
        Path to the blog file if it exists, None otherwise.
    """
    output_path = config.get_output_path()
    blogs_dir = config.blog_params.output_path if config.blog_params else "blogs"
    blog_path = output_path / blogs_dir / f"{blog_id}.md"

    if blog_path.exists():
        return blog_path
    return None
