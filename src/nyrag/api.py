import asyncio
import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from nyrag.config import Config
from nyrag.logger import get_logger
from nyrag.blog import (
    list_templates as blog_list_templates,
    generate_blog_task,
    get_blog,
    get_blog_path,
)
from nyrag.notes import (
    list_notes as notes_list_notes,
    get_note as notes_get_note,
    update_note as notes_update_note,
    delete_note as notes_delete_note,
)
from nyrag.jobs import get_job_queue
from nyrag.utils import (
    DEFAULT_EMBEDDING_MODEL,
    get_vespa_tls_config,
    is_vespa_cloud,
    make_vespa_client,
    resolve_vespa_cloud_mtls_paths,
    resolve_vespa_port,
)


DEFAULT_ENDPOINT = "http://localhost:8080"
DEFAULT_RANKING = "default"
DEFAULT_SUMMARY = "top_k_chunks"


class SearchRequest(BaseModel):
    query: str = Field(..., description="User query string")
    hits: int = Field(10, description="Number of Vespa hits to return")
    k: int = Field(3, description="Top-k chunks to keep per hit")
    ranking: Optional[str] = Field(None, description="Ranking profile to use (defaults to schema default)")
    summary: Optional[str] = Field(None, description="Document summary to request (defaults to top_k_chunks)")


def _resolve_mtls_paths(vespa_url: str, project_folder: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    cert_env = (os.getenv("VESPA_CLIENT_CERT") or "").strip() or None
    key_env = (os.getenv("VESPA_CLIENT_KEY") or "").strip() or None

    if not is_vespa_cloud(vespa_url):
        return cert_env, key_env

    if cert_env or key_env:
        if not (cert_env and key_env):
            raise RuntimeError("Vespa Cloud requires both VESPA_CLIENT_CERT and VESPA_CLIENT_KEY.")
        return cert_env, key_env

    if not project_folder:
        raise RuntimeError(
            "Vespa Cloud mTLS credentials not found. "
            "Export VESPA_CLIENT_CERT and VESPA_CLIENT_KEY with the paths to these files."
        )

    cert_path, key_path = resolve_vespa_cloud_mtls_paths(project_folder)
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    raise RuntimeError(
        "Vespa Cloud mTLS credentials not found at "
        f"{cert_path} and {key_path}. "
        "Export VESPA_CLIENT_CERT and VESPA_CLIENT_KEY with the paths to these files."
    )


def _load_settings() -> Dict[str, Any]:
    """Load schema, model, and Vespa connection from env or YAML config."""
    config_path = os.getenv("NYRAG_CONFIG")
    vespa_url = (os.getenv("VESPA_URL") or "").strip() or "http://localhost"
    vespa_port = resolve_vespa_port(vespa_url)

    if config_path and Path(config_path).exists():
        cfg = Config.from_yaml(config_path)
        rag_params = cfg.rag_params or {}
        return {
            "app_package_name": cfg.get_app_package_name(),
            "schema_name": cfg.get_schema_name(),
            "embedding_model": rag_params.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            "vespa_url": vespa_url,
            "vespa_port": vespa_port,
        }

    return {
        "app_package_name": None,
        "schema_name": os.getenv("VESPA_SCHEMA", "nyragwebrag"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "vespa_url": vespa_url,
        "vespa_port": vespa_port,
    }


settings = _load_settings()
logger = get_logger("api")
app = FastAPI(title="nyrag API", version="0.1.0")
model = SentenceTransformer(settings["embedding_model"])

# Get mTLS credentials (with Vespa Cloud fallback)
_cert, _key = _resolve_mtls_paths(settings["vespa_url"], settings.get("app_package_name"))
_, _, _ca, _verify = get_vespa_tls_config()

vespa_app = make_vespa_client(
    settings["vespa_url"],
    settings["vespa_port"],
    _cert,
    _key,
    _ca,
    _verify,
)

base_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")


def _deep_find_numeric_field(obj: Any, key: str) -> Optional[float]:
    if isinstance(obj, dict):
        if key in obj:
            value = obj.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return None
        for v in obj.values():
            found = _deep_find_numeric_field(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _deep_find_numeric_field(item, key)
            if found is not None:
                return found
    return None


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    """Return simple corpus statistics from Vespa (documents and chunks)."""
    doc_count: Optional[int] = None
    chunk_count: Optional[int] = None

    try:
        res = vespa_app.query(
            body={"yql": "select * from sources * where true", "hits": 0},
            schema=settings["schema_name"],
        )
        total = res.json.get("root", {}).get("fields", {}).get("totalCount")
        if isinstance(total, int):
            doc_count = total
        elif isinstance(total, str) and total.isdigit():
            doc_count = int(total)
    except Exception as e:
        logger.warning(f"Failed to fetch Vespa doc count: {e}")

    try:
        # Requires schema field `chunk_count` (added in this repo); if absent, this will likely return null.
        yql = "select * from sources * where true | " "all(group(1) each(output(count(), sum(chunk_count))))"
        res = vespa_app.query(
            body={"yql": yql, "hits": 0},
            schema=settings["schema_name"],
        )
        sum_value = _deep_find_numeric_field(res.json, "sum(chunk_count)")
        if sum_value is None:
            sum_value = _deep_find_numeric_field(res.json, "sum(chunk_count())")
        if sum_value is not None:
            chunk_count = int(sum_value)
    except Exception as e:
        logger.warning(f"Failed to fetch Vespa chunk count: {e}")

    return {
        "schema": settings["schema_name"],
        "documents": doc_count,
        "chunks": chunk_count,
    }


@app.post("/search")
async def search(req: SearchRequest) -> Dict[str, Any]:
    """Query Vespa using YQL with a precomputed query embedding."""
    embedding = model.encode(req.query, convert_to_numpy=True).tolist()
    body = {
        "yql": "select * from sources * where userInput(@query)",
        "query": req.query,
        "hits": req.hits,
        "summary": req.summary or DEFAULT_SUMMARY,
        "ranking.profile": req.ranking or DEFAULT_RANKING,
        "input.query(embedding)": embedding,
        "input.query(k)": req.k,
    }
    vespa_response = vespa_app.query(body=body, schema=settings["schema_name"])

    status_code = getattr(vespa_response, "status_code", 200)
    if status_code >= 400:
        detail = getattr(vespa_response, "json", vespa_response)
        raise HTTPException(status_code=status_code, detail=detail)

    return vespa_response.json


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation messages as list of {role, content} dicts",
    )
    hits: int = Field(5, description="Number of Vespa hits to retrieve")
    k: int = Field(3, description="Top-k chunks per hit to keep")
    query_k: int = Field(
        3,
        ge=0,
        description="Number of alternate search queries to generate with the LLM",
    )
    model: Optional[str] = Field(None, description="OpenRouter model id (optional, uses env default if set)")


def _fetch_chunks(query: str, hits: int, k: int) -> List[Dict[str, Any]]:
    embedding = model.encode(query, convert_to_numpy=True).tolist()
    body = {
        "yql": "select * from sources * where userInput(@query)",
        "query": query,
        "hits": hits,
        "summary": DEFAULT_SUMMARY,
        "ranking.profile": DEFAULT_RANKING,
        "input.query(embedding)": embedding,
        "input.query(k)": k,
        "presentation.summaryFeatures": True,
    }
    vespa_response = vespa_app.query(body=body, schema=settings["schema_name"])
    status_code = getattr(vespa_response, "status_code", 200)
    if status_code >= 400:
        detail = getattr(vespa_response, "json", vespa_response)
        raise HTTPException(status_code=status_code, detail=detail)

    hits_data = vespa_response.json.get("root", {}).get("children", []) or []
    chunks: List[Dict[str, Any]] = []
    for hit in hits_data:
        fields = hit.get("fields", {}) or {}
        loc = fields.get("loc") or fields.get("id") or ""
        chunk_texts = fields.get("chunks_topk") or []
        hit_score_raw = hit.get("relevance", 0.0)
        logger.info(f"Hit loc={loc} score={hit_score_raw} chunks={len(chunk_texts)}")
        try:
            hit_score = float(hit_score_raw)
        except (TypeError, ValueError):
            hit_score = 0.0
        summary_features = (
            hit.get("summaryfeatures") or hit.get("summaryFeatures") or fields.get("summaryfeatures") or {}
        )
        chunk_score_raw = summary_features.get("best_chunk_score", hit_score)
        logger.info(f"  best_chunk_score={chunk_score_raw}")
        try:
            chunk_score = float(chunk_score_raw)
        except (TypeError, ValueError):
            chunk_score = hit_score

        for chunk in chunk_texts:
            chunks.append(
                {
                    "loc": loc,
                    "chunk": chunk,
                    "score": chunk_score,
                    "hit_score": hit_score,
                    "source_query": query,
                }
            )
    return chunks


async def _fetch_chunks_async(query: str, hits: int, k: int) -> List[Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(_fetch_chunks, query, hits, k))


def _get_openrouter_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    default_headers = {}
    referer = os.getenv("OPENROUTER_REFERRER")
    if referer:
        default_headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_TITLE")
    if title:
        default_headers["X-Title"] = title
    return AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=default_headers or None)


def _extract_message_text(content: Any) -> str:
    """Handle OpenAI response content that may be str or list of text blocks."""
    if content is None:
        return ""
    if isinstance(content, dict) and "text" in content:
        return str(content.get("text", ""))
    if hasattr(content, "text"):
        return str(getattr(content, "text", ""))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif hasattr(part, "text"):
                texts.append(str(getattr(part, "text", "")))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join([t for t in texts if t])
    return str(content)


async def _generate_search_queries_stream(
    user_message: str,
    model_id: str,
    num_queries: int,
    hits: int,
    k: int,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Use the chat LLM to propose focused search queries grounded in retrieved chunks."""
    if num_queries <= 0:
        yield "result", []
        return

    grounding_chunks = (await _fetch_chunks_async(user_message, hits=hits, k=k))[:5]
    grounding_text = "\n".join(f"- [{c.get('loc','')}] {c.get('chunk','')}" for c in grounding_chunks)

    system_prompt = (
        "You generate concise, to-the-point search queries that help retrieve"
        " factual context for answering the user."
        " Do not change the meaning of the question."
        " Do not introduce any new information, words, concepts, or ideas."
        " Do not add any new words."
        " Prefer to reuse the provided context to stay on-topic."
        "Return only valid JSON."
    )

    # Build conversation context if history exists
    conversation_context = ""
    if history:
        conversation_context = "Previous conversation:\n"
        for msg in history[-4:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            conversation_context += f"{role}: {content}\n"
        conversation_context += "\n"

    user_prompt = (
        f"{conversation_context}"
        f"Create {num_queries} diverse, specific search queries (max 12 words each)"
        f' that would retrieve evidence to answer:\n"{user_message}".\n'
        f"Grounding context:\n{grounding_text or '(no context found)'}\n"
        'Respond as a JSON object like {"queries": ["query 1", "query 2"]}.'
    )

    full_content = ""
    try:
        client = _get_openrouter_client()
        stream = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            stream=True,
            extra_body={"reasoning": {"enabled": True}},
        )

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            reasoning = getattr(delta, "reasoning", None)
            reasoning_text = _extract_message_text(reasoning)
            if reasoning_text:
                yield "thinking", reasoning_text

            content_piece = _extract_message_text(getattr(delta, "content", None))
            if content_piece:
                full_content += content_piece
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    queries: List[str] = []
    try:
        parsed = json.loads(full_content)
        candidates = parsed.get("queries") if isinstance(parsed, dict) else parsed
        if isinstance(candidates, list):
            queries = [str(q).strip() for q in candidates if str(q).strip()]
    except Exception:
        queries = []

    # Fallback: try to parse line-separated text if JSON parsing fails
    if not queries:
        for line in full_content.splitlines():
            candidate = line.strip(" -•\t")
            if candidate:
                queries.append(candidate)

    cleaned: List[str] = []
    seen: Set[str] = set()
    for q in queries:
        q_norm = q.strip()
        key = q_norm.lower()
        if q_norm and key not in seen:
            cleaned.append(q_norm)
            seen.add(key)
        if len(cleaned) >= num_queries:
            break
    yield "result", cleaned


async def _prepare_queries_stream(
    user_message: str,
    model_id: str,
    query_k: int,
    hits: int,
    k: int,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Build the list of queries (original + enhanced) for retrieval."""
    enhanced = []
    async for event_type, payload in _generate_search_queries_stream(
        user_message, model_id, query_k, hits=hits, k=k, history=history
    ):
        if event_type == "thinking":
            yield "thinking", payload
        elif event_type == "result":
            enhanced = payload

    queries = [user_message] + enhanced

    deduped: List[str] = []
    seen: Set[str] = set()
    for q in queries:
        q_norm = q.strip()
        key = q_norm.lower()
        if q_norm and key not in seen:
            deduped.append(q_norm)
            seen.add(key)
    logger.info(f"Search queries ({len(deduped)}): {deduped}")
    yield "result", deduped


async def _prepare_queries(user_message: str, model_id: str, query_k: int, hits: int, k: int) -> List[str]:
    queries = []
    async for event_type, payload in _prepare_queries_stream(user_message, model_id, query_k, hits, k):
        if event_type == "result":
            queries = payload
    return queries


async def _fuse_chunks(queries: List[str], hits: int, k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Search Vespa for each query and return fused, deduped chunks."""
    all_chunks: List[Dict[str, Any]] = []
    logger.info(f"Fetching chunks for {len(queries)} queries")

    tasks = [_fetch_chunks_async(q, hits=hits, k=k) for q in queries]
    results = await asyncio.gather(*tasks)
    for res in results:
        all_chunks.extend(res)

    logger.info(f"Fetched total {len(all_chunks)} chunks from Vespa")
    if not all_chunks:
        return queries, []

    max_context = hits * k
    if max_context <= 0:
        max_context = len(all_chunks)

    # Aggregate duplicates (same loc+chunk) and average their scores.
    aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chunk in all_chunks:
        key = (chunk.get("loc", ""), chunk.get("chunk", ""))
        score = float(chunk.get("score", chunk.get("hit_score", 0.0)) or 0.0)
        hit_score = float(chunk.get("hit_score", 0.0) or 0.0)
        source_query = chunk.get("source_query")

        if key not in aggregated:
            aggregated[key] = {
                "loc": key[0],
                "chunk": key[1],
                "score_sum": score,
                "hit_sum": hit_score,
                "count": 1,
                "source_queries": [source_query] if source_query else [],
            }
        else:
            agg = aggregated[key]
            agg["score_sum"] += score
            agg["hit_sum"] += hit_score
            agg["count"] += 1
            if source_query and source_query not in agg["source_queries"]:
                agg["source_queries"].append(source_query)

    fused: List[Dict[str, Any]] = []
    for agg in aggregated.values():
        count = max(agg.pop("count", 1), 1)
        agg["score"] = agg.pop("score_sum", 0.0) / count
        agg["hit_score"] = agg.pop("hit_sum", 0.0) / count
        sources = agg.get("source_queries") or []
        agg["source_query"] = sources[0] if sources else ""
        fused.append(agg)

    fused.sort(key=lambda c: c.get("score", c.get("hit_score", 0.0)), reverse=True)
    fused = fused[:max_context]

    return queries, fused


async def _call_openrouter(context: List[Dict[str, str]], user_message: str, model_id: str) -> str:
    system_prompt = (
        "You are a helpful assistant. "
        "Answer user's question using only the provided context. "
        "Provide elaborate and informative answers where possible. "
        "If the context is insufficient, say you don't know."
    )
    context_text = "\n\n".join([f"[{c.get('loc','')}] {c.get('chunk','')}" for c in context])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Grounding context:\n{context_text}\n\nQuestion: {user_message}",
        },
    ]

    try:
        client = _get_openrouter_client()
        resp = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            extra_body={"reasoning": {"enabled": True}},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _extract_message_text(resp.choices[0].message.content)


async def _openrouter_stream(
    context: List[Dict[str, str]],
    user_message: str,
    model_id: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Tuple[str, str], None]:
    system_prompt = (
        "You are a helpful assistant. Answer using only the provided context. "
        "If the context is insufficient, say you don't know."
    )
    context_text = "\n\n".join([f"[{c.get('loc','')}] {c.get('chunk','')}" for c in context])

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided
    if history:
        messages.extend(history)

    # Add current user message with context
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {user_message}",
        }
    )

    try:
        client = _get_openrouter_client()
        stream = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            extra_body={"reasoning": {"enabled": True}},
        )

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            reasoning = _extract_message_text(getattr(delta, "reasoning", None))
            if reasoning:
                yield "thinking", reasoning

            content_piece = _extract_message_text(getattr(delta, "content", None))
            if content_piece:
                yield "token", content_piece
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    model_id = req.model or os.getenv("OPENROUTER_MODEL")
    queries, chunks = await _fuse_chunks(
        await _prepare_queries(req.message, model_id, req.query_k, hits=req.hits, k=req.k),
        hits=req.hits,
        k=req.k,
    )
    if not chunks:
        return {"answer": "No relevant context found.", "chunks": []}
    answer = await _call_openrouter(chunks, req.message, model_id)
    return {"answer": answer, "chunks": chunks, "queries": queries}


# Blog generation intent patterns
BLOG_INTENT_PATTERNS = [
    r"\b(generate|create|write|draft|make)\s+(a\s+)?(blog|post|article)\b",
    r"\b(blog|post|article)\s+(about|on|for)\b",
    r"\bturn\s+(this|these|my)\s+(notes?|content)?\s*into\s+(a\s+)?(blog|post|article)\b",
]


def _detect_blog_intent(message: str) -> Optional[str]:
    """Detect if the user wants to generate a blog post.

    Args:
        message: User's chat message.

    Returns:
        Extracted topic if blog intent detected, None otherwise.
    """
    message_lower = message.lower()

    for pattern in BLOG_INTENT_PATTERNS:
        if re.search(pattern, message_lower):
            # Extract topic: everything after the pattern or after "about/on"
            topic_match = re.search(r"(?:about|on|for|:)\s+(.+?)(?:\.|$)", message, re.IGNORECASE)
            if topic_match:
                return topic_match.group(1).strip()
            # Fallback: use the whole message as topic
            # Remove the command part
            cleaned = re.sub(r"^(please\s+)?(generate|create|write|draft|make)\s+(a\s+)?(blog|post|article)\s*(about|on|for)?\s*", "", message, flags=re.IGNORECASE)
            return cleaned.strip() if cleaned.strip() else message

    return None


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    model_id = req.model or os.getenv("OPENROUTER_MODEL")

    # Check for blog generation intent
    blog_topic = _detect_blog_intent(req.message)

    async def event_stream():
        # If blog intent detected, trigger blog generation
        if blog_topic:
            yield f"data: {json.dumps({'type': 'status', 'payload': 'Blog generation requested...'})}\n\n"

            config = _get_config()
            if not config:
                no_cfg_msg = "Sorry, I cannot generate blogs without a config. Please set NYRAG_CONFIG."
                yield f"data: {json.dumps({'type': 'token', 'payload': no_cfg_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            try:
                job_queue = get_job_queue()
                job_id = await job_queue.submit(
                    job_type="blog_generation",
                    coro=generate_blog_task(
                        topic=blog_topic,
                        config=config,
                        template=None,  # Could parse template from message
                        instructions=None,
                    ),
                )

                msg = f"I'm generating a blog post about **{blog_topic}**. This may take a minute or two..."
                yield f"data: {json.dumps({'type': 'token', 'payload': msg})}\n\n"
                blog_job_evt = {"type": "blog_job", "payload": {"job_id": job_id, "topic": blog_topic}}
                yield f"data: {json.dumps(blog_job_evt)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            except Exception as e:
                logger.error(f"Failed to start blog generation: {e}")
                err_msg = f"Sorry, I encountered an error starting blog generation: {str(e)}"
                yield f"data: {json.dumps({'type': 'token', 'payload': err_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

        # Normal chat flow
        yield f"data: {json.dumps({'type': 'status', 'payload': 'Generating search queries...'})}\n\n"

        queries = []
        async for event_type, payload in _prepare_queries_stream(
            req.message,
            model_id,
            req.query_k,
            hits=req.hits,
            k=req.k,
            history=req.history,
        ):
            if event_type == "thinking":
                yield f"data: {json.dumps({'type': 'thinking', 'payload': payload})}\n\n"
            elif event_type == "result":
                queries = payload

        yield f"data: {json.dumps({'type': 'queries', 'payload': queries})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'payload': 'Retrieving context from Vespa...'})}\n\n"
        queries, chunks = await _fuse_chunks(queries, hits=req.hits, k=req.k)
        yield f"data: {json.dumps({'type': 'chunks', 'payload': chunks})}\n\n"
        if not chunks:
            yield f"data: {json.dumps({'type': 'done', 'payload': 'No relevant context found.'})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'status', 'payload': 'Generating answer...'})}\n\n"
        async for type_, payload in _openrouter_stream(chunks, req.message, model_id, req.history):
            yield f"data: {json.dumps({'type': type_, 'payload': payload})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/notes", response_class=HTMLResponse)
async def notes_ui(request: Request) -> HTMLResponse:
    """Render the notes editor UI."""
    return templates.TemplateResponse("notes.html", {"request": request})


@app.get("/api/blog/templates")
async def get_blog_templates() -> List[Dict[str, Any]]:
    """Return list of available blog templates with descriptions.

    Returns:
        List of template info dicts with 'name', 'description', and 'structure' keys.
    """
    # Load config if available
    config_path = os.getenv("NYRAG_CONFIG")
    config = None
    if config_path and Path(config_path).exists():
        try:
            config = Config.from_yaml(config_path)
        except Exception:
            pass

    templates_list = blog_list_templates(config)
    return templates_list


# ============================================================================
# Blog Generation API Endpoints (Phase 4)
# ============================================================================


class BlogGenerateRequest(BaseModel):
    """Request model for blog generation."""

    topic: str = Field(..., description="Topic or title for the blog post")
    template: Optional[str] = Field(None, description="Template name (tutorial, opinion, roundup, technical)")
    instructions: Optional[str] = Field(None, description="Additional instructions for generation")


def _get_config() -> Optional[Config]:
    """Load configuration from environment."""
    config_path = os.getenv("NYRAG_CONFIG")
    if config_path and Path(config_path).exists():
        try:
            return Config.from_yaml(config_path)
        except Exception:
            pass
    return None


@app.post("/api/blog/generate")
async def generate_blog(req: BlogGenerateRequest) -> Dict[str, Any]:
    """Generate a blog post using RAG and LLM.

    Submits a blog generation job to the background queue and returns the job ID.
    Use GET /api/jobs/{job_id} to poll for status.

    Args:
        req: Blog generation request with topic, optional template, and instructions.

    Returns:
        Dict with job_id for status polling.
    """
    config = _get_config()
    if not config:
        raise HTTPException(status_code=500, detail="NYRAG_CONFIG not set or invalid")

    job_queue = get_job_queue()

    # Submit the blog generation task
    job_id = await job_queue.submit(
        job_type="blog_generation",
        coro=generate_blog_task(
            topic=req.topic,
            config=config,
            template=req.template,
            instructions=req.instructions,
        ),
    )

    logger.info(f"Blog generation job submitted: {job_id} for topic: {req.topic}")
    return {"job_id": job_id, "topic": req.topic, "template": req.template}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get the status of a background job.

    Args:
        job_id: The job ID returned from a job submission endpoint.

    Returns:
        Job status including: id, type, status, result (if complete), error (if failed).
    """
    job_queue = get_job_queue()
    job = job_queue.get_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    response = {
        "id": job.id,
        "type": job.type,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
    }

    if job.result is not None:
        # For blog generation, include the blog_id
        if isinstance(job.result, dict):
            response["result"] = job.result
        elif hasattr(job.result, "id"):
            response["result"] = {"blog_id": job.result.id, "topic": job.result.topic}
        else:
            response["result"] = str(job.result)

    if job.error:
        response["error"] = job.error

    return response


@app.get("/api/blog/{blog_id}")
async def get_blog_content(blog_id: str) -> Dict[str, Any]:
    """Get the content of a generated blog post.

    Args:
        blog_id: The blog post ID.

    Returns:
        Blog post details including content.
    """
    config = _get_config()
    if not config:
        raise HTTPException(status_code=500, detail="NYRAG_CONFIG not set or invalid")

    blog = get_blog(blog_id, config)
    if not blog:
        raise HTTPException(status_code=404, detail=f"Blog not found: {blog_id}")

    return {
        "id": blog.id,
        "topic": blog.topic,
        "template": blog.template,
        "content": blog.content,
        "source_notes": blog.source_notes,
        "status": blog.status.value,
        "created_at": blog.created_at.isoformat(),
    }


@app.get("/api/blog/{blog_id}/download")
async def download_blog(blog_id: str) -> FileResponse:
    """Download a generated blog post as a markdown file.

    Args:
        blog_id: The blog post ID.

    Returns:
        Markdown file download.
    """
    config = _get_config()
    if not config:
        raise HTTPException(status_code=500, detail="NYRAG_CONFIG not set or invalid")

    blog_path = get_blog_path(blog_id, config)
    if not blog_path:
        raise HTTPException(status_code=404, detail=f"Blog not found: {blog_id}")

    return FileResponse(
        path=blog_path,
        filename=f"{blog_id}.md",
        media_type="text/markdown",
    )


# ============================================================================
# Notes Management API Endpoints (Phase 6)
# ============================================================================


class NoteUpdateRequest(BaseModel):
    """Request model for note updates."""

    title: str = Field(..., description="Note title")
    content: str = Field(..., description="Markdown content")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


@app.get("/api/notes")
async def get_notes_list() -> List[Dict[str, Any]]:
    """Get list of all notes.

    Returns:
        List of note summaries with id, title, created_at, updated_at, and tags.
    """
    config = _get_config()
    if not config:
        raise HTTPException(status_code=500, detail="NYRAG_CONFIG not set or invalid")

    notes = notes_list_notes(config)

    # Return summaries (not full content for list view)
    return [
        {
            "id": note.id,
            "title": note.title,
            "created_at": note.created_at.isoformat(),
            "updated_at": note.updated_at.isoformat(),
            "tags": note.tags,
            "preview": note.content[:150] + "..." if len(note.content) > 150 else note.content,
        }
        for note in notes
    ]


@app.get("/api/notes/{note_id}")
async def get_note_by_id(note_id: str) -> Dict[str, Any]:
    """Get a single note by ID.

    Args:
        note_id: The note ID.

    Returns:
        Full note content.
    """
    config = _get_config()
    if not config:
        raise HTTPException(status_code=500, detail="NYRAG_CONFIG not set or invalid")

    note = notes_get_note(note_id, config)
    if not note:
        raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")

    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "images": note.images,
        "tags": note.tags,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat(),
    }


@app.put("/api/notes/{note_id}")
async def update_note_endpoint(note_id: str, req: NoteUpdateRequest) -> Dict[str, Any]:
    """Update an existing note.

    Args:
        note_id: The note ID to update.
        req: Updated note content.

    Returns:
        Updated note.
    """
    config = _get_config()
    if not config:
        raise HTTPException(status_code=500, detail="NYRAG_CONFIG not set or invalid")

    # Get existing note to preserve some fields
    existing = notes_get_note(note_id, config)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")

    # Update the note
    existing.title = req.title
    existing.content = req.content
    existing.tags = req.tags

    try:
        updated = notes_update_note(existing, config)
        return {
            "id": updated.id,
            "title": updated.title,
            "content": updated.content,
            "tags": updated.tags,
            "created_at": updated.created_at.isoformat(),
            "updated_at": updated.updated_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/notes/{note_id}")
async def delete_note_endpoint(note_id: str) -> Dict[str, Any]:
    """Delete a note.

    Args:
        note_id: The note ID to delete.

    Returns:
        Deletion status.
    """
    config = _get_config()
    if not config:
        raise HTTPException(status_code=500, detail="NYRAG_CONFIG not set or invalid")

    success = notes_delete_note(note_id, config)
    if not success:
        raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")

    return {"deleted": True, "id": note_id}
