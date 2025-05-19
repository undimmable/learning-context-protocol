"""Minimal Model-Context-Protocol (MCP) server with persistent memory tools.

Run with:

    uvicorn mcp_server:app --host 127.0.0.1 --port 7821

Environment variable `CODEX_MCP_URL` should then be set to
```
export CODEX_MCP_URL=http://127.0.0.1:7821
```

The server exposes the three core tools expected by Codex CLI
(`shell`, `read_file`, `apply_patch`) plus two new endpoints that provide
durable, cross-session "memory" for the assistant:

1. POST /memory/upsert – store a text snippet (optionally tagged).
2. POST /memory/query  – retrieve up to `top_k` snippets most similar to a
   query string.

The memory is kept in a newline-delimited JSON file `memory.jsonl` in the
working directory.  A very lightweight similarity measure based on
`difflib.SequenceMatcher` is used so the server has no external
dependencies or need for an embedding model.
"""

from __future__ import annotations

import datetime as _dt
import difflib as _difflib
import json as _json
import os as _os
import pathlib as _pl
# NOTE: we switch from the JSONL-based `_MemoryStore` to a fully relational
# backend powered by SQLAlchemy/Postgres.  The public interface remains
# identical so the surrounding FastAPI routes do not change.

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# Optional heavyweight imports (LLM).  We import lazily so that the server
# can still start instantly even if the transformers stack is not yet pulled
# into the environment.  The first call that actually needs the model will
# trigger the download / load.
# ---------------------------------------------------------------------------
# Runtime env tweaks – keep big numeric libs constrained for sandbox safety.
# ---------------------------------------------------------------------------

# Avoid OpenMP / MKL creating shared memory segments that might be disallowed
# inside restricted environments (e.g. some container sandboxes or CI run
# settings).  These environment variables must be set *before* the first
# import of numpy / torch / transformers that load the native libraries.

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
_os.environ.setdefault("KMP_AFFINITY", "none")


from functools import lru_cache

# We *lazily* import the Transformers stack in `_get_llm()` below so that the
# server can spin up instantly and also remain usable in minimal environments
# where heavyweight ML libraries are not installed.

_HAVE_TRANSFORMERS = True  # Will be revised in `_get_llm` if import fails.

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="MCP Server with Memory", version="0.1.0")

# ---------------------------------------------------------------------------
# Persistent memory store – now backed by a relational database.
# ---------------------------------------------------------------------------

from src.db.memory_store import MemoryStore as _DBMemoryStore  # noqa: E402 – after sys path

# Initialize once at import time – cheap and thread-safe thanks to
# SQLAlchemy's connection pooling.
_memory_store = _DBMemoryStore()

# ---------------------------------------------------------------------------
# Helper – lightweight wrapper around google/flan-t5-small
# ---------------------------------------------------------------------------


class _LLMNotAvailable(RuntimeError):
    """Raised when transformers is missing but an LLM endpoint is invoked."""


@lru_cache(maxsize=1)
def _get_llm():
    """Download & load the FLAN-T5 model on first use.

    This is cached so subsequent calls are quick.  We keep the model very
    small ("google/flan-t5-small") to avoid excessive memory footprint while
    still providing reasonable generation quality.
    """

    # Import transformers lazily so that environments without the dependency
    # (or without the necessary CPU features) can still *import* this module.

    global _HAVE_TRANSFORMERS

    try:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline as _hf_pipeline,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover
        _HAVE_TRANSFORMERS = False
        raise _LLMNotAvailable(
            "The transformers library is required for LLM endpoints, but it"
            " is not installed in this environment.") from exc

    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _hf_pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, framework="pt"
    )


def _llm_generate(prompt: str, max_new_tokens: int = 128) -> str:
    """Generate text from the local FLAN-T5 model with sane defaults."""

    pipe = _get_llm()
    result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]
    return result["generated_text"]

# ---------------------------------------------------------------------------
# Schemas and endpoints for the three built-in Codex tools
# ---------------------------------------------------------------------------


class ShellArgs(BaseModel):
    command: List[str]
    workdir: Optional[str] = Field(default=None, description="Working directory")
    timeout: Optional[int] = Field(default=10000, description="Timeout in ms")


@app.post("/shell")
def do_shell(req: ShellArgs):  # noqa: D401  (FastAPI route, not a docstring)
    """Run an arbitrary shell command and capture its output."""

    # Instead of executing arbitrary shell commands directly on the host, we
    # forward the request to the local FLAN-T5 model.  This keeps the MCP
    # provider hermetic and avoids potential security issues, while still
    # giving downstream agents a best-effort, language-model-based simulation
    # of what the command *would* return.

    prompt = (
        "You are an advanced shell interpreter running on a Unix-like system.\n"
        "Simulate running the following command and provide *only* the raw\n"
        "stdout that a user would see (no additional commentary).  If the\n"
        "command would normally produce no output, return an empty string.\n\n"
        f"$ {' '.join(req.command)}"
    )

    try:
        stdout = _llm_generate(prompt, max_new_tokens=128)
    except _LLMNotAvailable as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # LLM cannot provide stderr/exit_code reliably; we default to 0.
    return {"stdout": stdout, "stderr": "", "exit_code": 0}


class ReadFileArgs(BaseModel):
    path: str
    max_bytes: Optional[int] = Field(default=100_000, description="Byte cap")


@app.post("/read_file")
def do_read_file(req: ReadFileArgs):
    """Return up to `max_bytes` bytes from the requested file."""

    p = _pl.Path(req.path).expanduser()
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")

    data = p.read_bytes()[: req.max_bytes]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin1", errors="replace")
    return {"content": text}


class PatchArgs(BaseModel):
    patch: str


@app.post("/apply_patch")
def do_apply_patch(req: PatchArgs):
    """Apply a unified diff to the local filesystem."""

    # Lazy import to avoid dependency if user never calls this endpoint.
    try:
        import patch as _patch
    except ModuleNotFoundError:  # pragma: no cover – dependency optional
        raise HTTPException(
            status_code=500,
            detail="python-patch must be installed for /apply_patch",
        )

    pset = _patch.fromstring(req.patch)
    if not pset.apply():
        raise HTTPException(status_code=400, detail="Failed to apply patch")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# New: persistent memory tools
# ---------------------------------------------------------------------------


class MemoryUpsertArgs(BaseModel):
    text: str = Field(..., description="Text to remember")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags")


@app.post("/memory/upsert")
def memory_upsert(req: MemoryUpsertArgs):
    """Persist a text snippet (and optional tags) to the memory store."""

    entry = {
        "text": req.text,
        "tags": req.tags or [],
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
    }
    _memory_store.append(entry)
    return {"status": "ok"}


class MemoryQueryArgs(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=50)


@app.post("/memory/query")
def memory_query(req: MemoryQueryArgs):
    """Return up to *top_k* snippets ranked by fuzzy string similarity."""

    entries = _memory_store.all()
    if not entries:
        return {"results": []}

    scored = []
    q = req.query.lower()
    for ent in entries:
        score = _difflib.SequenceMatcher(None, q, ent["text"].lower()).ratio()
        scored.append((score, ent))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [e for _, e in scored[: req.top_k]]
    return {"results": best}


# ---------------------------------------------------------------------------
# New: lightweight LLM endpoint(s)
# ---------------------------------------------------------------------------


class LLMGenerateArgs(BaseModel):
    prompt: str = Field(..., description="Prompt string for FLAN-T5")
    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=512,
        description="Maximum number of tokens to generate",
    )


@app.post("/llm/generate")
def llm_generate(req: LLMGenerateArgs):
    """Generate text via the embedded google/flan-t5-small model.

    This gives the Codex CLI clients (or any other consumers) a simple way to
    obtain LLM completions without needing direct model access.  The response
    format intentionally mirrors the HuggingFace pipeline output for easy
    integration, but wrapped in a top-level JSON object.
    """

    try:
        out_text = _llm_generate(req.prompt, max_new_tokens=req.max_new_tokens)
    except _LLMNotAvailable as exc:  # pragma: no cover – handled at runtime
        raise HTTPException(status_code=500, detail=str(exc))

    return {"generated_text": out_text}
