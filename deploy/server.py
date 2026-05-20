"""
FastAPI server for PreReqGraph.

Start:  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
Or:     python server.py

Endpoints
---------
GET  /                  → SPA (index.html)
POST /api/query         → run pipeline, returns full result dict
GET  /api/metrics       → aggregated metrics + recent history
GET  /api/health        → server / resource loading status
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from metrics import MetricsStore, QueryRecord
from pipeline_runner import PrereqPipeline


# ── Config from environment ───────────────────────────────────────────────────

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)

_FINAL_DIR      = _env("FINAL_DIR",      "../final")
_SPECTER_PATH   = _env("SPECTER_PATH",   os.path.join(_FINAL_DIR, "outputs/embeddings/specter.npy"))
_GNN_PATH       = _env("GNN_PATH",       os.path.join(_FINAL_DIR, "outputs/embeddings/gnn_embeddings.pt"))
_METADATA_PATH  = _env("METADATA_PATH",  os.path.join(_FINAL_DIR, "outputs/metadata/papers.parquet"))
_DATASET_ROOT   = _env("DATASET_ROOT",   os.path.join(_FINAL_DIR, "dataset"))
_HOST           = _env("HOST",           "0.0.0.0")
_PORT           = int(_env("PORT",       "8000"))
_CACHE_TTL      = int(_env("CACHE_TTL",  "300"))
_CACHE_MAX      = int(_env("CACHE_MAX_SIZE", "50"))
_METRICS_HIST   = int(_env("METRICS_HISTORY", "500"))

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# ── Global singletons ─────────────────────────────────────────────────────────

pipeline:  Optional[PrereqPipeline] = None
mstore:    MetricsStore             = MetricsStore(max_history=_METRICS_HIST)
_startup_error: Optional[str]      = None
_startup_time:  float               = time.time()

# Simple TTL query cache: key → (result_dict, expiry_timestamp)
_cache: Dict[str, tuple] = {}


def _cache_key(query: str, year_cutoff: Optional[int], top_k: int) -> str:
    return f"{query.lower().strip()}|{year_cutoff}|{top_k}"


def _cache_get(key: str) -> Optional[Dict]:
    entry = _cache.get(key)
    if entry and (_CACHE_TTL == 0 or time.time() < entry[1]):
        return entry[0]
    _cache.pop(key, None)
    return None


def _cache_put(key: str, value: Dict) -> None:
    if _CACHE_TTL == 0:
        return
    # Evict oldest if over capacity
    if len(_cache) >= _CACHE_MAX:
        oldest = min(_cache.items(), key=lambda x: x[1][1])
        _cache.pop(oldest[0], None)
    _cache[key] = (value, time.time() + _CACHE_TTL)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, _startup_error
    loop = asyncio.get_event_loop()
    try:
        def _load():
            return PrereqPipeline(
                specter_path=_SPECTER_PATH,
                gnn_path=_GNN_PATH,
                metadata_path=_METADATA_PATH,
                dataset_root=_DATASET_ROOT,
            )
        pipeline = await loop.run_in_executor(None, _load)
    except Exception as exc:
        _startup_error = str(exc)
        print(f"[server] Pipeline load FAILED: {exc}")
    yield
    # Shutdown — nothing to clean up


app = FastAPI(
    title="PreReqGraph",
    description="Query-aware prerequisite curriculum for ML papers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str      = Field(..., min_length=1, max_length=300)
    year_cutoff: Optional[int] = Field(None, ge=1990, le=2030)
    top_k: int      = Field(10, ge=1, le=30)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    index = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.exists(index):
        return JSONResponse({"error": "static/index.html not found"}, status_code=404)
    return FileResponse(index, media_type="text/html")


@app.get("/api/health")
async def health() -> Dict:
    global pipeline, _startup_error, _startup_time
    if _startup_error:
        return {"status": "error", "message": _startup_error,
                "uptime_s": round(time.time() - _startup_time)}
    if pipeline is None or not pipeline.is_ready:
        msg = pipeline.status_message if pipeline else "Starting..."
        return {"status": "loading", "message": msg,
                "uptime_s": round(time.time() - _startup_time)}
    return {
        "status":   "ready",
        "message":  "Pipeline ready",
        "uptime_s": round(time.time() - _startup_time),
        "dataset":  pipeline.info,
        "cache": {
            "entries": len(_cache),
            "ttl_s":   _CACHE_TTL,
        },
    }


@app.post("/api/query")
async def query_endpoint(req: QueryRequest) -> Dict[str, Any]:
    if pipeline is None or not pipeline.is_ready:
        msg = _startup_error or (pipeline.status_message if pipeline else "Loading...")
        raise HTTPException(503, detail=f"Pipeline not ready: {msg}")

    key = _cache_key(req.query, req.year_cutoff, req.top_k)
    cached = _cache_get(key)
    if cached:
        cached["_cached"] = True
        return cached

    loop = asyncio.get_event_loop()
    error: Optional[str] = None
    result: Optional[Dict] = None

    try:
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.run(req.query, req.year_cutoff, req.top_k),
        )
        result["_cached"] = False
    except Exception as exc:
        error = str(exc)
        result = {
            "status": "error",
            "query":  req.query,
            "error":  error,
            "_cached": False,
        }

    # Record metrics
    timing = result.get("timing", {})
    bfs    = result.get("bfs_stats", {})
    intent = result.get("intent", {})
    mstore.record(QueryRecord(
        query=req.query,
        topic_label=intent.get("label", "general"),
        timestamp=time.time(),
        total_ms=timing.get("total_ms", 0),
        retrieval_ms=timing.get("retrieval_ms", 0),
        canon_ms=timing.get("canon_ms", 0),
        bfs_ms=timing.get("bfs_ms", 0),
        scoring_ms=timing.get("scoring_ms", 0),
        curriculum_ms=timing.get("curriculum_ms", 0),
        nodes_visited=bfs.get("nodes_visited", 0),
        candidates=bfs.get("candidates", 0),
        lineage_nodes=bfs.get("lineage_nodes", 0),
        canonical_matched=(result.get("query_target") or {}).get("match_type") == "canonical",
        prerequisites_returned=len(result.get("prerequisites", [])),
        curriculum_stages=len(result.get("curriculum", [])),
        error=error,
    ))

    if result["status"] == "ok":
        _cache_put(key, result)

    return result


@app.get("/api/metrics")
async def metrics_endpoint() -> Dict:
    return mstore.summary()


@app.post("/api/cache/clear")
async def clear_cache() -> Dict:
    _cache.clear()
    return {"cleared": True}


# ── Serve static files ────────────────────────────────────────────────────────
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host=_HOST, port=_PORT, reload=False, workers=1)
