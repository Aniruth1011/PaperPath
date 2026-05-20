"""
Stateful pipeline wrapper for the FastAPI server.

Loads all heavy resources once at startup (dataset, embeddings, SPECTER2 model)
and exposes a single `.run()` method that takes a query string and returns a
fully structured JSON-serialisable result dict.

Import path strategy
--------------------
This file lives at  PreReqGraph/deploy/pipeline_runner.py.
It adds  PreReqGraph/final/  to sys.path so it can import canon, scorer, etc.
without modifying anything under final/.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch

# ── Resolve final/ on sys.path ────────────────────────────────────────────────
_FINAL_DIR = os.path.abspath(
    os.environ.get("FINAL_DIR", os.path.join(os.path.dirname(__file__), ".."))
)
if _FINAL_DIR not in sys.path:
    sys.path.insert(0, _FINAL_DIR)

from canon import QueryIntent, identify_closest_paper, parse_intent, resolve_canonical_nodes
from inference.query import HybridPaperRetriever
from path_builder import CurriculumBuilder
from pipeline import _replace_survey_seeds
from scorer import PaperScorer
from subgraph import SubgraphExtractor


class PrereqPipeline:
    """
    Singleton loaded once per server process.
    Thread-safe for concurrent reads (all mutations are on per-call locals).
    """

    def __init__(
        self,
        specter_path: str,
        gnn_path: str,
        metadata_path: str,
        dataset_root: str,
        semantic_weight: float = 0.80,
        graph_weight: float = 0.20,
    ):
        self.is_ready = False
        self.status_message = "Initialising..."
        self._specter_path = specter_path
        self._gnn_path = gnn_path
        self._metadata_path = metadata_path

        # ── Load resources ────────────────────────────────────────────────────
        self.status_message = "Loading ogbn-arxiv graph..."
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=dataset_root)
        self._data = dataset[0]

        self.status_message = "Loading SPECTER2 embeddings..."
        self._specter_embs = np.load(specter_path).astype(np.float32)

        self.status_message = "Loading paper metadata..."
        self._papers = pd.read_parquet(metadata_path)
        self._node_years = self._data.node_year.squeeze().numpy()

        self.status_message = "Loading GNN embeddings..."
        self._gnn_embs: Optional[np.ndarray] = None
        if os.path.exists(gnn_path):
            self._gnn_embs = torch.load(gnn_path, map_location="cpu").numpy()

        self.status_message = "Building citation graph index..."
        self._extractor = SubgraphExtractor(
            edge_index=self._data.edge_index,
            num_nodes=self._data.num_nodes,
            embeddings=self._specter_embs,
            semantic_threshold=0.25,
            hub_threshold=500,
            max_hops=4,
            max_nodes=300,
        )

        # Pre-build out_adj_lists for CurriculumBuilder (avoids rebuilding per query)
        self.status_message = "Building adjacency lists..."
        self._out_adj_lists: Dict[int, List[int]] = defaultdict(list)
        src_arr = self._data.edge_index[0].numpy()
        dst_arr = self._data.edge_index[1].numpy()
        for s, d in zip(src_arr, dst_arr):
            self._out_adj_lists[int(s)].append(int(d))

        self.status_message = "Loading SPECTER2 language model..."
        self._retriever = HybridPaperRetriever(
            metadata_path=metadata_path,
            specter_path=specter_path,
            gnn_path=gnn_path if os.path.exists(gnn_path) else None,
            semantic_weight=semantic_weight,
            graph_weight=graph_weight,
        )

        self.status_message = "Ready"
        self.is_ready = True

        # Dataset info exposed for /api/health
        self.info = {
            "num_papers":    int(self._data.num_nodes),
            "num_edges":     int(self._data.edge_index.shape[1]),
            "year_min":      int(self._node_years.min()),
            "year_max":      int(self._node_years.max()),
            "gnn_available": self._gnn_embs is not None,
        }

    # ── Query entrypoint ──────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        year_cutoff: Optional[int] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Run the full pipeline and return a JSON-serialisable result dict."""

        t_total = time.perf_counter()

        # ── 1. Query encoding + seed retrieval ───────────────────────────────
        t0 = time.perf_counter()
        initial_results = self._retriever.retrieve(query=query, top_k=top_k)
        q_emb = self._retriever.encode_query(query).cpu().numpy()[0]
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # ── 2. Intent + canonical resolution ────────────────────────────────
        t0 = time.perf_counter()
        intent = parse_intent(query)
        resolve_canonical_nodes(intent, self._papers)
        target = identify_closest_paper(
            intent, self._papers, self._specter_embs, q_emb, self._node_years
        )
        canon_ms = (time.perf_counter() - t0) * 1000

        # ── 3. Survey filter + BFS ───────────────────────────────────────────
        t0 = time.perf_counter()
        seed_nodes = _replace_survey_seeds(
            seed_results=initial_results,
            papers_df=self._papers,
            out_adj=self._extractor.out_adj,
            specter_embs=self._specter_embs,
            q_emb=q_emb,
            top_k=top_k,
        )
        lineage_nodes: Set[int] = set()
        for cn in intent.canonical_nodes:
            lineage_nodes.update(self._extractor.out_adj.get(cn, []))

        visited, citation_counts = self._extractor.extract(seed_nodes, q_emb)
        seed_set = set(seed_nodes)
        candidates = {n: h for n, h in visited.items() if n not in seed_set}
        bfs_ms = (time.perf_counter() - t0) * 1000

        # ── 4. Scoring ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        scorer = PaperScorer(
            specter_embeddings=self._specter_embs,
            gnn_embeddings=self._gnn_embs,
            node_years=self._node_years,
            papers_df=self._papers,
        )
        ranked = scorer.score(
            candidates, citation_counts, q_emb, seed_nodes,
            query_text=query,
            year_cutoff=year_cutoff,
            canonical_nodes=set(intent.canonical_nodes),
            topic_keywords=intent.topic_keywords,
            lineage_nodes=lineage_nodes,
        )
        scoring_ms = (time.perf_counter() - t0) * 1000

        # ── 5. Curriculum ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        builder = CurriculumBuilder(
            papers_df=self._papers,
            embeddings=self._specter_embs,
            out_adj=self._out_adj_lists,
            node_years=self._node_years,
            n_stages=4,
            max_per_stage=6,
        )
        stages = builder.build(ranked[:80], q_emb)
        curriculum_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        # ── Serialise ────────────────────────────────────────────────────────
        canon_set = set(intent.canonical_nodes)

        prerequisites = []
        for rank, r in enumerate(ranked[:top_k * 2], 1):   # return 2× for UI pagination
            nid = r["node_idx"]
            row = self._papers.iloc[nid]
            prerequisites.append({
                "rank":         rank,
                "node_idx":     nid,
                "title":        str(row.get("title", f"node_{nid}")),
                "abstract":     str(row.get("abstract", ""))[:220],
                "year":         r.get("year"),
                "score":        round(r["score"], 4),
                "signals": {
                    "semantic":      round(r["semantic"], 3),
                    "topic_match":   round(r["topic_match"], 3),
                    "graph":         round(r["graph"], 3),
                    "citation_freq": round(r["citation_freq"], 3),
                    "hop_score":     round(1.0 / (1.0 + r["hop"]), 3),
                    "year_score":    round(r["year_score"], 3),
                    "canonical":     round(r["canonical"], 3),
                    "lineage":       round(r["lineage"], 3),
                    "paper_type":    round(r["paper_type"], 3),
                },
                "hop":           r["hop"],
                "is_canonical":  nid in canon_set,
                "is_lineage":    nid in lineage_nodes,
            })

        curriculum_out = []
        for s in stages:
            papers_in_stage = []
            for nid, title in zip(s["papers"], s["titles"]):
                yr = int(self._node_years[nid]) if nid < len(self._node_years) else None
                papers_in_stage.append({
                    "node_idx":    nid,
                    "title":       title,
                    "year":        yr,
                    "is_canonical": nid in canon_set,
                    "is_lineage":  nid in lineage_nodes,
                })
            curriculum_out.append({"stage": s["stage"], "papers": papers_in_stage})

        return {
            "status": "ok",
            "query":  query,
            "intent": {
                "label":                intent.label,
                "canonical_frags":      intent.canonical_frags,
                "topic_keywords_count": len(intent.topic_keywords),
            },
            "query_target": target,
            "prerequisites":  prerequisites,
            "curriculum":     curriculum_out,
            "timing": {
                "total_ms":      round(total_ms),
                "retrieval_ms":  round(retrieval_ms),
                "canon_ms":      round(canon_ms),
                "bfs_ms":        round(bfs_ms),
                "scoring_ms":    round(scoring_ms),
                "curriculum_ms": round(curriculum_ms),
            },
            "bfs_stats": {
                "nodes_visited":              len(visited),
                "candidates":                 len(candidates),
                "lineage_nodes":              len(lineage_nodes),
                "canonical_nodes_found":      len(intent.canonical_nodes),
                "canonical_in_candidates":    sum(1 for n in intent.canonical_nodes if n in candidates),
            },
        }
