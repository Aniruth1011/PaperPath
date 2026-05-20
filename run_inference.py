from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset

from canon import identify_closest_paper, parse_intent, resolve_canonical_nodes
from inference.query import HybridPaperRetriever
from path_builder import CurriculumBuilder
from scorer import PaperScorer
from subgraph import SubgraphExtractor
from pipeline import _replace_survey_seeds, _print_query_target, _print_ranked_table

DEFAULT_QUERIES = [
    "gan",
    "attention mechanism transformer",
    "variational autoencoder latent space",
    "graph neural network node classification",
    "recurrent neural network language model",
]

def load_resources(specter_path: str, metadata_path: str, gnn_path: str):
    print("Loading shared resources ...")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="./dataset")
    data = dataset[0]
    specter_embs = np.load(specter_path).astype(np.float32)
    papers = pd.read_parquet(metadata_path)
    node_years = data.node_year.squeeze().numpy()

    import os
    gnn_embs = None
    if os.path.exists(gnn_path):
        gnn_embs = torch.load(gnn_path, map_location="cpu").numpy()

    extractor = SubgraphExtractor(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        embeddings=specter_embs, semantic_threshold=0.25,
        hub_threshold=500, max_hops=4,
        max_nodes=300,)

    print(f" {data.num_nodes:,} nodes  |  {data.edge_index.shape[1]:,} edges  " f"|  year range {int(node_years.min())}–{int(node_years.max())}")
    return data, specter_embs, papers, node_years, gnn_embs, extractor

def run_one(
    query: str,
    retriever: HybridPaperRetriever,
    data,
    specter_embs: np.ndarray,
    papers: pd.DataFrame,
    node_years: np.ndarray,
    gnn_embs: Optional[np.ndarray],
    extractor: SubgraphExtractor,
    top_k_seeds: int = 10,
    year_cutoff: Optional[int] = None,
    show_scores: bool = False,
    n_results: int = 10,
) -> List[Dict]:
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  QUERY: {query!r}")
    if year_cutoff:
        print(f"  year_cutoff={year_cutoff}")
    print(bar)

    # Intent & canonical resolution
    intent = parse_intent(query)
    resolve_canonical_nodes(intent, papers)
    print(f"  Intent: {intent.label!r}  |  "
          f"{len(intent.canonical_nodes)} canonical node(s)  |  "
          f"{len(intent.topic_keywords)} topic keywords")

    # Query encoding
    initial_results = retriever.retrieve(query=query, top_k=top_k_seeds)
    q_emb = retriever.encode_query(query).cpu().numpy()[0]

    # Query target
    target = identify_closest_paper(intent, papers, specter_embs, q_emb, node_years)
    _print_query_target(target, intent)

    # Lineage
    lineage_nodes: Set[int] = set()
    for cn in intent.canonical_nodes:
        lineage_nodes.update(extractor.out_adj.get(cn, []))

    # Survey filter + BFS
    seed_nodes = _replace_survey_seeds(
        seed_results=initial_results,
        papers_df=papers,
        out_adj=extractor.out_adj,
        specter_embs=specter_embs,
        q_emb=q_emb,
        top_k=top_k_seeds,
    )
    visited, citation_counts = extractor.extract(seed_nodes, q_emb)
    candidates = {n: h for n, h in visited.items() if n not in set(seed_nodes)}
    print(f"\n  BFS: {len(visited)} visited  |  {len(candidates)} candidates  "
          f"|  {len(lineage_nodes)} lineage nodes")

    # Score
    scorer = PaperScorer(
        specter_embeddings=specter_embs,
        gnn_embeddings=gnn_embs,
        node_years=node_years,
        papers_df=papers,
    )
    ranked = scorer.score(
        candidates, citation_counts, q_emb, seed_nodes,
        query_text=query,
        year_cutoff=year_cutoff,
        canonical_nodes=set(intent.canonical_nodes),
        topic_keywords=intent.topic_keywords,
        lineage_nodes=lineage_nodes,
    )

    _print_ranked_table(
        ranked, papers,
        n=n_results,
        show_scores=show_scores,
        year_cutoff=year_cutoff,
    )

    # Curriculum
    out_adj_lists: Dict[int, List[int]] = defaultdict(list)
    for s, d in zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()):
        out_adj_lists[int(s)].append(int(d))

    builder = CurriculumBuilder(
        papers_df=papers, embeddings=specter_embs,
        out_adj=out_adj_lists, node_years=node_years,
        n_stages=4, max_per_stage=6,
    )
    stages = builder.build(ranked[:80], q_emb)

    canon_set = set(intent.canonical_nodes)
    print(f"\n  CURRICULUM")
    print(f"  {'─'*66}")
    for s in stages:
        print(f"\n  Stage {s['stage']}:")
        for nid, title in zip(s["papers"], s["titles"]):
            yr = int(node_years[nid]) if nid < len(node_years) else "?"
            mark = " ★" if nid in canon_set else (" →" if nid in lineage_nodes else "  ")
            print(f"    •{mark} [{yr}] {title[:72]}")

    return ranked


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-query inference demo")
    ap.add_argument("--queries", nargs="+", default=None,
                    help="Override default query list (quote each query)")
    ap.add_argument("--year-cutoff", type=int, default=None, metavar="YEAR",
                    help="Penalise papers published at or after YEAR (e.g. 2019)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--n-results", type=int, default=10)
    ap.add_argument("--show-scores", action="store_true",
                    help="Print per-signal breakdown (Sem/Top/Can/Lin/Yr/Typ)")
    ap.add_argument("--specter-path",  default="./outputs/embeddings/specter.npy")
    ap.add_argument("--gnn-path",      default="./outputs/embeddings/gnn_embeddings.pt")
    ap.add_argument("--metadata-path", default="./outputs/metadata/papers.parquet")
    args = ap.parse_args()

    queries = args.queries or DEFAULT_QUERIES

    data, specter_embs, papers, node_years, gnn_embs, extractor = load_resources(
        specter_path=args.specter_path,
        metadata_path=args.metadata_path,
        gnn_path=args.gnn_path,
    )
    retriever = HybridPaperRetriever(
        metadata_path=args.metadata_path,
        specter_path=args.specter_path,
        gnn_path=args.gnn_path,
        semantic_weight=0.80,
        graph_weight=0.20,
    )

    print(f"\nRunning {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}")
    if args.year_cutoff:
        print(f"Pre-{args.year_cutoff} mode active — papers from {args.year_cutoff}+ penalised")

    for q in queries:
        run_one(
            query=q,
            retriever=retriever,
            data=data,
            specter_embs=specter_embs,
            papers=papers,
            node_years=node_years,
            gnn_embs=gnn_embs,
            extractor=extractor,
            top_k_seeds=args.top_k,
            year_cutoff=args.year_cutoff,
            show_scores=args.show_scores,
            n_results=args.n_results,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
