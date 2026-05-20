import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset

from canon import QueryIntent, identify_closest_paper, parse_intent, resolve_canonical_nodes
from inference.query import HybridPaperRetriever
from path_builder import CurriculumBuilder
from scorer import PaperScorer
from subgraph import SubgraphExtractor

_SURVEY_KEYWORDS: Set[str] = {
    "survey", "review", "overview", "tutorial", "comprehensive",
    "introduction to", "a survey of", "a review of",
}


def _is_survey(node_idx: int, title: str, out_adj: Dict, deg_threshold: int) -> bool:
    if any(kw in title.lower() for kw in _SURVEY_KEYWORDS):
        return True
    return len(out_adj.get(node_idx, [])) > deg_threshold


def _replace_survey_seeds(
    seed_results: List[Dict],
    papers_df: pd.DataFrame,
    out_adj: Dict,
    specter_embs: np.ndarray,
    q_emb: np.ndarray,
    top_k: int,
) -> List[int]:
    """Replace survey-paper seeds with their most query-relevant cited neighbours."""
    seed_degs = [len(out_adj.get(r["node_idx"], [])) for r in seed_results]
    deg_threshold = max(20, int(np.median(seed_degs) * 3))

    effective: List[int] = []
    n_surveys = 0

    for r in seed_results:
        nid = r["node_idx"]
        title = str(papers_df.iloc[nid].get("title", ""))
        if _is_survey(nid, title, out_adj, deg_threshold):
            n_surveys += 1
            cited = out_adj.get(nid, [])
            if cited:
                sims = sorted(
                    ((c, float(np.dot(specter_embs[c], q_emb))) for c in cited),
                    key=lambda x: -x[1],
                )
                for c, _ in sims[:3]:
                    if c not in effective:
                        effective.append(c)
        else:
            if nid not in effective:
                effective.append(nid)

    if n_surveys > 0:
        print(f"  Survey filter: replaced {n_surveys}/{len(seed_results)} seeds "
              f"(deg_threshold={deg_threshold})")

    return effective[:top_k]


def _print_query_target(target: Dict, intent: QueryIntent) -> None:
    """Print the prominent 'closest paper' box at the top of output."""
    match_tag = "canonical match" if target["match_type"] == "canonical" else "semantic top-1"
    year_str  = f"[{target['year']}]" if target["year"] else ""
    sim_str   = f"sim={target['semantic_sim']:.3f}"

    print(f"\n  ┌{'─'*66}┐")
    print(f"  │  QUERY TARGET  ({match_tag}, {sim_str})")
    print(f"  │  {year_str:<7} {target['title'][:60]}")
    if intent.label != "general":
        print(f"  │  Topic: {intent.label}  ·  "
              f"{len(intent.canonical_nodes)} canonical node(s) found  ·  "
              f"{len(intent.topic_keywords)} topic keywords")
    print(f"  └{'─'*66}┘")


def _print_ranked_table(
    ranked: List[Dict],
    papers_df: pd.DataFrame,
    n: int = 10,
    show_scores: bool = False,
    year_cutoff: Optional[int] = None,
) -> None:
    cutoff_tag = f"  [year_cutoff={year_cutoff}]" if year_cutoff else ""
    print(f"\nTop-{n} Prerequisites:{cutoff_tag}")

    if show_scores:
        header = (f"  {'#':>2}  {'Year':>4}  {'Score':>6}  "
                  f"{'Sem':>5}  {'Top':>5}  {'Can':>5}  {'Lin':>5}  "
                  f"{'Yr':>5}  {'Typ':>5}  H  Title")
        print(header)
        print("  " + "─" * (len(header) - 2))
        for i, r in enumerate(ranked[:n], 1):
            title = str(papers_df.iloc[r["node_idx"]].get("title", ""))[:52]
            canon_mark = "★" if r["canonical"] > 0.9 else " "
            lineage_mark = "→" if r["lineage"] > 0.8 else " "
            print(
                f"  {i:>2}  {r.get('year','?'):>4}  {r['score']:>6.3f}  "
                f"{r['semantic']:>5.3f}  {r['topic_match']:>5.3f}  "
                f"{r['canonical']:>5.3f}  {r['lineage']:>5.3f}  "
                f"{r['year_score']:>5.3f}  {r['paper_type']:>5.3f}  "
                f"{r['hop']}  {canon_mark}{lineage_mark}{title}"
            )
        print("  (★ = canonical paper  · → = lineage / cited-by-canonical)")
    else:
        for i, r in enumerate(ranked[:n], 1):
            title = str(papers_df.iloc[r["node_idx"]].get("title", ""))[:72]
            year_tag = f"[{r.get('year', '?')}]"
            canon_mark = " ★" if r["canonical"] > 0.9 else "  "
            lineage_mark = "→" if r["lineage"] > 0.8 else " "
            print(f"  {i:>2}.{canon_mark}{lineage_mark} {year_tag:<7} "
                  f"{r['score']:.3f}  {title}")
        print("  (★ canonical  · → lineage precursor)")

def main(
    query: str,
    top_k_seeds: int = 10,
    year_cutoff: Optional[int] = None,
    show_scores: bool = False,
) -> None:
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  Query: {query!r}")
    print(bar)

    intent = parse_intent(query)
    print(f"\nIntent: topic={intent.label!r}  "
          f"canonical_frags={len(intent.canonical_frags)}  "
          f"topic_keywords={len(intent.topic_keywords)}")

    retriever = HybridPaperRetriever(
        metadata_path="./outputs/metadata/papers.parquet",
        specter_path="./outputs/embeddings/specter.npy",
        gnn_path="./outputs/embeddings/gnn_embeddings.pt",
    )
    initial_results = retriever.retrieve(query=query, top_k=top_k_seeds)

    print(f"\nInitial seeds ({len(initial_results)}):")
    for r in initial_results[:5]:
        print(f"  [{r['rank']}] {r['title'][:70]}  (score={r['score']:.3f})")

    q_tensor = retriever.encode_query(query)
    q_emb = q_tensor.cpu().numpy()[0]

    print("\nLoading graph ...")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="./dataset")
    data = dataset[0]
    specter_embs = np.load("./outputs/embeddings/specter.npy").astype(np.float32)
    papers = pd.read_parquet("./outputs/metadata/papers.parquet")
    node_years = data.node_year.squeeze().numpy()

    resolve_canonical_nodes(intent, papers)
    target = identify_closest_paper(intent, papers, specter_embs, q_emb, node_years)
    _print_query_target(target, intent)

    extractor = SubgraphExtractor(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        embeddings=specter_embs,
        semantic_threshold=0.25,
        hub_threshold=500,
        max_hops=4,
        max_nodes=300,
    )

    lineage_nodes: Set[int] = set()
    for cn in intent.canonical_nodes:
        lineage_nodes.update(extractor.out_adj.get(cn, []))
    if lineage_nodes:
        print(f"\nLineage: {len(lineage_nodes)} papers cited by "
              f"{len(intent.canonical_nodes)} canonical paper(s)")

    seed_nodes = _replace_survey_seeds(
        seed_results=initial_results,
        papers_df=papers,
        out_adj=extractor.out_adj,
        specter_embs=specter_embs,
        q_emb=q_emb,
        top_k=top_k_seeds,
    )

    print(f"\nEffective seeds ({len(seed_nodes)}):")
    for nid in seed_nodes[:5]:
        title = str(papers.iloc[nid].get("title", ""))
        print(f"  node {nid}: {title[:70]}")

    # ── 5. BFS prerequisite discovery ───────────────────────────────
    visited, citation_counts = extractor.extract(seed_nodes, q_emb)
    seed_set = set(seed_nodes)
    candidates = {n: h for n, h in visited.items() if n not in seed_set}
    print(f"\nSubgraph: {len(visited)} nodes visited, "
          f"{len(candidates)} candidates")

    # ── 6. Score candidates ──────────────────────────────────────────
    gnn_embs = torch.load("./outputs/embeddings/gnn_embeddings.pt").numpy()

    scorer = PaperScorer(
        specter_embeddings=specter_embs,
        gnn_embeddings=gnn_embs,
        node_years=node_years,
        papers_df=papers,
    )
    ranked = scorer.score(
        candidates,
        citation_counts,
        q_emb,
        seed_nodes,
        query_text=query,
        year_cutoff=year_cutoff,
        canonical_nodes=set(intent.canonical_nodes),
        topic_keywords=intent.topic_keywords,
        lineage_nodes=lineage_nodes,
    )
    top_ranked = ranked[:80]

    _print_ranked_table(
        top_ranked, papers,
        n=10,
        show_scores=show_scores,
        year_cutoff=year_cutoff,
    )

    # ── 7. Build and display curriculum ─────────────────────────────
    out_adj_lists: Dict[int, List[int]] = defaultdict(list)
    for s, d in zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()):
        out_adj_lists[int(s)].append(int(d))

    builder = CurriculumBuilder(
        papers_df=papers,
        embeddings=specter_embs,
        out_adj=out_adj_lists,
        node_years=node_years,
        n_stages=4,
        max_per_stage=6,
    )
    stages = builder.build(top_ranked, q_emb)

    print(f"\n{'─'*68}")
    print("CURRICULUM  (earliest → most recent)")
    print(f"{'─'*68}")
    for s in stages:
        print(f"\nStage {s['stage']}:")
        for nid, title in zip(s["papers"], s["titles"]):
            yr = int(node_years[nid]) if nid < len(node_years) else "?"
            # Mark canonical / lineage papers in the curriculum too
            mark = ""
            if nid in set(intent.canonical_nodes):
                mark = "  ★"
            elif nid in lineage_nodes:
                mark = "  →"
            print(f"  • [{yr}] {title[:75]}{mark}")

    print(f"\n{'─'*68}")
    print("CHAIN VIEW")
    print(f"{'─'*68}")
    print(builder.format_output(stages))
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prerequisite curriculum pipeline")
    ap.add_argument("query", nargs="*", default=["attention", "transformer"])
    ap.add_argument(
        "--year-cutoff", type=int, default=None, metavar="YEAR",
        help="Strongly penalise papers published at or after YEAR (e.g. --year-cutoff 2019)",
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--show-scores", action="store_true",
        help="Print per-signal score breakdown table",
    )
    args = ap.parse_args()

    main(
        query=" ".join(args.query),
        top_k_seeds=args.top_k,
        year_cutoff=args.year_cutoff,
        show_scores=args.show_scores,
    )
