from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class CurriculumBuilder:
    """
    Turns scored prerequisite papers into a staged learning curriculum.

    Pipeline
    --------
    1. Stage by hop distance.  hop=1 papers are split further by publication-
       year quantile into n_stages temporal bands — this creates meaningful
       curriculum stages even when broad queries produce a flat hop structure
       (all seeds are surveys that cite everything at hop=1).
    2. Within each stage, sort chronologically then by citation topology.
    3. Prune with MMR using the scorer's combined score as relevance (not raw
       cosine sim to query), so the most important prerequisite papers are
       always selected first.
    4. Insert bridge papers between semantically distant adjacent stages.

    Year-based sub-staging rationale
    ---------------------------------
    When seed papers are surveys, almost every prerequisite lands at hop=1.
    Splitting that flat set by publication year recovers the temporal
    progression of ideas:
      Band 1 (oldest)  → foundational seq2seq / encoder-decoder papers
      Band 2           → early attention mechanisms
      Band 3           → Transformer, contextualised embeddings
      Band 4 (newest)  → post-BERT analysis and variants

    Score-based MMR relevance
    --------------------------
    Bahdanau attention ranks highest in the combined scorer but its SPECTER
    embedding is framed around NMT, not "attention transformer", so raw cosine
    similarity to the query underestimates its importance.  Using the scorer
    score fixes this.
    """

    def __init__(
        self,
        papers_df: pd.DataFrame,
        embeddings: np.ndarray,
        out_adj: Dict[int, List[int]],
        node_years: Optional[np.ndarray] = None,
        n_stages: int = 4,
        max_per_stage: int = 6,
        mmr_lambda: float = 0.7,
        bridge_threshold: float = 0.4,
        min_papers_for_year_split: int = 8,
    ):
        self.papers = papers_df
        self.embeddings = normalize(embeddings, norm="l2")
        self.out_adj = out_adj
        self.node_years = node_years
        self.n_stages = n_stages
        self.max_per_stage = max_per_stage
        self.mmr_lambda = mmr_lambda
        self.bridge_threshold = bridge_threshold
        self.min_papers_for_year_split = min_papers_for_year_split

    # ------------------------------------------------------------------
    # Step 1: Bin by hop, then split hop=1 by year
    # ------------------------------------------------------------------

    def _bin_by_hop(
        self, ranked_results: List[Dict]
    ) -> Dict[int, List[Tuple[int, float]]]:
        bins: Dict[int, List[Tuple[int, float]]] = {}
        for r in ranked_results:
            hop = r["hop"]
            if hop == 0:
                continue
            stage = min(hop, self.n_stages)
            bins.setdefault(stage, []).append((r["node_idx"], r["score"]))
        for stage in bins:
            bins[stage].sort(key=lambda x: -x[1])
        return bins

    def _split_by_year(
        self, node_score_pairs: List[Tuple[int, float]], n_splits: int
    ) -> List[List[Tuple[int, float]]]:
        """
        Partition a list of (node, score) pairs into n_splits temporal bands
        using year-quantile boundaries.  Papers from the same year always land
        in the same band (no arbitrary mid-year splits).
        """
        if self.node_years is None or len(node_score_pairs) < self.min_papers_for_year_split:
            return [node_score_pairs]

        years = np.array([self.node_years[n] for n, _ in node_score_pairs])

        # Percentile thresholds between bands (n_splits-1 interior boundaries)
        pcts = np.linspace(0, 100, n_splits + 1)[1:-1]
        thresholds = np.percentile(years, pcts)

        splits: List[List[Tuple[int, float]]] = [[] for _ in range(n_splits)]
        for ns, yr in zip(node_score_pairs, years):
            band = int(np.searchsorted(thresholds, yr, side="right"))
            splits[band].append(ns)

        return [s for s in splits if s]

    # ------------------------------------------------------------------
    # Step 2: Sort by year then by citation topology
    # ------------------------------------------------------------------

    def _sort_by_year(self, nodes: List[int]) -> List[int]:
        if self.node_years is None:
            return nodes
        return sorted(nodes, key=lambda n: self.node_years[n])

    def _topo_sort(self, nodes: List[int]) -> List[int]:
        node_set = set(nodes)
        in_deg = {n: 0 for n in nodes}
        for n in nodes:
            for neighbor in self.out_adj.get(n, []):
                if neighbor in node_set:
                    in_deg[neighbor] += 1

        def yr(n: int) -> int:
            return int(self.node_years[n]) if self.node_years is not None else 0

        queue = sorted([n for n, d in in_deg.items() if d == 0], key=yr)
        result: List[int] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            ready = []
            for nb in self.out_adj.get(node, []):
                if nb in in_deg:
                    in_deg[nb] -= 1
                    if in_deg[nb] == 0:
                        ready.append(nb)
            queue = sorted(ready, key=yr) + queue

        remaining = [n for n in nodes if n not in result]
        result.extend(sorted(remaining, key=yr))
        return result

    # ------------------------------------------------------------------
    # Step 3: MMR with scorer-score relevance
    # ------------------------------------------------------------------

    def _mmr_prune(
        self, node_score_pairs: List[Tuple[int, float]]
    ) -> List[int]:
        if len(node_score_pairs) <= 2:
            return [n for n, _ in node_score_pairs]

        nodes = [n for n, _ in node_score_pairs]
        raw = np.array([s for _, s in node_score_pairs], dtype=np.float32)
        rng = raw.max() - raw.min()
        norm_scores = (raw - raw.min()) / (rng + 1e-8) if rng > 0 else np.ones_like(raw)

        selected: List[int] = []
        remaining = list(range(len(nodes)))

        while remaining and len(selected) < self.max_per_stage:
            rem = np.array(remaining)
            relevance = norm_scores[rem]

            if not selected:
                best_local = int(np.argmax(relevance))
            else:
                cand_embs = self.embeddings[[nodes[i] for i in rem]]
                sel_embs = self.embeddings[selected]
                redundancy = (cand_embs @ sel_embs.T).max(axis=1)
                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * redundancy
                best_local = int(np.argmax(mmr))

            chosen = remaining[best_local]
            selected.append(nodes[chosen])
            remaining.pop(best_local)

        return selected

    # ------------------------------------------------------------------
    # Step 4: Bridge insertion
    # ------------------------------------------------------------------

    def _semantic_gap(self, a: List[int], b: List[int]) -> float:
        ea = self.embeddings[a].mean(axis=0)
        eb = self.embeddings[b].mean(axis=0)
        return float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))

    def _find_bridge(
        self, a: List[int], b: List[int], pool: Set[int]
    ) -> Optional[int]:
        target = (self.embeddings[a].mean(axis=0) + self.embeddings[b].mean(axis=0)) / 2
        candidates = list(pool)
        if not candidates:
            return None
        sims = self.embeddings[candidates] @ target
        return candidates[int(np.argmax(sims))]

    # ------------------------------------------------------------------
    # Main build
    # ------------------------------------------------------------------

    def _process_hop_band(
        self, node_score_pairs: List[Tuple[int, float]]
    ) -> Optional[List[int]]:
        """Sort + topo-sort + MMR-prune one year/hop band."""
        nodes = self._sort_by_year([n for n, _ in node_score_pairs])
        nodes = self._topo_sort(nodes)
        score_map = {n: s for n, s in node_score_pairs}
        pairs = [(n, score_map[n]) for n in nodes]
        pruned = self._mmr_prune(pairs)
        return pruned if pruned else None

    def build(
        self, ranked_results: List[Dict], query_embedding: np.ndarray
    ) -> List[Dict]:
        """
        Parameters
        ----------
        ranked_results  : output of PaperScorer.score() — dicts with
                          {node_idx, score, hop, ...}.
        query_embedding : L2-normalised SPECTER2 query vector (used only for
                          bridge-paper detection, not for MMR relevance).
        """
        if not ranked_results:
            return []

        hop_bins = self._bin_by_hop(ranked_results)
        stages: List[List[int]] = []

        for hop in sorted(hop_bins.keys()):
            pairs = hop_bins[hop]

            if hop == 1 and len(pairs) >= self.min_papers_for_year_split:
                # Split hop-1 papers into temporal sub-stages
                year_bands = self._split_by_year(pairs, self.n_stages)
                for band in year_bands:
                    result = self._process_hop_band(band)
                    if result:
                        stages.append(result)
            else:
                result = self._process_hop_band(pairs)
                if result:
                    stages.append(result)

        if not stages:
            return []

        # Bridge paper insertion between semantically distant adjacent stages
        all_nodes: Set[int] = {r["node_idx"] for r in ranked_results}
        final_stages: List[List[int]] = [stages[0]]
        for i in range(1, len(stages)):
            if stages[i - 1] and stages[i]:
                if self._semantic_gap(stages[i - 1], stages[i]) < self.bridge_threshold:
                    bridge = self._find_bridge(stages[i - 1], stages[i], all_nodes)
                    if bridge is not None and bridge not in all_nodes:
                        final_stages.append([bridge])
            final_stages.append(stages[i])

        results = []
        for stage_num, nodes in enumerate(final_stages, start=1):
            # Final display order: chronological within each stage
            if self.node_years is not None and len(nodes) > 1:
                nodes = sorted(nodes, key=lambda n: self.node_years[n])
            titles = []
            for n in nodes:
                title = (
                    str(self.papers.iloc[n].get("title", f"node_{n}"))
                    if n < len(self.papers)
                    else f"node_{n}"
                )
                titles.append(title)
            results.append({"stage": stage_num, "papers": nodes, "titles": titles})

        return results

    def format_output(self, stages: List[Dict]) -> str:
        return " → ".join(
            f"Stage {s['stage']}: {', '.join(s['titles'])}" for s in stages
        )
