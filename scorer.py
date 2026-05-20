from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

_FOUNDATION_KEYWORDS: Set[str] = {
    "transformer", "attention mechanism", "self-attention", "multi-head attention",
    "neural attention", "attention model", "attention",
    "bert", "gpt", "vit", "vision transformer",
    "resnet", "residual network", "residual connection", "skip connection",
    "generative adversarial", "diffusion model", "score matching",
    "variational autoencoder", "vae",
    "word2vec", "glove", "fasttext",
    "seq2seq", "sequence to sequence", "encoder-decoder", "encoder decoder",
    "layer normalization", "batch normalization", "dropout",
    "relu", "activation function",
    "language model", "pretrain", "pre-train", "fine-tun",
    "transfer learning", "contrastive learning", "self-supervised",
    "representation learning",
    "convolutional neural", "recurrent neural", "lstm", "gru",
    "backpropagation", "gradient descent",
    "graph convolution", "message passing", "graph attention",
    "normaliz", "embedding",
}

_SYSTEM_KEYWORDS: Set[str] = {
    "distributed training", "data parallel", "model parallel",
    "hardware accelerat", "tpu", "gpu cluster",
    "serving", "deployment", "mlops", "inference optimization",
    "quantiz", "model pruning", "model compression", "weight pruning",
    "throughput", "memory efficient", "mixed precision",
    "federated learning", "data pipeline", "high performance computing",
    "latency reduction", "scalable training",
}

class PaperScorer:
    # Default weights — total must be 1.0
    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "semantic":      0.40,   # dominant but tempered by topic constraints
        "topic_match":   0.10,   # query-topic alignment: keeps results on-topic
        "graph":         0.05,   # structural proximity — kept small
        "citation_freq": 0.02,   # subgraph citation frequency — minimal
        "hop":           0.03,   # BFS distance — minimal
        "year":          0.12,   # temporal foundational preference
        "canonical":     0.15,   # landmark paper prior — strongest non-semantic signal
        "lineage":       0.05,   # cited-by-canonical precursor bonus
        "paper_type":    0.08,   # static architecture vs. systems filter
    }

    def __init__(
        self,
        specter_embeddings: np.ndarray,
        gnn_embeddings: Optional[np.ndarray] = None,
        node_years: Optional[np.ndarray] = None,
        papers_df: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.specter = specter_embeddings
        self.gnn = gnn_embeddings
        self.node_years = node_years
        self.papers_df = papers_df
        self.weights = weights or dict(self._DEFAULT_WEIGHTS)

    def _l2(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / (norm + 1e-8)

    def _paper_text(self, node: int) -> str:
        """Padded lowercase title+abstract for substring matching."""
        if self.papers_df is None or node >= len(self.papers_df):
            return ""
        row = self.papers_df.iloc[node]
        title = str(row.get("title", "")).lower()
        abstract = str(row.get("abstract", "")).lower()[:500]
        return " " + title + " " + abstract + " "


    def _semantic_sim(self, node: int, query_emb: np.ndarray) -> float:
        return float(np.dot(self._l2(self.specter[node]), self._l2(query_emb)))

    def _graph_sim(self, node: int, seed_nodes: List[int]) -> float:
        if not seed_nodes:
            return 0.5
        src = self.gnn if self.gnn is not None else self.specter
        emb = self._l2(src[node])
        seed_embs = self._l2(src[np.array(seed_nodes, dtype=np.int64)])
        return float((seed_embs @ emb).mean())

    def _hop_score(self, hop: int) -> float:
        return 1.0 / (1.0 + hop)

    def _year_score(self, node: int, max_year: int) -> float:
        if self.node_years is None:
            return 0.5
        year = int(self.node_years[node])
        gap = float(max(0, max_year - year))
        if gap <= 1:
            return 0.15
        elif gap <= 4:
            return 0.15 + (gap - 1) / 3.0 * 0.40
        elif gap <= 20:
            return 0.55 + (gap - 4) / 16.0 * 0.45
        else:
            return max(0.50, 1.0 - (gap - 20) * 0.025)

    def _paper_type_score(self, node: int, query_text_lower: str) -> float:
        """Foundation/architecture papers → 1.0; systems papers → 0.0 (unless queried)."""
        text = self._paper_text(node)
        if not text:
            return 0.5
        n_f = sum(1 for kw in _FOUNDATION_KEYWORDS if kw in text)
        n_s = sum(1 for kw in _SYSTEM_KEYWORDS if kw in text)
        query_is_sys = any(kw in query_text_lower for kw in _SYSTEM_KEYWORDS)
        if n_f > 0:
            return min(1.0, 0.5 + n_f * 0.12)
        if n_s > 0 and not query_is_sys:
            return max(0.0, 0.5 - n_s * 0.15)
        return 0.5

    def _canonical_score(self, node: int, canonical_nodes: Optional[Set[int]]) -> float:
        if not canonical_nodes:   # None or empty — no information
            return 0.5
        return 1.0 if node in canonical_nodes else 0.3

    def _topic_match_score(self, node: int, topic_keywords: Optional[Set[str]]) -> float:
        if not topic_keywords:
            return 0.5
        text = self._paper_text(node)
        if not text:
            return 0.5
        hits = sum(1 for kw in topic_keywords if kw in text)
        if hits == 0:
            return 0.25
        return min(1.0, 0.25 + hits * 0.25)

    def _lineage_score(self, node: int, lineage_nodes: Optional[Set[int]]) -> float:
        if not lineage_nodes:
            return 0.5
        return 0.9 if node in lineage_nodes else 0.4


    def score(self,candidates: Dict[int, int],    citation_counts: Dict[int, int],   
        query_embedding: np.ndarray, seed_nodes: List[int], query_text: str = "",
        year_cutoff: Optional[int] = None, canonical_nodes: Optional[Set[int]] = None,
        topic_keywords: Optional[Set[str]] = None,lineage_nodes: Optional[Set[int]] = None,) :
        if not candidates:
            return []

        max_cite = max(citation_counts.values()) if citation_counts else 1
        max_year = int(self.node_years.max()) if self.node_years is not None else 2023
        query_lower = query_text.lower()

        canon_set: Optional[Set[int]] = (canonical_nodes if isinstance(canonical_nodes, set) else set(canonical_nodes) if canonical_nodes else None )
        lineage_set: Optional[Set[int]] = ( lineage_nodes if isinstance(lineage_nodes, set)else set(lineage_nodes) if lineage_nodes else None)

        results = []
        for node, hop in candidates.items():
            sem   = self._semantic_sim(node, query_embedding)
            gph   = self._graph_sim(node, seed_nodes)
            cite  = citation_counts.get(node, 0) / max_cite
            hop_s = self._hop_score(hop)
            yr    = self._year_score(node, max_year)
            ptype = self._paper_type_score(node, query_lower)
            canon = self._canonical_score(node, canon_set)
            topic = self._topic_match_score(node, topic_keywords)
            lin   = self._lineage_score(node, lineage_set)

            paper_year = (int(self.node_years[node]) if self.node_years is not None else max_year)
            if year_cutoff is not None and paper_year >= year_cutoff:
                yr *= 0.25   # strong penalty for post-cutoff papers

            total = (
                self.weights["semantic"] * sem
                + self.weights["topic_match"]  * topic
                + self.weights["graph"] * gph
                + self.weights["citation_freq"] * cite
                + self.weights["hop"] * hop_s
                + self.weights["year"] * yr
                + self.weights["canonical"] * canon
                + self.weights["lineage"] * lin
                + self.weights["paper_type"] * ptype
            )
            results.append({"node_idx": node,
                "score": total,"semantic": sem,
                "topic_match": topic,"graph": gph,
                "citation_freq": cite,"hop": hop,
                "year": paper_year,"year_score": yr,
                "canonical": canon,"lineage": lin,"paper_type": ptype,})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
