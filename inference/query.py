from __future__ import annotations
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class HybridPaperRetriever:
    """
    Retrieves seed papers using semantic (SPECTER2) and graph-aware (GNN)
    embeddings.  Falls back to semantic-only when gnn_embeddings_path is absent
    or the file doesn't exist yet (before training completes).

    Hybrid scoring
    --------------
    scores = semantic_weight * sem_scores + graph_weight * graph_scores

    The query is projected into graph embedding space by taking the mean of the
    top-K semantic hits' GNN embeddings — this avoids requiring a separate
    graph-space query encoder.
    """

    def __init__(self,metadata_path: str,specter_path: str,gnn_path: Optional[str] = None,
        model_name: str = "allenai/specter2_base",device: Optional[str] = None,semantic_weight: float = 0.80,graph_weight: float = 0.20,):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight
        self.df = pd.read_parquet(metadata_path)
        self._node_col = "node_idx" if "node_idx" in self.df.columns else "node_id"

        print(f"Using device: {self.device}")
        print(f"Loaded metadata: {self.df.shape}  (col: {self._node_col})")

        self.specter_embs = F.normalize(torch.tensor(np.load(specter_path), dtype=torch.float32),p=2,dim=1,).to(self.device)
        print(f"Semantic embs: {self.specter_embs.shape}")

        self.gnn_embs: Optional[torch.Tensor] = None
        if gnn_path and os.path.exists(gnn_path):
            self.gnn_embs = F.normalize(torch.load(gnn_path, map_location=self.device), p=2, dim=1)
            print(f"Graph embs: {self.gnn_embs.shape}  (hybrid mode ON)")
        else:
            print("Graph embs: not found — semantic-only mode")

        print("Loading SPECTER2 tokenizer & model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_query(self, query: str) -> torch.Tensor:
        encoded = self.tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors="pt", )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        out = self.model(**encoded)
        return F.normalize(out.last_hidden_state[:, 0], p=2, dim=1)  # [1, 768]

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 10) -> List[dict]:
        print(f"\nQuery: {query}")
        q_emb = self.encode_query(query)  # [1, 768]

        sem_scores = torch.matmul(self.specter_embs, q_emb.T).squeeze(1)  # [N]

        if self.gnn_embs is not None:
            k_bridge = min(20, sem_scores.shape[0])
            top_sem_idx = torch.topk(sem_scores, k=k_bridge).indices
            q_graph = F.normalize(self.gnn_embs[top_sem_idx].mean(dim=0, keepdim=True), p=2, dim=1)
            graph_scores = torch.matmul(self.gnn_embs, q_graph.T).squeeze(1)
            scores = self.semantic_weight * sem_scores + self.graph_weight * graph_scores
        else:
            scores = sem_scores

        top_scores, top_indices = torch.topk(scores, k=top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()), start=1):
            paper = self.df.iloc[idx]
            results.append({"rank": rank, "score": float(score), "node_idx": int(paper[self._node_col])
            ,"title": str(paper.get("title", "")), "abstract": str(paper.get("abstract", ""))[:300],})
        return results

    def print_results(self, results: List[dict]):
        print("\nTop Retrieved Papers")
        for r in results:
            print(f"\n[{r['rank']}] Score: {r['score']:.4f}")
            print(f"Node: {r['node_idx']}")
            print(f"Title: {r['title']}")
            print(f"Abstract: {r['abstract']}...")


if __name__ == "__main__":
    retriever = HybridPaperRetriever(metadata_path="./outputs/metadata/papers.parquet",
        specter_path="./outputs/embeddings/specter.npy",gnn_path="./outputs/embeddings/gnn_embeddings.pt",)

    for query in ["attention mechanism transformer", "variational autoencoder latent space", "graph neural network node classification",]:
        results = retriever.retrieve(query=query, top_k=10)
        retriever.print_results(results)
