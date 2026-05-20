import os
from typing import Optional
import faiss
import numpy as np
import torch

class FaissIndex:
    def __init__(self, dim: int, use_gpu: bool = False):
        self.dim = dim
        index = faiss.IndexFlatIP(dim)
        if use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_all_gpus(index)
        self.index = index

    def add(self, embeddings: np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)

    def search(self, query: np.ndarray, top_k: int = 10):
        if isinstance(query, torch.Tensor):
            query = query.cpu().numpy()
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        return scores[0], indices[0]

    def save(self, path: str):
        faiss.write_index(self.index, path)
        print(f"Saved FAISS index: {path}  ({self.ntotal} vectors, dim={self.dim})")

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        index = faiss.read_index(path)
        obj = cls.__new__(cls)
        obj.index = index
        obj.dim = index.d
        return obj

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

def build_semantic_index(specter_path: str, output_path: Optional[str] = None):
    embeddings = np.load(specter_path).astype(np.float32)
    print(f"Building semantic FAISS index from {embeddings.shape}")
    idx = FaissIndex(dim=embeddings.shape[1])
    idx.add(embeddings)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        idx.save(output_path)
    return idx

def build_graph_index(gnn_path: str, output_path: Optional[str] = None):
    embeddings = torch.load(gnn_path).numpy().astype(np.float32)
    print(f"Building graph FAISS index from {embeddings.shape}")
    idx = FaissIndex(dim=embeddings.shape[1])
    idx.add(embeddings)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        idx.save(output_path)
    return idx

if __name__ == "__main__":
    build_semantic_index(specter_path="./outputs/embeddings/specter.npy",output_path="./outputs/indices/semantic.faiss",)

    gnn_path = "./outputs/embeddings/gnn_embeddings20.pt"
    if os.path.exists(gnn_path):
        build_graph_index(gnn_path=gnn_path,output_path="./outputs/indices/graph.faiss",)