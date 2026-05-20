from collections import defaultdict, deque
from typing import Dict, List, Tuple
import numpy as np
import torch

class SubgraphExtractor:
    def __init__(self, edge_index: torch.Tensor, num_nodes: int,
        embeddings: np.ndarray, semantic_threshold: float = 0.25, hub_threshold: int = 500, max_hops: int = 4, max_nodes: int = 300, multi_citation_min: int = 2,):
        self.num_nodes = num_nodes
        self.embeddings = embeddings
        self.semantic_threshold = semantic_threshold
        self.hub_threshold = hub_threshold
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.multi_citation_min = multi_citation_min

        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()

        self.out_adj: Dict[int, List[int]] = defaultdict(list)
        self.in_degree = np.zeros(num_nodes, dtype=np.int32)

        for s, d in zip(src, dst):
            self.out_adj[int(s)].append(int(d))
            self.in_degree[int(d)] += 1

    def _is_hub(self, node: int) -> bool:
        return int(self.in_degree[node]) > self.hub_threshold

    def _cosine_sim(self, node: int, query_emb: np.ndarray) -> float:
        return float(np.dot(self.embeddings[node], query_emb))

    def extract(self,seed_nodes: List[int],query_embedding: np.ndarray,) -> Tuple[Dict[int, int], Dict[int, int]]:
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        visited: Dict[int, int] = {}
        citation_counts: Dict[int, int] = defaultdict(int)
        frontier: deque = deque()

        for s in seed_nodes:
            if 0 <= s < self.num_nodes and s not in visited:
                visited[s] = 0
                frontier.append((s, 0))

        while frontier and len(visited) < self.max_nodes:
            node, hop = frontier.popleft()
            if hop >= self.max_hops:
                continue
            for neighbor in self.out_adj.get(node, []):
                citation_counts[neighbor] += 1
                if neighbor in visited:
                    continue
                multi_cited = citation_counts[neighbor] >= self.multi_citation_min
                sim = self._cosine_sim(neighbor, query_embedding)

                if self._is_hub(neighbor):
                    threshold = self.semantic_threshold + 0.15
                else:
                    threshold = (self.semantic_threshold * 0.95 if multi_cited
                                 else self.semantic_threshold)
                if hop == 0:
                    threshold *= 0.75

                if sim < threshold:
                    continue

                visited[neighbor] = hop + 1
                frontier.append((neighbor, hop + 1))

        return visited, dict(citation_counts)