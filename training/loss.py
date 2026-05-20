from typing import Dict, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F

def link_prediction_loss(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
) -> torch.Tensor:
    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    return -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()


def sample_negative_edges(
    num_nodes: int, num_neg: int, device: torch.device
) -> torch.Tensor:
    """Baseline: uniformly random negative edges (used as fallback)."""
    src = torch.randint(0, num_nodes, (num_neg,), device=device)
    dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    return torch.stack([src, dst])

def sample_hard_negatives(
    seed_globals: np.ndarray,
    batch_n_id: torch.Tensor,
    batch_size: int,
    node_labels: np.ndarray,
    label_to_nodes: Dict[int, np.ndarray],
    out_adj_sets: Dict[int, Set[int]],
    num_neg_per_node: int,
    device: torch.device,
) -> torch.Tensor:
    # Map global seed IDs → local indices [0, batch_size)
    seed_global_to_local = {int(g): i for i, g in enumerate(seed_globals)}

    src_list, dst_list = [], []

    for local_i, global_i in enumerate(seed_globals):
        label = int(node_labels[global_i])
        candidates = label_to_nodes.get(label, np.array([], dtype=np.int64))
        cited = out_adj_sets.get(int(global_i), set())

        # Valid = same category + not cited + is also a seed node in this batch
        valid = [
            int(c) for c in candidates
            if int(c) != global_i
            and int(c) not in cited
            and int(c) in seed_global_to_local
        ]

        if not valid:
            continue

        n = min(num_neg_per_node, len(valid))
        chosen = np.random.choice(valid, size=n, replace=False)
        for c in chosen:
            src_list.append(local_i)
            dst_list.append(seed_global_to_local[int(c)])

    if not src_list:
        return sample_negative_edges(batch_size, max(1, batch_size), device)

    return torch.stack([
        torch.tensor(src_list, dtype=torch.long, device=device),
        torch.tensor(dst_list, dtype=torch.long, device=device),
    ])


# ---------------------------------------------------------------------------
# Co-citation contrastive loss
# ---------------------------------------------------------------------------

def cocitation_contrastive_loss(
    z: torch.Tensor,
    batch_n_id: torch.Tensor,
    batch_size: int,
    out_adj_sets: Dict[int, Set[int]],
    temperature: float = 0.07,
    max_pairs: int = 256,
) -> Optional[torch.Tensor]:
    n_id_np = batch_n_id.cpu().numpy()
    node_to_local = {int(n): i for i, n in enumerate(n_id_np)}

    pairs = []
    for src_global in n_id_np[:batch_size]:
        cited_globals = out_adj_sets.get(int(src_global), set())
        # Keep only cited nodes that happen to be in this batch
        cited_local = [node_to_local[c] for c in cited_globals if c in node_to_local]
        if len(cited_local) < 2:
            continue
        # Cap at 8 to avoid O(n²) blow-up on high-degree nodes
        cap = min(len(cited_local), 8)
        for i in range(cap):
            for j in range(i + 1, cap):
                pairs.append((cited_local[i], cited_local[j]))

    if len(pairs) < 4:
        return None

    if len(pairs) > max_pairs:
        idx = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    pairs_t = torch.tensor(pairs, dtype=torch.long, device=z.device)
    z_i = F.normalize(z[pairs_t[:, 0]], dim=1)
    z_j = F.normalize(z[pairs_t[:, 1]], dim=1)

    # Symmetric InfoNCE: each row is one anchor, positives are on the diagonal
    logits = torch.mm(z_i, z_j.T) / temperature
    labels = torch.arange(len(pairs_t), device=z.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss
