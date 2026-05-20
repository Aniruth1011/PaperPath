import os
import sys
from collections import defaultdict
from typing import Dict, Set

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from config import CFG
from data.dataloader import ArxivGraphDataset
from models.model import GraphSAGE
from training.loss import (
    cocitation_contrastive_loss,
    link_prediction_loss,
    sample_hard_negatives,
    sample_negative_edges,
)


class Trainer:
    def __init__(self, config=CFG):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Graph for message-passing (undirected for richer neighbourhood)
        self.dataset = ArxivGraphDataset(self.cfg.data)
        self.data = self.dataset.get_data()
        self.train_loader, _, _ = self.dataset.get_loaders()

        # Load node features
        feat_path = "./outputs/embeddings/node_features.pt"
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Run feature_builder.py first: {feat_path}"
            )
        self.node_features = torch.load(feat_path).to(self.device)
        print(f"Loaded node features: {self.node_features.shape}")

        # Precompute structures for hard negatives and co-citation loss.
        # Uses the ORIGINAL directed graph (no undirected transform) so that
        # "paper A cites paper B" has a clear direction.
        self._build_loss_structures()

        # Model & optimiser
        self.model = GraphSAGE(
            in_channels=self.cfg.model.in_channels,
            hidden_channels=self.cfg.model.hidden_channels,
            out_channels=self.cfg.model.out_channels,
            num_layers=self.cfg.model.num_layers,
            dropout=self.cfg.model.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.train.lr
        )
        os.makedirs(self.cfg.train.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Precompute adjacency and label structures (one-time, on CPU)
    # ------------------------------------------------------------------

    def _build_loss_structures(self):
        print("Precomputing adjacency and label structures...")
        directed = PygNodePropPredDataset(
            name=self.cfg.data.dataset_name, root="./dataset"
        )[0]

        src = directed.edge_index[0].numpy()
        dst = directed.edge_index[1].numpy()

        # Directed out-adjacency as sets — O(1) membership test for filtering
        self.out_adj_sets: Dict[int, Set[int]] = defaultdict(set)
        for s, d in zip(src.tolist(), dst.tolist()):
            self.out_adj_sets[s].add(d)

        # Node category labels for hard negative sampling
        self.node_labels: np.ndarray = directed.y.squeeze().numpy()
        nodes = np.arange(directed.num_nodes, dtype=np.int64)
        self.label_to_nodes: Dict[int, np.ndarray] = {
            int(lbl): nodes[self.node_labels == lbl]
            for lbl in np.unique(self.node_labels)
        }
        n_labels = len(self.label_to_nodes)
        print(f"  Adjacency: {len(self.out_adj_sets):,} nodes with out-edges")
        print(f"  Labels: {n_labels} arxiv categories")

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {epoch}")
        ):
            batch = batch.to(self.device)
            batch_size = batch.batch_size

            x = self.node_features[batch.n_id]
            z = self.model(x, batch.edge_index)
            z_seed = z[:batch_size]

            # Positive edges (both endpoints are seed nodes)
            mask = (batch.edge_index[0] < batch_size) & (
                batch.edge_index[1] < batch_size
            )
            pos_edge = batch.edge_index[:, mask]
            if pos_edge.shape[1] == 0:
                continue

            # Negative edges ─ hard or random fallback
            if self.cfg.train.use_hard_negatives:
                seed_globals = batch.n_id[:batch_size].cpu().numpy()
                neg_edge = sample_hard_negatives(
                    seed_globals=seed_globals,
                    batch_n_id=batch.n_id,
                    batch_size=batch_size,
                    node_labels=self.node_labels,
                    label_to_nodes=self.label_to_nodes,
                    out_adj_sets=self.out_adj_sets,
                    num_neg_per_node=self.cfg.train.hard_neg_per_node,
                    device=self.device,
                )
            else:
                num_neg = pos_edge.shape[1] * self.cfg.train.neg_ratio
                neg_edge = sample_negative_edges(batch_size, num_neg, self.device)

            loss = link_prediction_loss(z_seed, pos_edge, neg_edge)

            # Co-citation contrastive loss
            if self.cfg.train.cocitation_weight > 0:
                cc_loss = cocitation_contrastive_loss(
                    z=z,
                    batch_n_id=batch.n_id,
                    batch_size=batch_size,
                    out_adj_sets=self.out_adj_sets,
                    temperature=self.cfg.train.cocitation_temperature,
                )
                if cc_loss is not None:
                    loss = loss + self.cfg.train.cocitation_weight * cc_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % self.cfg.train.log_interval == 0:
                print(
                    f"  Batch {batch_idx + 1}  "
                    f"loss={total_loss / num_batches:.4f}"
                )

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Inference: generate embeddings for every node
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_embeddings(self) -> torch.Tensor:
        self.model.eval()
        num_nodes = self.data.num_nodes
        embeddings = torch.zeros(num_nodes, self.cfg.model.out_channels)

        inference_loader = NeighborLoader(
            data=self.data,
            num_neighbors=list(self.cfg.data.num_neighbors),
            batch_size=self.cfg.data.batch_size * 2,
            input_nodes=torch.arange(num_nodes),
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )

        for batch in tqdm(inference_loader, desc="Generating embeddings"):
            batch = batch.to(self.device)
            x = self.node_features[batch.n_id]
            z = self.model(x, batch.edge_index)
            embeddings[batch.n_id[: batch.batch_size]] = z[: batch.batch_size].cpu()

        return F.normalize(embeddings, p=2, dim=1)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, loss: float):
        path = os.path.join(
            self.cfg.train.checkpoint_dir, f"checkpoint_epoch{epoch}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def train(self) -> torch.Tensor:
        print(f"\nStarting training for {self.cfg.train.epochs} epochs")
        print(f"  Hard negatives: {self.cfg.train.use_hard_negatives}")
        print(f"  Co-citation weight: {self.cfg.train.cocitation_weight}")

        for epoch in range(1, self.cfg.train.epochs + 1):
            loss = self._train_epoch(epoch)
            print(f"Epoch {epoch}/{self.cfg.train.epochs}  loss={loss:.4f}")
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, loss)

        print("\nGenerating GNN embeddings for all papers...")
        embeddings = self.generate_embeddings()
        out_path = self.cfg.train.embeddings_output
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(embeddings, out_path)
        print(f"Saved GNN embeddings: {out_path}  {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()