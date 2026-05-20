import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import Compose, NormalizeFeatures, ToUndirected


class ArxivGraphDataset:
    """
    Wraps ogbn-arxiv as a PyG dataset with configurable transforms and
    NeighborLoader splits for mini-batch GNN training.
    """

    def __init__(self, config):
        self.config = config
        self._data = None
        self._split_idx = None
        self._load()

    def _build_transform(self):
        transforms = []
        if self.config.normalize_features:
            transforms.append(NormalizeFeatures())
        if self.config.make_undirected:
            transforms.append(ToUndirected())
        return Compose(transforms) if transforms else None

    def _load(self):
        dataset = PygNodePropPredDataset(
            name=self.config.dataset_name,
            root="./dataset",
            transform=self._build_transform(),
        )
        self._data = dataset[0]
        self._split_idx = dataset.get_idx_split()
        print(f"Loaded {self.config.dataset_name}")
        print(f"  Nodes: {self._data.num_nodes:,}  Edges: {self._data.num_edges:,}")
        print(f"  Node feature dim: {self._data.x.shape[1]}")

    def get_data(self):
        return self._data

    def get_split_idx(self):
        return self._split_idx

    def get_loaders(self):
        shared = dict(
            data=self._data,
            num_neighbors=list(self.config.num_neighbors),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        train_loader = NeighborLoader(
            input_nodes=self._split_idx["train"], shuffle=True, **shared
        )
        val_loader = NeighborLoader(
            input_nodes=self._split_idx["valid"], shuffle=False, **shared
        )
        test_loader = NeighborLoader(
            input_nodes=self._split_idx["test"], shuffle=False, **shared
        )
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config import CFG

    ds = ArxivGraphDataset(CFG.data)
    data = ds.get_data()
    print(f"\nNode features: {data.x.shape}")
    print(f"Edge index:    {data.edge_index.shape}")
    train_loader, val_loader, test_loader = ds.get_loaders()
    batch = next(iter(train_loader))
    print(f"\nSample batch: {batch}")
