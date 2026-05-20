import os
import torch
import torch.nn.functional as F
from data.mag_dataloader import MagGraphDataset
from config import CFG
import numpy as np

class FeatureBuilder:

    def __init__(self, specter_path, output_path):

        self.specter_path = specter_path
        self.output_path = output_path

        self.dataset = MagGraphDataset(CFG.data)
        self.data = self.dataset.get_data()

    def load_specter_embeddings(self):
        specter_embeddings = np.load(self.specter_path)
        specter_embeddings = torch.tensor(specter_embeddings,dtype=torch.float32)
        print(f"Specter shape : {specter_embeddings.shape}")

        return specter_embeddings

    def validate_shapes(self, specter_embeddings):

        graph_nodes = self.data.num_nodes
        embedding_nodes = specter_embeddings.shape[0]
        print(f"Graph nodes : {graph_nodes}")
        print(f"Embedding nodes : {embedding_nodes}")

        assert graph_nodes == embedding_nodes, "Mismatch between graph nodes and embeddings"

    def build_features(self):

        specter_embeddings = self.load_specter_embeddings()
        self.validate_shapes(specter_embeddings)
        graph_features = self.data.x.float()
        print(f"Graph feature shape : {graph_features.shape}")
        final_features = torch.cat([graph_features, specter_embeddings],dim=1)
        final_features = F.normalize(final_features,p=2,dim=1)
        print(f"Final feature shape : {final_features.shape}")

        return final_features

    def save_features(self, final_features):

        os.makedirs(os.path.dirname(self.output_path),exist_ok=True)
        torch.save(final_features,self.output_path)
        print(f"Saved features : {self.output_path}")

    def run(self):

        final_features = self.build_features()
        self.save_features(final_features)

if __name__ == "__main__":

    builder = FeatureBuilder(specter_path="./outputs/embeddings/specter.npy",output_path="./outputs/embeddings/node_features.pt")
    builder.run()