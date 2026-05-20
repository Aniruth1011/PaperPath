import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

class SpecterEncoder:
    def __init__(self, model_name="allenai/specter2_base", device=None, batch_size=16, max_length=512):

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\nUsing device: {self.device}")
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("Loading model...")
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_batch(self, texts: List[str]):

        encoded = self.tokenizer(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt")

        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def encode_dataframe(self, df, text_column="text"):

        texts = df[text_column].tolist()
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        embeddings = np.concatenate(all_embeddings, axis=0)

        return embeddings

    def save_embeddings(self, embeddings, output_path):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, embeddings)
        print(f"\nSaved embeddings -> {output_path}")


if __name__ == "__main__":

    metadata_path = "outputs/metadata/papers.parquet"
    output_path = "outputs/embeddings/specter.npy"
    print("\nLoading metadata...")
    df = pd.read_parquet(metadata_path)
    print(df.shape)
    encoder = SpecterEncoder(batch_size=32)
    #sample_df = df.iloc[:1000]
    embeddings = encoder.encode_dataframe(df)
    print("\nEmbedding shape:")
    print(embeddings.shape)
    encoder.save_embeddings(embeddings, output_path)