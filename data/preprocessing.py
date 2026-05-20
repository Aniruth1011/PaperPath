import io
import os

import pandas as pd


class ArxivMetadataBuilder:
    """
    Alignment backbone: joins nodeidx2paperid.csv.gz with titleabs.tsv so
    that row i of the output parquet corresponds exactly to graph node i.

    titleabs.tsv uses MAG paper IDs (not OGB node indices) in column 0,
    so a direct read without joining the mapping file gives wrong alignment.
    """

    def __init__(self, mapping_path: str, titleabs_path: str, output_path: str):
        self.mapping_path = mapping_path
        self.titleabs_path = titleabs_path
        self.output_path = output_path

    def load_node_mapping(self) -> pd.DataFrame:
        df = pd.read_csv(self.mapping_path, compression="gzip")
        df.columns = ["node_idx", "paper_id"]
        df["paper_id"] = df["paper_id"].astype(int)
        print(f"Loaded node mapping: {df.shape}")
        return df

    def load_titleabs(self) -> pd.DataFrame:
        # Read as bytes first to strip NUL characters that crash the CSV parser
        with open(self.titleabs_path, "rb") as f:
            raw = f.read().replace(b"\x00", b"")
        df = pd.read_csv(
            io.StringIO(raw.decode("utf-8", errors="replace")),
            sep="\t",
            header=None,
            skiprows=1,
            names=["paper_id", "title", "abstract"],
        )
        df = df.dropna(subset=["paper_id"])
        df["paper_id"] = df["paper_id"].astype(int)
        df = df.drop_duplicates(subset=["paper_id"])
        df["title"] = df["title"].fillna("")
        df["abstract"] = df["abstract"].fillna("")
        df["text"] = df["title"].astype(str) + " " + df["abstract"].astype(str)
        print(f"Loaded titleabs: {df.shape}")
        return df

    def align_metadata(self, mapping_df: pd.DataFrame, titleabs_df: pd.DataFrame) :
        merged = mapping_df.merge(titleabs_df, on="paper_id", how="inner")
        merged = merged.sort_values("node_idx").reset_index(drop=True)
        missing = mapping_df.shape[0] - merged.shape[0]
        if missing > 0:
            print(f"Warning: {missing} nodes have missing")
        print(f"Aligned metadata: {merged.shape}  (row i = graph node i)")
        return merged

    def save_metadata(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_parquet(self.output_path, index=False)
        print(f"Saved: {self.output_path}")

    def build(self) -> pd.DataFrame:
        mapping_df = self.load_node_mapping()
        titleabs_df = self.load_titleabs()
        merged = self.align_metadata(mapping_df, titleabs_df)
        self.save_metadata(merged)
        print("\nSample:")
        print(merged[["node_idx", "paper_id", "title"]].head())
        return merged


if __name__ == "__main__":
    builder = ArxivMetadataBuilder(mapping_path="./dataset/ogbn_mag/mapping/nodeidx2paperid.csv.gz",titleabs_path="./dataset/ogbn_arxiv/raw/titleabs.tsv",output_path="./outputs/metadata/papers.parquet",)
    builder.build()
