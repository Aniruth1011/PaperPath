# PreReqGraph

**Automatic prerequisite curriculum discovery for research papers.**

Given a natural-language query (e.g. `"attention transformer"`), PreReqGraph traverses the academic citation graph, identifies the foundational papers you need to read first, and organises them into a staged learning path from earliest to most recent.

---

## Problem

Understanding a new research topic requires knowing *which papers to read, and in what order*. Manually tracing citation chains across thousands of papers is slow and requires domain expertise.

## Approach

PreReqGraph combines semantic embeddings with graph-structure reasoning over the [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) citation network (169K CS papers, 1.2M citation edges):

1. **SPECTER2 encoding** — encode every paper's title + abstract into a 768-dim scientific language embedding.
2. **GraphSAGE training** — fine-tune a 3-layer GraphSAGE on the citation graph using link-prediction loss with hard negative sampling (same-category negatives) and a co-citation contrastive objective. Output: 128-dim graph-aware embeddings per node.
3. **Hybrid retrieval** — combine SPECTER2 semantic scores (80%) and GNN graph-proximity scores (20%) to retrieve seed papers. FAISS `IndexFlatIP` enables sub-10ms ANN search over all 169K paper embeddings.
4. **Intent parsing** — extract canonical paper references, topic keywords, and query intent from the natural-language query.
5. **Survey filter** — survey papers in the seed set are replaced by their most query-relevant cited papers to avoid broad/shallow results.
6. **BFS subgraph extraction** — breadth-first search from seed nodes through the citation graph, gating expansion on semantic similarity to the query and suppressing high-degree hub papers.
7. **Multi-signal scoring** — each candidate paper is scored across 9 signals: semantic similarity, topic keyword match, canonical-paper prior, lineage (cited-by-canonical), graph proximity, citation frequency, BFS hop distance, publication year, and paper type (foundational vs systems).
8. **Curriculum builder** — top-ranked papers are grouped into chronological stages to produce a readable learning path.

---

## Tech Stack

| Component | Technology |
|---|---|
| Citation graph | ogbn-arxiv (OGB), PyTorch Geometric |
| Semantic embeddings | SPECTER2 (`allenai/specter2_base`), HuggingFace Transformers |
| GNN model | GraphSAGE (PyTorch Geometric) |
| ANN search | FAISS `IndexFlatIP` |
| Data | pandas, NumPy |
| Training | PyTorch, NeighborLoader mini-batch sampling |

---

## Usage

### 1. Preprocess & encode

```bash
python data/preprocessing.py          # build papers.parquet
python data/spectral_encoder.py       # generate specter.npy (SPECTER2 embeddings)
python feature_builder.py             # concatenate OGB + SPECTER2 node features
```

### 2. Train the GNN

```bash
python training/trainer_arxiv.py      # trains GraphSAGE, saves gnn_embeddings.pt
```

### 3. Build FAISS indices

```bash
python utils/faiss_index.py
```

### 4. Run a query

```bash
python pipeline.py "attention mechanism transformer"
python pipeline.py "variational autoencoder" --top-k 15 --show-scores
python pipeline.py "graph neural network" --year-cutoff 2020
```

**Example output:**
```
  ┌──────────────────────────────────────────────────────────────────┐
  │  QUERY TARGET  (canonical match, sim=0.912)
  │  [2017]  Attention Is All You Need
  └──────────────────────────────────────────────────────────────────┘

Top-10 Prerequisites:
   1. ★  [2013]  0.821  Efficient Estimation of Word Representations in Vector Space
   2. →  [2014]  0.807  Neural Machine Translation by Jointly Learning to Align and Translate
   ...

CURRICULUM  (earliest → most recent)
Stage 1:  [2013] Word2Vec  •  [2014] Seq2Seq  •  [2015] Residual Networks
Stage 2:  [2015] Batch Norm  •  [2016] Layer Norm
Stage 3:  [2017] Attention Is All You Need
Stage 4:  [2018] BERT
```

---

## Architecture

```
Query (text)
    │
    ├─► SPECTER2 encoder ──► query embedding (768-dim)
    │
    ├─► Hybrid Retriever (FAISS)
    │       └─ 80% semantic + 20% GNN graph proximity → top-k seed papers
    │
    ├─► Intent Parser
    │       └─ canonical paper nodes, topic keywords
    │
    ├─► Survey Filter
    │       └─ replace survey seeds with cited neighbours
    │
    ├─► BFS Subgraph Extractor
    │       └─ citation-graph traversal (max 4 hops, semantic gating)
    │
    ├─► Multi-Signal Scorer (9 signals, weighted sum)
    │
    └─► Curriculum Builder
            └─ chronological stages → learning path
```

---

## Configuration

Key settings are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `in_channels` | 896 | 128 OGB features + 768 SPECTER2 |
| `hidden_channels` | 256 | GNN hidden size |
| `out_channels` | 128 | GNN embedding dim |
| `num_layers` | 3 | GraphSAGE depth |
| `hard_neg_per_node` | 2 | Same-category negatives per training step |
| `cocitation_weight` | 0.3 | Co-citation contrastive loss weight |
| `semantic_threshold` | 0.25 | BFS expansion cosine similarity gate |
| `max_hops` | 4 | Max BFS depth |
| `max_nodes` | 300 | Max subgraph size |
