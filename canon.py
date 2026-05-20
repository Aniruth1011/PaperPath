from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

@dataclass
class TopicDescriptor:
    label: str
    canonical_frags: List[str]
    topic_keywords: Set[str]

REGISTRY: Dict[str, TopicDescriptor] = {

    "attention is all you need": TopicDescriptor(
        label="transformer",
        canonical_frags=["attention is all you need"],
        topic_keywords={"transformer", "self-attention", "multi-head", "encoder",
                        "decoder", "positional encoding", "attention mechanism",
                        "sequence transduction"},
    ),
    "transformer": TopicDescriptor(
        label="transformer",
        canonical_frags=["attention is all you need"],
        topic_keywords={"transformer", "self-attention", "multi-head", "attention",
                        "encoder", "decoder", "sequence", "positional"},
    ),
    "attention mechanism": TopicDescriptor(
        label="attention",
        canonical_frags=["neural machine translation by jointly learning to align",
                         "attention is all you need"],
        topic_keywords={"attention", "alignment", "encoder", "decoder",
                        "neural machine translation", "seq2seq", "context vector"},
    ),
    "self-attention": TopicDescriptor(
        label="self_attention",
        canonical_frags=["attention is all you need"],
        topic_keywords={"self-attention", "transformer", "multi-head", "attention"},
    ),

    "language model pretraining": TopicDescriptor(
        label="lm_pretraining",
        canonical_frags=["bert: pre-training",
                         "language models are unsupervised multitask learners",
                         "improving language understanding by generative pre-training"],
        topic_keywords={"language model", "pretraining", "fine-tuning", "bert", "gpt",
                        "transfer learning", "masked language model", "nlp"},
    ),
    "bert": TopicDescriptor(
        label="bert",
        canonical_frags=["bert: pre-training of deep bidirectional transformers",
                         "bert:"],
        topic_keywords={"bert", "bidirectional", "language model", "pretraining",
                        "fine-tuning", "masked", "next sentence", "transformer"},
    ),
    "gpt": TopicDescriptor(
        label="gpt",
        canonical_frags=["improving language understanding by generative pre-training",
                         "language models are unsupervised multitask learners"],
        topic_keywords={"gpt", "language model", "autoregressive", "pretraining",
                        "transformer", "text generation", "few-shot"},
    ),

    "generative adversarial network": TopicDescriptor(
        label="gan",
        canonical_frags=["generative adversarial net"],
        topic_keywords={"generative", "adversarial", "generator", "discriminator",
                        "image synthesis", "image generation", "generative model",
                        "minimax", "nash equilibrium"},
    ),
    "generative adversarial": TopicDescriptor(
        label="gan",
        canonical_frags=["generative adversarial net"],
        topic_keywords={"generative", "adversarial", "generator", "discriminator",
                        "image synthesis", "generative model"},
    ),
    "gan": TopicDescriptor(
        label="gan",
        canonical_frags=["generative adversarial net"],
        topic_keywords={"generative", "adversarial", "generator", "discriminator",
                        "image synthesis", "generative model", "unsupervised"},
    ),

    "variational autoencoder": TopicDescriptor(
        label="vae",
        canonical_frags=["auto-encoding variational bayes",
                         "stochastic backpropagation and approximate inference"],
        topic_keywords={"variational", "autoencoder", "latent", "encoder", "decoder",
                        "reparameterization", "elbo", "kl divergence",
                        "probabilistic", "generative model", "inference"},
    ),
    "vae": TopicDescriptor(
        label="vae",
        canonical_frags=["auto-encoding variational bayes"],
        topic_keywords={"variational", "autoencoder", "latent", "encoder", "decoder",
                        "reparameterization", "generative model", "probabilistic"},
    ),

    # ── Diffusion / Score Matching ────────────────────────────────────────────
    "denoising diffusion": TopicDescriptor(
        label="ddpm",
        canonical_frags=["denoising diffusion probabilistic models"],
        topic_keywords={"diffusion", "denoising", "noise", "markov chain",
                        "score matching", "generative", "sampling"},
    ),
    "diffusion model": TopicDescriptor(
        label="diffusion",
        canonical_frags=["denoising diffusion probabilistic models",
                         "score-based generative modeling through stochastic differential"],
        topic_keywords={"diffusion", "denoising", "score matching", "stochastic",
                        "generative", "noise schedule", "langevin"},
    ),
    "score matching": TopicDescriptor(
        label="score_matching",
        canonical_frags=["estimation of non-normalized statistical models",
                         "generative modeling by estimating gradients of the data distribution"],
        topic_keywords={"score matching", "energy-based", "unnormalized", "gradient",
                        "statistical model", "noise contrastive"},
    ),

    "deep residual": TopicDescriptor(
        label="resnet",
        canonical_frags=["deep residual learning for image recognition"],
        topic_keywords={"residual", "deep network", "convolutional", "skip connection",
                        "identity mapping", "image recognition", "image classification"},
    ),
    "residual network": TopicDescriptor(
        label="resnet",
        canonical_frags=["deep residual learning for image recognition",
                         "identity mappings in deep residual networks"],
        topic_keywords={"residual", "deep network", "skip connection", "convolutional",
                        "image classification"},
    ),
    "resnet": TopicDescriptor(
        label="resnet",
        canonical_frags=["deep residual learning for image recognition"],
        topic_keywords={"residual", "skip connection", "deep network", "convolutional",
                        "image recognition"},
    ),

    "graph neural network": TopicDescriptor(
        label="gnn",
        canonical_frags=["semi-supervised classification with graph convolutional",
                         "the graph neural network model"],
        topic_keywords={"graph", "node", "edge", "convolution", "message passing",
                        "graph neural", "network embedding", "spectral", "laplacian",
                        "node classification", "link prediction"},
    ),
    "graph convolutional": TopicDescriptor(
        label="graph_conv",
        canonical_frags=["semi-supervised classification with graph convolutional"],
        topic_keywords={"graph", "convolutional", "node", "spectral", "laplacian",
                        "message passing", "semi-supervised"},
    ),
    "graph attention": TopicDescriptor(
        label="graph_attention",
        canonical_frags=["graph attention networks"],
        topic_keywords={"graph", "attention", "node", "edge", "graph neural",
                        "graph attention"},
    ),
    "node classification": TopicDescriptor(
        label="node_clf",
        canonical_frags=["semi-supervised classification with graph convolutional"],
        topic_keywords={"graph", "node", "classification", "semi-supervised",
                        "label propagation", "network"},
    ),
    "link prediction": TopicDescriptor(
        label="link_pred",
        canonical_frags=["link prediction in networks"],
        topic_keywords={"link prediction", "graph", "network", "edge", "node embedding",
                        "knowledge graph"},
    ),
    "gnn": TopicDescriptor(
        label="gnn",
        canonical_frags=["semi-supervised classification with graph convolutional"],
        topic_keywords={"graph", "node", "edge", "convolution", "message passing",
                        "graph embedding", "network"},
    ),

    "word2vec": TopicDescriptor(
        label="word2vec",
        canonical_frags=["efficient estimation of word representations in vector space",
                         "distributed representations of words and phrases"],
        topic_keywords={"word2vec", "word embedding", "skip-gram", "cbow",
                        "distributed representation", "semantic", "word vector",
                        "neural language model"},
    ),
    "word embedding": TopicDescriptor(
        label="word_embedding",
        canonical_frags=["efficient estimation of word representations in vector space",
                         "glove: global vectors for word representation"],
        topic_keywords={"word embedding", "word vector", "distributed representation",
                        "word2vec", "glove", "semantic", "language model"},
    ),
    "glove": TopicDescriptor(
        label="glove",
        canonical_frags=["glove: global vectors for word representation"],
        topic_keywords={"glove", "word vector", "co-occurrence", "word embedding",
                        "semantic", "distributed representation"},
    ),

    "sequence to sequence": TopicDescriptor(
        label="seq2seq",
        canonical_frags=["sequence to sequence learning with neural networks"],
        topic_keywords={"seq2seq", "encoder", "decoder", "machine translation",
                        "sequence modeling", "lstm", "recurrent"},
    ),
    "recurrent neural network": TopicDescriptor(
        label="rnn",
        canonical_frags=["long short-term memory", "finding structure in time"],
        topic_keywords={"recurrent", "rnn", "sequence", "temporal", "memory",
                        "lstm", "gru", "language model", "time series"},
    ),
    "lstm": TopicDescriptor(
        label="lstm",
        canonical_frags=["long short-term memory"],
        topic_keywords={"lstm", "recurrent", "memory", "gate", "vanishing gradient",
                        "sequence", "time series", "rnn"},
    ),

    # ── Normalisation ─────────────────────────────────────────────────────────
    "batch normalization": TopicDescriptor(
        label="batch_norm",
        canonical_frags=["batch normalization: accelerating deep network training"],
        topic_keywords={"batch normalization", "normalization", "internal covariate",
                        "deep network", "training stability", "optimization"},
    ),
    "layer normalization": TopicDescriptor(
        label="layer_norm",
        canonical_frags=["layer normalization"],
        topic_keywords={"layer normalization", "normalization", "recurrent",
                        "transformer", "training"},
    ),

    # ── Dropout / Regularisation ──────────────────────────────────────────────
    "dropout": TopicDescriptor(
        label="dropout",
        canonical_frags=["dropout: a simple way to prevent neural networks from overfitting"],
        topic_keywords={"dropout", "regularization", "overfitting", "stochastic",
                        "neural network training", "co-adaptation"},
    ),

    # ── AlexNet / ImageNet ────────────────────────────────────────────────────
    "alexnet": TopicDescriptor(
        label="alexnet",
        canonical_frags=["imagenet classification with deep convolutional neural networks"],
        topic_keywords={"alexnet", "imagenet", "deep convolutional", "relu",
                        "image classification", "gpu", "convolutional neural network"},
    ),
    "imagenet": TopicDescriptor(
        label="imagenet",
        canonical_frags=["imagenet classification with deep convolutional neural networks",
                         "imagenet large scale visual recognition challenge"],
        topic_keywords={"imagenet", "image classification", "convolutional", "deep learning",
                        "visual recognition", "benchmark"},
    ),

    # ── Vision Transformer ────────────────────────────────────────────────────
    "vision transformer": TopicDescriptor(
        label="vit",
        canonical_frags=["an image is worth 16x16 words"],
        topic_keywords={"vision transformer", "vit", "patch", "image classification",
                        "transformer", "visual recognition"},
    ),
    "vit": TopicDescriptor(
        label="vit",
        canonical_frags=["an image is worth 16x16 words"],
        topic_keywords={"vit", "vision transformer", "image", "transformer",
                        "patch", "classification"},
    ),

    # ── Optimisers ────────────────────────────────────────────────────────────
    "adam optimizer": TopicDescriptor(
        label="adam",
        canonical_frags=["adam: a method for stochastic optimization"],
        topic_keywords={"adam", "optimizer", "stochastic gradient", "adaptive",
                        "momentum", "rmsprop", "learning rate", "convergence"},
    ),
    "stochastic gradient descent": TopicDescriptor(
        label="sgd",
        canonical_frags=["on the importance of initialization and momentum in deep learning"],
        topic_keywords={"sgd", "stochastic gradient", "momentum", "learning rate",
                        "optimization", "convergence"},
    ),

    # ── Contrastive Learning ──────────────────────────────────────────────────
    "contrastive learning": TopicDescriptor(
        label="contrastive",
        canonical_frags=["a simple framework for contrastive learning of visual representations",
                         "momentum contrast for unsupervised visual representation learning"],
        topic_keywords={"contrastive", "self-supervised", "representation learning",
                        "augmentation", "positive pair", "negative pair", "instance"},
    ),

    # ── Reinforcement Learning ────────────────────────────────────────────────
    "deep reinforcement learning": TopicDescriptor(
        label="deep_rl",
        canonical_frags=["playing atari with deep reinforcement learning",
                         "human-level control through deep reinforcement learning"],
        topic_keywords={"reinforcement learning", "q-learning", "policy", "reward",
                        "atari", "dqn", "actor-critic", "environment"},
    ),
    "policy gradient": TopicDescriptor(
        label="policy_gradient",
        canonical_frags=["proximal policy optimization algorithms",
                         "high-dimensional continuous control using generalized advantage estimation"],
        topic_keywords={"policy gradient", "actor-critic", "ppo", "advantage",
                        "reward", "policy", "value function"},
    ),
}

@dataclass
class QueryIntent:
    raw_query: str
    label: str
    canonical_frags: List[str]
    topic_keywords: Set[str]
    canonical_nodes: List[int] = field(default_factory=list)

def parse_intent(query: str) -> QueryIntent:
    """
    Match the lowercased query against REGISTRY triggers (longest-match wins).
    """
    q = query.lower().strip()

    best_key: Optional[str] = None
    best_len: int = 0
    for trigger in REGISTRY:
        if trigger in q and len(trigger) > best_len:
            best_key = trigger
            best_len = len(trigger)

    if best_key is not None:
        desc = REGISTRY[best_key]
        return QueryIntent(
            raw_query=query,
            label=desc.label,
            canonical_frags=list(desc.canonical_frags),
            topic_keywords=set(desc.topic_keywords),
        )

    # Generic fallback: use non-trivial query words as topic keywords
    words = {w for w in q.split() if len(w) > 3}
    return QueryIntent(
        raw_query=query,
        label="general",
        canonical_frags=[],
        topic_keywords=words,
    )


def resolve_canonical_nodes(intent: QueryIntent, papers_df: pd.DataFrame) -> List[int]:
    """
    Scan papers_df titles for canonical fragments; update intent.canonical_nodes.
    Uses vectorised pandas .str.contains for speed (~170 K rows in ogbn-arxiv).
    """
    if not intent.canonical_frags:
        intent.canonical_nodes = []
        return []

    titles_lower = papers_df["title"].str.lower().fillna("")
    seen: Dict[int, None] = {}  # ordered dedup

    for frag in intent.canonical_frags:
        mask = titles_lower.str.contains(frag.lower(), regex=False, na=False)
        for idx in papers_df.index[mask].tolist():
            seen[idx] = None

    intent.canonical_nodes = list(seen.keys())
    return intent.canonical_nodes


def identify_closest_paper(
    intent: QueryIntent,
    papers_df: pd.DataFrame,
    specter_embs: np.ndarray,
    q_emb: np.ndarray,
    node_years: Optional[np.ndarray] = None,
) -> Dict:
    # Normalise embeddings for cosine similarity
    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / (n + 1e-8)

    q_norm = _l2(q_emb)

    if intent.canonical_nodes:
        nodes = np.array(intent.canonical_nodes)
        embs = _l2(specter_embs[nodes])           # (k, 768)
        sims = embs @ q_norm                       # (k,)
        best_local = int(np.argmax(sims))
        best_node = int(nodes[best_local])
        sim_val = float(sims[best_local])
        match_type = "canonical"
    else:
        embs = _l2(specter_embs)
        sims = embs @ q_norm                       # (N,)
        best_node = int(np.argmax(sims))
        sim_val = float(sims[best_node])
        match_type = "semantic_top1"

    row = papers_df.iloc[best_node]
    year = (int(node_years[best_node])
            if node_years is not None and best_node < len(node_years)
            else None)

    return {
        "node_idx":     best_node,
        "title":        str(row.get("title", f"node_{best_node}")),
        "year":         year,
        "semantic_sim": sim_val,
        "match_type":   match_type,
    }
