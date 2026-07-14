# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A two-stage multi-modal pipeline for predicting 3D genome (Hi-C) contact matrices from DNA sequence:

1. **Stage 1 — Enformer fine-tuning**: Fine-tune a pretrained Enformer model to predict ChIP-seq epigenomic tracks from raw DNA sequence.
2. **Stage 2 — Graph Neural Network**: Use the ChIP-seq track outputs as node features in a GAT-based GNN to predict intra-chromosomal Hi-C contact matrices.

## Code Structure

```
models/
├── enformer.py       # EnformerFinetune, EnformerFineTunerPL, ChIPPredictionModel
└── gnn.py            # DNA3D_GAT, DNA3DIntteractionModel, DNA3DInteractionLinear, Predict3D/2

data_pipeline/
└── graph_data.py     # GraphDataset, EpiConcatMatrix, CreateGraphData, create_neighbor_graph
```

## Architecture

### Stage 1: Enformer Fine-tuning (`models/enformer.py`)
- `EnformerFinetune` — wraps pretrained Enformer, freezes backbone, replaces `_heads` with custom Linear+Softplus (3072 → `num_chip_tracks`)
- `EnformerFineTunerPL` — PyTorch Lightning wrapper; Poisson NLL loss; logs per-track Pearson correlation
- `ChIPPredictionModel` — standalone wrapper for inference use

### Stage 2: Graph Data & GNN

**`data_pipeline/graph_data.py`**
- `EpiConcatMatrix` — runs fine-tuned Enformer in sliding windows (stride = `ENFORMER_LENGTH - 320*128*2`) and concatenates outputs into per-bin ChIP-seq feature vectors
- `CreateGraphData` — builds `torch_geometric.data.Data` objects: node features = ChIP tracks, edges = k-nearest neighbor graph, target = Hi-C matrix from `.cool` file
- `GraphDataset` — PyG `InMemoryDataset` that saves/loads processed graphs to disk
- `create_neighbor_graph` — builds symmetric band-diagonal edge index with bandwidth `num_neighbors`

**`models/gnn.py`**
- `DNA3D_GAT` — primary model: 2× GATConv + positional embeddings + distance-aware MLP edge predictor (Softplus output); reconstructs symmetric N×N contact matrix; supports distance-decay regularization
- `DNA3DIntteractionModel` — multi-layer GAT + relative positional embeddings; dense N×N prediction
- `DNA3DInteractionLinear` — baseline: two linear layers + dense pairwise prediction, no GNN
- `Predict3D` / `Predict3D2` — batch wrappers for the above

### Key constants
- `ENFORMER_LENGTH = 196_608` bp — Enformer receptive field
- `GRAPH_REGION_DICT` — maps `"1M"` … `"32M"` to region sizes in bp
- Default resolution: 4096 bp/bin; default neighbors: 50 bins

## Data Dependencies

Expected file paths (not included in repo):
- `./data/hg38.fa` — reference genome (FASTA)
- `.cool` files — Hi-C contact matrices (via `cooler`)
- `.bed` files — genomic interval lists for training/validation
- Enformer checkpoint: `EleutherAI/enformer-official-rough` (loaded via `enformer_pytorch.from_pretrained`)
- Fine-tuned Enformer checkpoint: e.g. `./checkpoints/enformer-finetuned-epoch=09-val_loss_epoch=-0.1210.ckpt`

## Key Dependencies

```
torch, torch_geometric, pytorch_lightning
enformer_pytorch
cooler
kipoiseq, pyfaidx, pyBigWig
torchmetrics
```

## Running the Pipeline

**Build graph dataset:**
```python
from models import EnformerFineTunerPL
from data_pipeline import CreateGraphData, GraphDataset

data = CreateGraphData(
    interval_file="./develop_test/train_3d_region.bed",
    fasta_file="./data/hg38.fa",
    cool_file="./develop_test/wt_hic_4096.cool",
    region_length="2M",
    model=EnformerFineTunerPL.load_from_checkpoint("<ckpt_path>"),
)
outputs = data()
dataset = GraphDataset(root="./develop_test/graph_data_train_4096_2M", data_list=outputs)
```

**Train GNN** — see `finally_graph_neural_network.ipynb`.

**Fine-tune Enformer** — see `1_finetuning_enformer.ipynb`.

## Known Issues

- `data_pipeline/graph_data.py` imports `from data import FastaInterval` — this `data` module must be provided externally or replaced with `kipoiseq`/`pyfaidx` equivalents.
- `GraphDataset.process` saves `len(self.data_list)` as the third tuple element but loads it as `self.sizes` — the sizes field is the graph count, not per-attribute sizes.
