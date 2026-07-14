# Epi3D

> A two-stage deep learning framework that predicts 3D genome organization (Hi-C contact matrices) directly from raw DNA sequence, validated on real ChIP-seq and Hi-C sequencing data.

---

## Overview

Understanding how the genome folds in three dimensions is fundamental to deciphering gene regulation. Epi3D addresses this by combining two complementary models:

1. **Stage 1 — Epigenomic feature extraction**: Fine-tune a pretrained [Enformer](https://github.com/lucidrains/enformer-pytorch) model to predict ChIP-seq epigenomic tracks from raw DNA sequence.
2. **Stage 2 — 3D contact prediction**: Use the predicted epigenomic tracks as node features in a Graph Attention Network (GAT) to predict intra-chromosomal Hi-C contact matrices.

The key insight is that epigenomic signals such as histone modifications and transcription factor binding capture the intermediate regulatory state between DNA sequence and 3D chromatin structure.

---

## Architecture

### Stage 1: Enformer Fine-tuning

```
DNA sequence (196,608 bp, one-hot encoded)
            ↓
  Enformer backbone (weights frozen)
            ↓
  Linear(3072 → num_chip_tracks) + Softplus
            ↓
  ChIP-seq tracks  [896 bins × num_chip_tracks]
```

The Enformer backbone is kept frozen; only the output head is trained, reducing compute requirements while leveraging the model's pretrained sequence representations.

### Stage 2: GAT-based Contact Prediction

```
ChIP-seq feature vectors  (one node per 4 kb genomic bin)
            ↓
  Band-diagonal k-nearest neighbor graph
            ↓
  Positional embeddings + 2× GATConv layers
            ↓
  Distance-aware MLP edge predictor  (Softplus output)
            ↓
  Symmetric N×N Hi-C contact matrix
```

A distance-decay regularization term encourages the model to respect the well-established inverse relationship between genomic distance and contact frequency.

### Model Variants

| Model | Description |
|-------|-------------|
| `DNA3D_GAT` | Primary model: 2-layer GAT with positional embeddings and distance-decay regularization |
| `DNA3DInteractionModel` | Multi-layer GAT with relative positional embeddings; dense N×N prediction |
| `DNA3DInteractionLinear` | Linear baseline with pairwise MLP; no message passing |

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torch_geometric`, `pytorch_lightning`, `enformer_pytorch`, `cooler`, `kipoiseq`, `pyfaidx`, `pyBigWig`.

---

## Data Preparation

| File | Description |
|------|-------------|
| `./data/hg38.fa` | Human reference genome (GRCh38/hg38) in FASTA format |
| `*.cool` | Hi-C contact matrices at target resolution (e.g., 4 kb), [cooler](https://cooler.readthedocs.io) format |
| `*.bw` | ChIP-seq signal tracks in BigWig format, used as training targets for Stage 1 |
| `*.bed` | Genomic interval lists defining training and validation regions |

Regions in the BED file should be sized to match the chosen `region_length` (e.g., 2 Mb). Overlapping train/val regions should be avoided to prevent data leakage across chromosomes.

---

## Usage

### Stage 1: Fine-tune Enformer on ChIP-seq tracks

```bash
python train_enformer.py \
    --train_bed    ./data/train_regions.bed \
    --val_bed      ./data/val_regions.bed \
    --fasta        ./data/hg38.fa \
    --bigwig_files h3k27ac.bw ctcf.bw h3k4me3.bw \
    --num_chip_tracks 3 \
    --batch_size   2 \
    --max_epochs   20 \
    --checkpoint_dir ./checkpoints
```

The best checkpoint is saved automatically based on validation loss.

### Stage 2: Build graph dataset from fine-tuned Enformer

```python
from models.enformer import EnformerFineTunerPL
from data_pipeline.graph_data import CreateGraphData, GraphDataset

model = EnformerFineTunerPL.load_from_checkpoint(
    "checkpoints/enformer-finetuned.ckpt"
)

data = CreateGraphData(
    interval_file="./data/train_regions.bed",
    fasta_file="./data/hg38.fa",
    cool_file="./data/hic_4096.cool",
    region_length="2M",
    model=model,
)
dataset = GraphDataset(root="./data/graph_train_4096_2M", data_list=data())
```

### Stage 2: Train the GNN

```bash
python train_gnn.py \
    --data_dir       ./data/graph_train_4096_2M \
    --in_feats       256 \
    --num_bins       512 \
    --hidden_dim     128 \
    --heads          4 \
    --max_epochs     100 \
    --checkpoint_dir ./checkpoints
```

`--num_bins` should equal `region_length / resolution` (e.g., 2 Mb / 4 kb = 512). `--in_feats` should equal the flattened node feature size output by Stage 1.

---

## Project Structure

```
models/
├── enformer.py          # EnformerFinetune, EnformerFineTunerPL, ChIPPredictionModel
└── gnn.py               # DNA3D_GAT, DNA3DInteractionModel, DNA3DInteractionLinear

data_pipeline/
├── data.py              # FastaInterval, EnformerSignalDataset
└── graph_data.py        # EpiConcatMatrix, CreateGraphData, GraphDataset

train_enformer.py        # Stage 1 training script
train_gnn.py             # Stage 2 training script

1_finetuning_enformer.ipynb
finally_graph_neural_network.ipynb
```

---

## Citation

If you use Epi3D in your research, please also cite the upstream tools this work builds on:

- Avsec et al. (2021) *Effective gene expression prediction from sequence by integrating long-range interactions.* **Nature Methods.** (Enformer)
- Veličković et al. (2018) *Graph Attention Networks.* **ICLR.** (GAT)

---

## License

This project is intended for academic research use only.
