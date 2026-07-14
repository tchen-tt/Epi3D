# Epi3D

**Predicting 3D genome organization from DNA sequence via epigenomic intermediates.**

Epi3D is a two-stage deep learning framework that takes raw DNA sequence as input and outputs intra-chromosomal Hi-C contact matrices — no experimental 3D data required at inference time. The pipeline was developed and validated on real ChIP-seq and Hi-C sequencing data from human cells.

---

## Motivation

The three-dimensional organization of chromatin — how DNA folds and which genomic loci come into spatial contact — plays a central role in gene regulation. Hi-C experiments can measure these contacts genome-wide, but they are expensive and cell-type-specific. Computational prediction from sequence offers a scalable alternative and a way to interrogate the regulatory logic encoded in DNA.

Epi3D bridges sequence and structure through an intermediate layer of epigenomic signals (histone modifications, chromatin accessibility, CTCF binding), which are known to be strong predictors of 3D genome topology. Rather than attempting a direct sequence-to-contact prediction, we decompose the problem into two tractable stages.

---

## Architecture

### Stage 1 — Epigenomic feature extraction

A pretrained [Enformer](https://github.com/lucidrains/enformer-pytorch) model is fine-tuned to predict ChIP-seq tracks from DNA sequence. The Enformer backbone (196,608 bp receptive field, transformer architecture) is kept frozen; only a lightweight output head is trained. This leverages Enformer's pretrained long-range sequence representations while keeping compute requirements low.

```
DNA sequence (196,608 bp, one-hot encoded)
            ↓
  Enformer backbone  [frozen]
            ↓
  Linear(3072 → num_chip_tracks) + Softplus
            ↓
  ChIP-seq tracks  [896 bins × num_chip_tracks]
```

Tracks used in our experiments: H3K27ac, H3K4me1, H3K4me3, CTCF, ATAC-seq (5 tracks total).

### Stage 2 — Graph-based contact prediction

Predicted epigenomic feature vectors (one per 4 kb genomic bin) become node features in a graph. A band-diagonal k-nearest-neighbor graph encodes the linear chromosome topology. A Graph Attention Network (GAT) propagates information between neighboring bins, and a distance-aware MLP predicts the contact frequency for each node pair.

```
ChIP-seq feature vectors  (one node per 4 kb bin)
            ↓
  Band-diagonal k-nearest-neighbor graph
            ↓
  Positional embeddings  +  2× GATConv layers
            ↓
  Distance-aware MLP edge predictor  (Softplus output)
            ↓
  Symmetric N×N Hi-C contact matrix
```

A distance-decay regularization term penalizes high predicted contacts at large genomic distances, reflecting the well-established polymer physics of chromatin.

### Model variants

| Class | Description |
|---|---|
| `DNA3D_GAT` | Primary model: 2-layer GAT, positional embeddings, distance-decay regularization |
| `DNA3DIntteractionModel` | Multi-layer GAT with relative positional embeddings; full dense N×N prediction |
| `DNA3DInteractionLinear` | Linear baseline: pairwise MLP without message passing |

---

## Experimental setup

| | Detail |
|---|---|
| Reference genome | GRCh38/hg38 |
| Hi-C resolution | 4,096 bp/bin |
| Region size | 2 Mb → 512 bins per graph |
| ChIP-seq tracks | H3K27ac, H3K4me1, H3K4me3, CTCF, ATAC (5 tracks) |
| Node feature dim | 160 (32 Enformer bins/node × 5 tracks) |
| Training graphs | 8,433 regions |
| Train chromosomes | chr3–7, 9–17, 19, 21, 22, X |
| Validation chromosomes | chr2, 18, 20 |
| Test chromosomes | chr1, 8 |

Chromosome-level splits prevent any sequence overlap between train, val, and test sets.

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torch_geometric`, `pytorch_lightning`, `enformer_pytorch`, `cooler`, `kipoiseq`, `pyfaidx`, `pyBigWig`.

---

## Data preparation

| File | Description |
|---|---|
| `./data/hg38.fa` | Human reference genome (GRCh38) in FASTA format |
| `*.cool` | Hi-C contact matrices at target resolution, [cooler](https://cooler.readthedocs.io) format |
| `*.bw` | ChIP-seq signal tracks in BigWig format (one file per mark) |
| `*.bed` | Genomic interval lists for training and validation regions |

BED intervals should match the chosen `region_length` (e.g., 2 Mb). Use chromosome-level splits rather than random splits to avoid data leakage.

---

## Usage

### Stage 1: Fine-tune Enformer on ChIP-seq tracks

```bash
python train_enformer.py \
    --train_bed      ./data/train_regions.bed \
    --val_bed        ./data/val_regions.bed \
    --fasta          ./data/hg38.fa \
    --bigwig_files   h3k27ac.bw h3k4me1.bw h3k4me3.bw ctcf.bw atac.bw \
    --num_chip_tracks 5 \
    --batch_size     2 \
    --max_epochs     20 \
    --checkpoint_dir ./checkpoints
```

The best checkpoint is saved automatically based on validation loss. Per-track Pearson correlation is logged each epoch.

### Stage 2: Build graph dataset

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

The processed dataset is saved to disk and reused on subsequent runs.

### Stage 2: Train the GNN

```bash
python train_gnn.py \
    --train_dir      ./data/graph_train_4096_2M \
    --val_dir        ./data/graph_val_4096_2M \
    --in_feats       160 \
    --num_bins       512 \
    --hidden_dim     128 \
    --heads          4 \
    --loss_fn        poisson \
    --max_epochs     100 \
    --checkpoint_dir ./checkpoints
```

`--num_bins` = `region_length / resolution` (2 Mb / 4 kb = 512).  
`--in_feats` = bins per node × num tracks (32 × 5 = 160).  
If `--val_dir` is omitted, the last 10% of training data is used as validation.

---

## Project structure

```
models/
├── enformer.py        # EnformerFinetune, EnformerFineTunerPL, ChIPPredictionModel
└── gnn.py             # DNA3D_GAT, DNA3DIntteractionModel, DNA3DInteractionLinear

data_pipeline/
├── data.py            # FastaInterval, EnformerSignalDataset
└── graph_data.py      # EpiConcatMatrix, CreateGraphData, GraphDataset

train_enformer.py      # Stage 1 training script (PyTorch Lightning)
train_gnn.py           # Stage 2 training script

1_finetuning_enformer.ipynb      # Stage 1 exploration notebook
finally_graph_neural_network.ipynb  # Stage 2 exploration notebook
```

---

## Citation

If you use Epi3D in your work, please also cite the upstream models it builds on:

- Avsec et al. (2021) *Effective gene expression prediction from sequence by integrating long-range interactions.* **Nature Methods.** — Enformer
- Veličković et al. (2018) *Graph Attention Networks.* **ICLR.** — GAT

---

## License

For academic research use only.
