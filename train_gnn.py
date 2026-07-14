import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from data_pipeline.graph_data import GraphDataset
from models.gnn import DNA3D_GAT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GNN for 3D genome interaction prediction")
    p.add_argument("--train_dir", required=True, help="Root directory of training GraphDataset")
    p.add_argument("--val_dir", default=None,
                   help="Root directory of validation GraphDataset; if omitted, 10%% of train data is used")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--in_feats", type=int, required=True)
    p.add_argument("--num_bins", type=int, required=True)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--loss_fn", default="poisson", choices=["poisson", "mse", "mae"])
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Weight for distance-decay regularization")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def train_epoch(model, loader, optimizer, device, loss_fn, alpha):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(batch, batch.target, loss_type=loss_fn, alpha=alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device, loss_fn, alpha):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        loss = model.compute_loss(batch, batch.target, loss_type=loss_fn, alpha=alpha)
        total_loss += loss.item()
    return total_loss / len(loader)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_ds = GraphDataset(root=args.train_dir)
    if args.val_dir is not None:
        val_ds = GraphDataset(root=args.val_dir)
    else:
        split = int(len(train_ds) * 0.9)
        train_ds, val_ds = train_ds[:split], train_ds[split:]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DNA3D_GAT(
        num_bins=args.num_bins,
        in_feats=args.in_feats,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.loss_fn, args.alpha)
        val_loss = eval_epoch(model, val_loader, device, args.loss_fn, args.alpha)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            ckpt = Path(args.checkpoint_dir) / f"gnn_best_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
