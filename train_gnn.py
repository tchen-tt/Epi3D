import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.transforms import KNNGraph
from torch_geometric.loader import DataLoader

from models.gnn import DNA3D_GAT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GNN for 3D genome interaction prediction")
    p.add_argument("--data_dir", required=True, help="Directory with .pt graph files")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--in_feats", type=int, required=True)
    p.add_argument("--num_bins", type=int, required=True)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--k_neighbors", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_graphs(data_dir: str, k: int) -> list:
    knn = KNNGraph(k=k, loop=True)
    graphs = []
    for pt_file in sorted(Path(data_dir).glob("*.pt")):
        data = torch.load(pt_file)
        num_nodes = data.x.size(0)
        data.pos = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)
        graphs.append(knn(data))
    return graphs


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.poisson_nll_loss(pred, batch.target, log_input=False, full=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = F.poisson_nll_loss(pred, batch.target, log_input=False, full=True)
        total_loss += loss.item()
    return total_loss / len(loader)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    graphs = load_graphs(args.data_dir, args.k_neighbors)
    split = int(len(graphs) * 0.9)
    train_loader = DataLoader(graphs[:split], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(graphs[split:], batch_size=args.batch_size, shuffle=False)

    model = DNA3D_GAT(
        num_bins=args.num_bins,
        in_feats=args.in_feats,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    from pathlib import Path
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            ckpt = Path(args.checkpoint_dir) / f"gnn_best_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
