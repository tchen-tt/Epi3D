import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import Tuple, Optional


class DNA3D_GAT(nn.Module):
    def __init__(self, num_bins: int, in_feats: int, hidden_dim: int, heads: int = 4,
                 bin_size: int = 1024, max_distance: int = 1_000_000, dropout: float = 0.1,
                 softplus_beta: float = 1.0, softplus_threshold: float = 20.0,
                 lambda_decay: float = 1e-3, use_layer_norm: bool = True,
                 use_residual: bool = True):
        super().__init__()
        self.num_bins = num_bins
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.bin_size = bin_size
        self.max_distance = max_distance
        self.lambda_decay = lambda_decay
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        self.pos_emb = nn.Embedding(num_bins, hidden_dim)
        self.input_proj = nn.Linear(in_feats, hidden_dim)

        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim * heads)
            self.layer_norm3 = nn.LayerNorm(hidden_dim)

        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                             concat=False, dropout=dropout)
        self.dropout_layer = nn.Dropout(dropout)

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(beta=softplus_beta, threshold=softplus_threshold)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def _predict_edges(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        N = x.size(0)

        idx = torch.arange(N, device=x.device)
        h = F.elu(self.input_proj(x) + self.pos_emb(idx))
        if self.use_layer_norm:
            h = self.layer_norm1(h)
        h = self.dropout_layer(h)

        h1 = F.elu(self.conv1(h, edge_index))
        if self.use_layer_norm:
            h1 = self.layer_norm2(h1)
        h1 = self.dropout_layer(h1)

        h2 = F.elu(self.conv2(h1, edge_index))
        if self.use_layer_norm:
            h2 = self.layer_norm3(h2)
        h2 = self.dropout_layer(h2)

        h = (h + h2) if self.use_residual else h2

        src, dst = edge_index
        dist_bp = ((src - dst).abs().float() * self.bin_size / self.max_distance).unsqueeze(1)
        edge_feat = torch.cat([h[src], h[dst], dist_bp], dim=-1)
        return self.edge_predictor(edge_feat).squeeze(-1)

    def _reconstruct_full_matrix(self, data: Data, intensity: torch.Tensor) -> torch.Tensor:
        N = data.x.size(0)
        full = data.x.new_zeros((N, N))
        src, dst = data.edge_index
        full[src, dst] = intensity
        return (full + full.t()) / 2

    def regularization(self, data: Data, intensity: Optional[torch.Tensor] = None) -> torch.Tensor:
        if intensity is None:
            intensity = self._predict_edges(data)
        src, dst = data.edge_index
        dist_bp = (src - dst).abs().float() * self.bin_size / self.max_distance
        return torch.mean(intensity * dist_bp)

    def predict_global_interactions(
            self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intensity = self._predict_edges(data)
        full = self._reconstruct_full_matrix(data, intensity)
        reg = self.regularization(data, intensity)
        return full, intensity, reg

    def forward(self, data: Data) -> torch.Tensor:
        full, _, _ = self.predict_global_interactions(data)
        return full

    def compute_loss(self, data: Data, target: torch.Tensor,
                     loss_type: str = "mse", alpha: float = 0.1) -> torch.Tensor:
        pred, _, reg = self.predict_global_interactions(data)
        if loss_type == "mse":
            main_loss = F.mse_loss(pred, target)
        elif loss_type == "mae":
            main_loss = F.l1_loss(pred, target)
        elif loss_type == "poisson":
            main_loss = F.poisson_nll_loss(pred, target, log_input=False)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return main_loss + alpha * self.lambda_decay * reg


class DNA3DIntteractionModel(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, num_bins: int,
                 pos_embedding_dim: int = 16, num_gnn_layers: int = 4,
                 num_final_predictor_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.relative_pos_embedding = nn.Embedding(num_bins, pos_embedding_dim)
        self.gnn_layer = nn.ModuleList(
            [GATConv(hidden_dim, hidden_dim // 4, heads=4) for _ in range(num_gnn_layers)]
        )
        self.gnn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        layers, in_dim = [], hidden_dim * 2 + pos_embedding_dim
        for _ in range(num_final_predictor_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.interaction_predictor = nn.Sequential(*layers)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index.to(data.x.device)
        N = x.size(0)

        h = self.input_proj(x)
        for layer in self.gnn_layer:
            h = self.dropout(F.relu(layer(h, edge_index)))
        h = self.gnn_norm(h)

        pos = torch.arange(N, device=x.device)
        rel_pos = torch.clamp(torch.abs(pos.unsqueeze(-1) - pos.unsqueeze(0)),
                              max=self.relative_pos_embedding.num_embeddings - 1)
        pair_feat = torch.cat([h.unsqueeze(1).expand(-1, N, -1),
                                h.unsqueeze(0).expand(N, -1, -1),
                                self.relative_pos_embedding(rel_pos)], dim=-1)
        out = self.interaction_predictor(pair_feat).squeeze(-1)
        return F.softplus((out + out.T) / 2)


class DNA3DInteractionLinear(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.output_proj = nn.Linear(hidden_dim * 2, 1)

    def forward(self, data: Data) -> torch.Tensor:
        h = self.net(data.x)
        N = h.size(0)
        pair_feat = torch.cat([h.unsqueeze(1).expand(-1, N, -1),
                                h.unsqueeze(0).expand(N, -1, -1)], dim=-1)
        out = self.output_proj(pair_feat).squeeze(-1)
        return F.softplus((out + out.T) / 2)


class Predict3D(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = DNA3DIntteractionModel(**kwargs)

    def forward(self, data) -> torch.Tensor:
        return torch.stack([self.model(data[i]) for i in range(data.num_graphs)])


class Predict3D2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = DNA3DInteractionLinear(**kwargs)

    def forward(self, data) -> torch.Tensor:
        return torch.stack([self.model(data[i]) for i in range(data.num_graphs)])
