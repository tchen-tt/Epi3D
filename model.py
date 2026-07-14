from sympy.multipledispatch.conflict import edge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import Tuple, Optional, Dict, Any


class DNA3D_GAT(nn.Module):
    """
    DNA3D_GAT: A graph attention-based model for predicting 3D genome intra-chromosomal contact matrices.

    Target submission: Nature Communications
    - Supports multiple human cell lines (GM12878, K562, IMR90) and mouse.
    - Learns positional embeddings, distance-sensitive GAT message passing, and Softplus regression.

    Limitations:
    - Currently only models intra-chromosomal interactions; inter-chromosomal prediction remains future work.
    - Requires high-quality Hi-C data; performance depends on resolution and depth.

    Future directions:
    - Extend to inter-chromosomal contact prediction by incorporating chromosome territory priors.
    - Validate predicted novel loops with ChIA-PET or CRISPR assays.
    
    Args:
        num_bins (int): Number of genomic bins
        in_feats (int): Input feature dimension
        hidden_dim (int): Hidden dimension for GAT layers
        heads (int): Number of attention heads in first GAT layer
        bin_size (int): Size of each genomic bin in base pairs
        max_distance (int): Maximum genomic distance for normalization
        dropout (float): Dropout rate
        softplus_beta (float): Beta parameter for Softplus activation
        softplus_threshold (float): Threshold parameter for Softplus activation
        lambda_decay (float): Weight for distance decay regularization
        use_layer_norm (bool): Whether to use layer normalization
        use_residual (bool): Whether to use residual connections
    """
    def __init__(self,
                 num_bins: int,
                 in_feats: int,
                 hidden_dim: int,
                 heads: int = 4,
                 bin_size: int = 1024,
                 max_distance: int = 1_000_000,
                 dropout: float = 0.1,
                 softplus_beta: float = 1.0,
                 softplus_threshold: float = 20.0,
                 lambda_decay: float = 1e-3,
                 use_layer_norm: bool = True,
                 use_residual: bool = True):
        super().__init__()

        # Validate input parameters
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")
        if in_feats <= 0:
            raise ValueError("in_feats must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")
        if bin_size <= 0:
            raise ValueError("bin_size must be positive")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if lambda_decay < 0:
            raise ValueError("lambda_decay must be non-negative")

        # Store parameters
        self.num_bins = num_bins
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.bin_size = bin_size
        self.max_distance = max_distance
        self.dropout = dropout
        self.softplus_beta = softplus_beta
        self.softplus_threshold = softplus_threshold
        self.lambda_decay = lambda_decay
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        # Positional embeddings
        self.pos_emb = nn.Embedding(num_bins, hidden_dim)
        
        # Input feature projection
        self.input_proj = nn.Linear(in_feats, hidden_dim)
        
        # Layer normalization (optional)
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim * heads)
            self.layer_norm3 = nn.LayerNorm(hidden_dim)

        # GAT layers
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Edge interaction strength predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(beta=softplus_beta, threshold=softplus_threshold)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass: returns global contact matrix (N×N) representing interaction strength between each bin pair.
        
        Args:
            data (Data): PyTorch Geometric Data object with:
                - x: Node features [N, in_feats]
                - edge_index: Edge indices [2, E]
        
        Returns:
            torch.Tensor: Global contact matrix [N, N]
        """
        full, _, _ = self.predict_global_interactions(data)
        return full

    def _predict_edges(self, data: Data) -> torch.Tensor:
        """
        Predict interaction strengths for edges in the graph.
        
        Args:
            data (Data): PyTorch Geometric Data object
            
        Returns:
            torch.Tensor: Edge interaction strengths [E]
        """
        x, edge_index = data.x, data.edge_index
        N = x.size(0)
        
        # Validate input dimensions
        if x.size(1) != self.in_feats:
            raise ValueError(f"Expected input features dimension {self.in_feats}, got {x.size(1)}")
        if N > self.num_bins:
            warnings.warn(f"Number of nodes ({N}) exceeds num_bins ({self.num_bins})")
        
        # Positional encoding + input projection
        idx = torch.arange(N, device=x.device)
        h = self.input_proj(x) + self.pos_emb(idx)
        h = nn.ELU()(h)
        
        if self.use_layer_norm:
            h = self.layer_norm1(h)
        h = self.dropout_layer(h)
        
        # First GAT layer
        h_conv1 = self.conv1(h, edge_index)
        h_conv1 = nn.ELU()(h_conv1)
        
        if self.use_layer_norm:
            h_conv1 = self.layer_norm2(h_conv1)
        h_conv1 = self.dropout_layer(h_conv1)
        
        # Second GAT layer
        h_conv2 = self.conv2(h_conv1, edge_index)
        h_conv2 = nn.ELU()(h_conv2)
        
        if self.use_layer_norm:
            h_conv2 = self.layer_norm3(h_conv2)
        h_conv2 = self.dropout_layer(h_conv2)
        
        # Residual connection (optional)
        if self.use_residual:
            h = h + h_conv2
        else:
            h = h_conv2
        
        # Concatenate node pairs + distance features
        src, dst = edge_index
        h_i, h_j = h[src], h[dst]
        
        # Calculate genomic distance in base pairs
        dist_bp = ((src - dst).abs().float() * self.bin_size) / self.max_distance
        print(src, dst)
        print(dist_bp)
        dist_bp = dist_bp.unsqueeze(1)
        
        # Edge features: [h_i, h_j, distance]
        edge_feat = torch.cat([h_i, h_j, dist_bp], dim=-1)
        
        return self.edge_predictor(edge_feat).squeeze(-1)  # [E]

    def _reconstruct_full_matrix(self, data: Data, intensity: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct full contact matrix from sparse edge predictions.
        
        Args:
            data (Data): PyTorch Geometric Data object
            intensity (torch.Tensor): Edge interaction strengths [E]
            
        Returns:
            torch.Tensor: Full contact matrix [N, N]
        """
        x, edge_index = data.x, data.edge_index
        N = x.size(0)
        
        # Initialize full matrix
        full = x.new_zeros((N, N))
        src, dst = edge_index
        
        # Fill in predicted interactions
        full[src, dst] = intensity
        
        # Symmetrize matrix (contacts are symmetric)
        full = (full + full.t()) / 2
        
        return full

    def regularization(self, data: Data, intensity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Distance decay regularization: encourages lower interaction strength for distant bins.
        
        Args:
            data (Data): PyTorch Geometric Data object
            intensity (torch.Tensor, optional): Pre-computed edge intensities
            
        Returns:
            torch.Tensor: Regularization loss
        """
        if intensity is None:
            intensity = self._predict_edges(data)
        
        src, dst = data.edge_index
        dist_bp = ((src - dst).abs().float() * self.bin_size) / self.max_distance
        
        return torch.mean(intensity * dist_bp)

    def predict_global_interactions(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end prediction: returns global contact matrix, sparse intensities, and regularization.
        
        Args:
            data (Data): PyTorch Geometric Data object
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Full contact matrix [N, N]
                - Sparse edge intensities [E]
                - Regularization loss
        """
        intensity = self._predict_edges(data)
        full = self._reconstruct_full_matrix(data, intensity)
        reg = self.regularization(data, intensity)
        return full, intensity, reg

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model configuration and parameter information.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "DNA3D_GAT",
            "num_bins": self.num_bins,
            "in_feats": self.in_feats,
            "hidden_dim": self.hidden_dim,
            "heads": self.heads,
            "bin_size": self.bin_size,
            "max_distance": self.max_distance,
            "dropout": self.dropout,
            "lambda_decay": self.lambda_decay,
            "use_layer_norm": self.use_layer_norm,
            "use_residual": self.use_residual,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }

    def compute_loss(self, 
                    data: Data, 
                    target: torch.Tensor,
                    loss_type: str = "mse",
                    alpha: float = 0.1) -> torch.Tensor:
        """
        Compute loss for training.
        
        Args:
            data (Data): Input data
            target (torch.Tensor): Target contact matrix [N, N]
            loss_type (str): Loss type ("mse", "mae", "poisson")
            alpha (float): Weight for regularization loss
            
        Returns:
            torch.Tensor: Total loss
        """
        pred, intensity, reg = self.predict_global_interactions(data)
        
        # Main prediction loss
        if loss_type == "mse":
            main_loss = F.mse_loss(pred, target)
        elif loss_type == "mae":
            main_loss = F.l1_loss(pred, target)
        elif loss_type == "poisson":
            # Poisson loss for count data
            main_loss = F.poisson_nll_loss(pred, target, log_input=False)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Regularization loss
        reg_loss = self.lambda_decay * reg
        
        return main_loss + alpha * reg_loss


class DNA3DIntteractionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        node_feature_dim = kwargs.get("node_feature_dim")
        hidden_dim = kwargs.get("hidden_dim")
        pos_embedding_dim = kwargs.get("pos_embedding_dim", 16)
        num_gnn_layers = kwargs.get("num_gnn_layers", 4)
        num_final_predictor_layers = kwargs.get("num_final_predictor_layers", 2)
        dropout = kwargs.get("dropout", 0.1)
        num_bins = kwargs.get("num_bins")

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.relative_pos_embedding = nn.Embedding(num_bins, pos_embedding_dim)
        self.gnn_layer = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layer.append(GATConv(hidden_dim, hidden_dim // 4, heads=4) )
        self.gnn_norm = nn.LayerNorm(hidden_dim)

        predictor_input_dim = hidden_dim * 2 + pos_embedding_dim
        predictor_layers = []

        for _ in range(num_final_predictor_layers):
            predictor_layers.extend([
                nn.Linear(predictor_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            predictor_input_dim = hidden_dim
        predictor_layers.append(nn.Linear(hidden_dim, 1))
        self.interaction_predictor = nn.Sequential(*predictor_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass: returns interaction strengths for all edges in the graph.
        
        Args:
            data (Data): PyTorch Geometric Data object with:
                - x: Node features [N, in_feats]
                - edge_index: Edge indices [2, E]
        
        Returns:
            torch.Tensor: Interaction strengths for all nodes [N, N]
        """
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(x.device)
        num_nodes = x.size(0)
         
        h = self.input_proj(x)
        h_gnn = h
        for gnn_layer in self.gnn_layer:
            h_gnn = gnn_layer(h_gnn, edge_index)
            h_gnn = F.relu(h_gnn)
            h_gnn = self.dropout(h_gnn)
        h_gnn = self.gnn_norm(h_gnn)

       
        h_i = h_gnn.unsqueeze(1).expand(-1, num_nodes, -1)
        h_j = h_gnn.unsqueeze(0).expand(num_nodes, -1, -1)
        
        pos = torch.arange(num_nodes, device=x.device)
        relative_pos = torch.abs(pos.unsqueeze(-1) - pos.unsqueeze(0))
        relative_pos = torch.clamp(relative_pos, max=self.relative_pos_embedding.num_embeddings - 1)
        pos_embed = self.relative_pos_embedding(relative_pos)


        pair_feature = torch.cat([h_i, h_j, pos_embed], dim=-1)

        interaction_logits = self.interaction_predictor(pair_feature).squeeze(-1)
        interaction_logits = (interaction_logits + interaction_logits.T) / 2
        interaction_logits = F.softplus(interaction_logits)

        return interaction_logits


class DNA3DInteractionLinear(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        node_feature_dim = kwargs.get("node_feature_dim")
        hidden_dim = kwargs.get("hidden_dim")
        dropout = kwargs.get("dropout", 0.1)
        
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.input_project2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.to(x.device)
        num_nodes = x.size(0)
        
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout1(h)
        h = self.input_project2(h)
        h = F.relu(h)
        h = self.dropout2(h)

        h_i = h.unsqueeze(1).expand(-1, num_nodes, -1)
        h_j = h.unsqueeze(0).expand(num_nodes, -1, -1)

        pair_feature = torch.cat([h_i, h_j], dim=-1)
        interaction_logits = self.output_proj(pair_feature).squeeze(-1)
        interaction_logits = (interaction_logits + interaction_logits.T) / 2
        interaction_logits = F.softplus(interaction_logits)
        return interaction_logits


class Predict3D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DNA3DIntteractionModel(**kwargs)

    def forward(self, data: Data) -> torch.Tensor:
        num_graph = data.num_graphs

        for idx in range(num_graph):
            data_idx = data[idx]
            x = self.model(data_idx)
            # x = x.view(1, -1)

            if idx == 0:    
                full_matrix = x
            else:
                full_matrix = torch.cat([full_matrix, x], dim=0)

        return full_matrix


class Predict3D2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DNA3DInteractionLinear(**kwargs)

    def forward(self, data: Data) -> torch.Tensor:
        num_graph = data.num_graphs

        for idx in range(num_graph):
            data_idx = data[idx]
            x = self.model(data_idx)
            # x = x.view(1, -1)

            if idx == 0:    
                full_matrix = x
            else:
                full_matrix = torch.cat([full_matrix, x], dim=0)

        return full_matrix
    
   