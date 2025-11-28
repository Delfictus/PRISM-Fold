"""
Multi-Task GATv2 for Graph Coloring

Architecture:
- 6-layer GATv2 backbone
- 4 task-specific heads:
  1. Node color prediction (classification, 50% loss weight)
  2. Chromatic number (regression, 25%)
  3. Graph type (classification, 15%)
  4. Difficulty score (regression, 10%)

Optimized for H100 GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class MultiTaskGATv2(nn.Module):
    """Multi-task GATv2 for graph coloring"""

    def __init__(
        self,
        node_feature_dim: int = 16,
        hidden_dim: int = 256,
        num_gnn_layers: int = 6,
        num_attention_heads: int = 8,
        max_colors: int = 200,
        num_graph_types: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_heads = num_attention_heads
        self.max_colors = max_colors
        self.num_graph_types = num_graph_types
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GATv2 backbone
        self.gat_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim // num_attention_heads

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_attention_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=None,
                )
            )

        # Layer normalization for each GATv2 layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # Task-specific heads

        # 1. Node color prediction (per-node classification)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_colors)
        )

        # 2. Chromatic number prediction (graph-level regression)
        self.chromatic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 3. Graph type classification (graph-level)
        self.graph_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_graph_types)
        )

        # 4. Difficulty score prediction (graph-level regression)
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data: Batch):
        """
        Forward pass

        Args:
            data: PyG Batch object with:
                - x: [total_nodes, node_feature_dim]
                - edge_index: [2, total_edges]
                - batch: [total_nodes] batch assignment

        Returns:
            dict with predictions for each task
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        h = self.input_proj(x)  # [N, hidden_dim]

        # GATv2 layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = gat(h, edge_index)  # [N, hidden_dim]
            h_new = F.elu(h_new)
            h_new = norm(h_new)

            # Residual connection
            h = h + h_new

            h = F.dropout(h, p=self.dropout, training=self.training)

        # Node-level features: [N, hidden_dim]
        node_embeddings = h

        # Task 1: Node color prediction (per-node)
        color_logits = self.color_head(node_embeddings)  # [N, max_colors]

        # Graph-level pooling for graph-level tasks
        graph_mean = global_mean_pool(node_embeddings, batch)  # [B, hidden_dim]
        graph_max = global_max_pool(node_embeddings, batch)    # [B, hidden_dim]
        graph_repr = torch.cat([graph_mean, graph_max], dim=-1)  # [B, hidden_dim*2]

        # Task 2: Chromatic number prediction
        chromatic_pred = self.chromatic_head(graph_repr).squeeze(-1)  # [B]

        # Task 3: Graph type classification
        graph_type_logits = self.graph_type_head(graph_repr)  # [B, num_types]

        # Task 4: Difficulty score prediction
        difficulty_pred = self.difficulty_head(graph_repr).squeeze(-1)  # [B]

        return {
            'color_logits': color_logits,       # [N, max_colors]
            'chromatic': chromatic_pred,         # [B]
            'graph_type_logits': graph_type_logits,  # [B, num_types]
            'difficulty': difficulty_pred,       # [B]
        }


class MultiTaskLoss(nn.Module):
    """Multi-task loss with configurable weights"""

    def __init__(
        self,
        color_weight: float = 0.5,
        chromatic_weight: float = 0.25,
        graph_type_weight: float = 0.15,
        difficulty_weight: float = 0.1,
    ):
        super().__init__()
        self.color_weight = color_weight
        self.chromatic_weight = chromatic_weight
        self.graph_type_weight = graph_type_weight
        self.difficulty_weight = difficulty_weight

        # Loss functions
        self.color_loss_fn = nn.CrossEntropyLoss()
        self.chromatic_loss_fn = nn.L1Loss()  # MAE for chromatic number
        self.graph_type_loss_fn = nn.CrossEntropyLoss()
        self.difficulty_loss_fn = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        Compute multi-task loss

        Args:
            predictions: dict from model forward pass
            targets: dict with ground truth labels

        Returns:
            total_loss, dict of individual losses
        """
        # Task 1: Node color prediction
        color_loss = self.color_loss_fn(
            predictions['color_logits'],
            targets['y_colors']
        )

        # Task 2: Chromatic number (regression)
        chromatic_loss = self.chromatic_loss_fn(
            predictions['chromatic'],
            targets['y_chromatic']
        )

        # Task 3: Graph type classification
        # Use squeeze(-1) to only remove the last dimension if it's size 1
        # This prevents issues when batch_size=1
        graph_type_loss = self.graph_type_loss_fn(
            predictions['graph_type_logits'],
            targets['y_graph_type'].squeeze(-1) if targets['y_graph_type'].dim() > 1 else targets['y_graph_type']
        )

        # Task 4: Difficulty score (regression)
        difficulty_loss = self.difficulty_loss_fn(
            predictions['difficulty'],
            targets['y_difficulty']
        )

        # Weighted sum
        total_loss = (
            self.color_weight * color_loss +
            self.chromatic_weight * chromatic_loss +
            self.graph_type_weight * graph_type_loss +
            self.difficulty_weight * difficulty_loss
        )

        losses = {
            'total': total_loss.item(),
            'color': color_loss.item(),
            'chromatic': chromatic_loss.item(),
            'graph_type': graph_type_loss.item(),
            'difficulty': difficulty_loss.item(),
        }

        return total_loss, losses


if __name__ == '__main__':
    # Test model architecture
    print("Testing Multi-Task GATv2 model...")

    model = MultiTaskGATv2(
        node_feature_dim=16,
        hidden_dim=256,
        num_gnn_layers=6,
        num_attention_heads=8,
        max_colors=200,
        num_graph_types=8,
        dropout=0.2,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with dummy data
    from torch_geometric.data import Batch, Data

    # Create dummy graph
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data, data])

    model.eval()
    with torch.no_grad():
        output = model(batch)

    print(f"\nOutput shapes:")
    print(f"  Color logits: {output['color_logits'].shape}")
    print(f"  Chromatic: {output['chromatic'].shape}")
    print(f"  Graph type logits: {output['graph_type_logits'].shape}")
    print(f"  Difficulty: {output['difficulty'].shape}")

    print("\n✅ Model architecture test successful!")
