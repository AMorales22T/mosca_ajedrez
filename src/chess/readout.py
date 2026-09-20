import torch
from torch import nn

class ChessReadout(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        # Heads
        self.head_src = nn.Linear(hidden, 64)
        self.head_dst = nn.Linear(hidden, 64)
        self.head_promo = nn.Linear(hidden, 4)
        self.head_val = nn.Linear(hidden, 1) # Optional value head
        
    def forward(self, x):
        """
        x: (Batch, in_features)
        Returns:
            src: (Batch, 64)
            dst: (Batch, 64)
            promo: (Batch, 4)
            val: (Batch, 1)
        """
        h = self.mlp(x)
        return self.head_src(h), self.head_dst(h), self.head_promo(h), torch.tanh(self.head_val(h))
