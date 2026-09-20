import numpy as np
import scipy.sparse as sp
import torch
from .readout import ChessReadout
from .encoding import NUM_FEATURES

def make_shuffled_connectome(W: sp.csr_matrix, seed: int = 42) -> sp.csr_matrix:
    np.random.seed(seed)
    P1 = np.random.permutation(W.shape[0])
    P2 = np.random.permutation(W.shape[1])
    return W[P1, :][:, P2]

def make_random_sparse(N: int, density: float, seed: int = 42) -> sp.csr_matrix:
    np.random.seed(seed)
    W = sp.random(N, N, density=density, format="csr", dtype=np.float32)
    W.setdiag(0)
    W.eliminate_zeros()
    return W

class BaselineLinear(torch.nn.Module):
    def __init__(self, target_params: int, hidden=128):
        super().__init__()
        self.readout = ChessReadout(in_features=NUM_FEATURES, hidden=hidden)
        
        current_params = sum(p.numel() for p in self.parameters())
        if current_params < target_params:
            self.budget_pad = torch.nn.Parameter(torch.zeros(target_params - current_params))
            
    def forward(self, board_features):
        src, dst, promo, val = self.readout(board_features)
        if hasattr(self, "budget_pad"):
            # Ensure it receives gradients
            src = src + self.budget_pad.sum() * 0.0
        return src, dst, promo, val
