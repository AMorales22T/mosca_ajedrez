import torch
from .readout import ChessReadout
from .controls import BaselineLinear, make_shuffled_connectome, make_random_sparse
from .config import get_profile
import scipy.sparse as sp
import numpy as np

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_controls_budget():
    profile = get_profile("tiny")
    N = profile.max_neurons
    
    # 1. Readout original sobre conectoma LIF
    original_readout = ChessReadout(in_features=N)
    target_p = parameter_count(original_readout)
    
    # 2. Baseline Linear sobre características crudas
    baseline = BaselineLinear(target_params=target_p)
    
    print(f"Parámetros Readout Conectoma: {target_p}")
    print(f"Parámetros Baseline (Características crudas): {parameter_count(baseline)}")
    
    assert target_p == parameter_count(baseline), "Los parámetros no coinciden exactamente."
    
    print("\n--- Verificando matrices de conectoma ---")
    np.random.seed(42)
    W_sub = sp.random(N, N, density=0.01, format="csr", dtype=np.float32)
    W_sub.setdiag(0)
    W_sub.eliminate_zeros()
    
    W_shuffled = make_shuffled_connectome(W_sub)
    W_random = make_random_sparse(N, density=0.01)
    
    print(f"Original - NNs: {W_sub.nnz}")
    print(f"Shuffled - NNs: {W_shuffled.nnz}")
    print(f"Random   - NNs: {W_random.nnz}")
    
    assert W_sub.nnz == W_shuffled.nnz, "Shuffled no conservó el número de sinapsis."
    print("Controles verificados con éxito.")

if __name__ == "__main__":
    check_controls_budget()
