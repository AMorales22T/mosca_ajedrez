import os
import time
import torch
import numpy as np
import scipy.sparse as sp
from typing import List, Dict

DATA_DIR = "/run/media/dolfius/Juegos/mosca_ajedrez_data"

class CudaLIFEngine:
    def __init__(self, W_np, device="cuda"):
        self.device = device
        self.N = W_np.shape[0]
        
        print(f"Subiendo matriz dispersa de {self.N}x{self.N} a la {device}...")
        W_coo = W_np.tocoo()
        indices = torch.tensor(np.vstack((W_coo.row, W_coo.col)), dtype=torch.long)
        values = torch.tensor(W_coo.data, dtype=torch.float32)
        
        # Guardamos la matriz pesada multiplicada por 10 (Calibración Hito 4)
        self.W = torch.sparse_coo_tensor(indices, values * 10.0, size=(self.N, self.N), device=device)
        self.W = self.W.to_sparse_csr() # Más rápido para matrix mul
        
        self.tau_m = 20.0
        self.v_th = -45.0
        self.v_rest = -52.0
        self.v_reset = -52.0
        self.dt = 1.0
        self.decay = float(np.exp(-self.dt / self.tau_m))
        self.gain = float(1.0 - self.decay)

    def simulate_batch(self, stimuli_tensor, steps=50):
        B = stimuli_tensor.size(0)
        v = torch.full((B, self.N), self.v_rest, dtype=torch.float32, device=self.device)
        spikes_sum = torch.zeros((B, self.N), dtype=torch.float32, device=self.device)
        
        for _ in range(steps):
            # En torch: (B, N) x (N, N) sparse. A veces se escribe como W.matmul(spikes.T).T
            # Pero para sparse_csr es W @ x. Así que hay que transponer:
            # (W @ spikes.T).T -> (N, N) @ (N, B) -> (N, B) -> (B, N)
            
            # current_spikes es spikes del frame anterior. Usamos la suma total para simplificar o 
            # necesitamos guardar los spikes exactos del step anterior? 
            # Wait, las ecuaciones LIF usan los spikes que acaban de disparar en el t-1.
            # En CPU sumábamos `spikes_sum` que es INCORRECTO matemáticamente pero nos daba un "burst".
            # Lo correcto matemáticamente:
            pass
            
# Vamos a crear el script completo de caché
