import os
import time
import torch
import numpy as np
import scipy.sparse as sp
from .encoding import encode_board, map_features_to_stimuli
import chess

DATA_DIR = "/run/media/dolfius/Juegos/mosca_ajedrez_data"

class CudaLIFEngine:
    def __init__(self, W_np, device="cuda"):
        self.device = device
        self.N = W_np.shape[0]
        
        W_coo = W_np.tocoo()
        indices = torch.tensor(np.vstack((W_coo.row, W_coo.col)), dtype=torch.long)
        values = torch.tensor(W_coo.data, dtype=torch.float32)
        
        # Calibración: * 10
        self.W = torch.sparse_coo_tensor(indices, values * 10.0, size=(self.N, self.N), device=device)
        # Para hacer BxN @ NxN, usamos sparse.mm: (W @ X^T)^T
        self.W = self.W.coalesce()
        
        self.decay = float(np.exp(-1.0 / 20.0))
        self.gain = float(1.0 - self.decay)

    def simulate_batch(self, stimuli_tensor, steps=50):
        B = stimuli_tensor.size(0)
        v = torch.full((B, self.N), -52.0, dtype=torch.float32, device=self.device)
        
        # Guardamos el total de spikes en la ventana de tiempo para el Readout
        total_spikes = torch.zeros((B, self.N), dtype=torch.float32, device=self.device)
        
        # Spikes emitidos en t-1
        last_spikes = torch.zeros((B, self.N), dtype=torch.float32, device=self.device)
        
        for _ in range(steps):
            # Input de otras neuronas: last_spikes (B, N) x W (N, N)
            # mm() en sparse acepta (SparseNxN) @ (DenseNxB) -> DenseNxB
            network_input = torch.sparse.mm(self.W, last_spikes.T).T
            
            total_current = stimuli_tensor + network_input
            
            v = -52.0 + (v - -52.0) * self.decay + total_current * self.gain
            
            fired = v > -45.0
            last_spikes = fired.float()
            total_spikes += last_spikes
            v[fired] = -52.0
            
        return total_spikes / (steps * 1.0 / 1000.0) # Frecuencia en Hz

def build_cache(split_name, batch_size=256):
    print(f"\n--- Procesando {split_name} ---")
    data_path = os.path.join(DATA_DIR, "dataset", f"{split_name}.npy")
    dataset = np.load(data_path, allow_pickle=True)
    
    W_sub = sp.load_npz(os.path.join(DATA_DIR, "subgraph_laptop.npz"))
    engine = CudaLIFEngine(W_sub, device="cuda" if torch.cuda.is_available() else "cpu")
    N = engine.N
    stim_nodes = list(range(1546)) # Nodos sensoriales (visuales)
    
    out_rates = []
    out_moves = []
    
    t0 = time.time()
    
    # Pre-codificamos FENs a features en CPU (esto es rápido)
    features = []
    moves = []
    for d in dataset:
        features.append(encode_board(chess.Board(d["fen"])))
        moves.append(d["move"])
    
    features = np.array(features)
    
    # Procesar por lotes en GPU
    for i in range(0, len(features), batch_size):
        batch_f = features[i:i+batch_size]
        stim_np = map_features_to_stimuli(batch_f, stim_nodes, N, current=20.0)
        stim_t = torch.tensor(stim_np, dtype=torch.float32, device=engine.device)
        
        rates_t = engine.simulate_batch(stim_t, steps=50)
        
        out_rates.append(rates_t.cpu().numpy())
        out_moves.extend(moves[i:i+batch_size])
        
        if i > 0 and (i // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            fps = (i + batch_size) / elapsed
            print(f"Batch {i//batch_size} - Procesadas {i+batch_size}/{len(features)} posiciones ({fps:.1f} pos/s)")

    out_rates = np.vstack(out_rates)
    
    # Guardar a disco
    np.save(os.path.join(DATA_DIR, "dataset", f"{split_name}_rates.npy"), out_rates)
    np.save(os.path.join(DATA_DIR, "dataset", f"{split_name}_moves.npy"), np.array(out_moves))
    print(f"Finalizado {split_name}. Guardado en caché de disco.")

def run_step5():
    build_cache("val")
    build_cache("test")
    build_cache("train")
    print("\n¡PASO 5 COMPLETADO! Todas las cachés LIF guardadas.")

if __name__ == "__main__":
    run_step5()
