import time
import os
import numpy as np
import scipy.sparse as sp

# Para LIFEngine usaremos una simplificación que no dependa de flypoke entero para ir rápido
class FastLIFEngine:
    def __init__(self, W, N):
        self.W = W
        self.N = N
        self.tau_m = 20.0
        self.v_th = -45.0
        self.v_rest = -52.0
        self.v_reset = -52.0
        self.dt = 1.0
        self.decay = np.exp(-self.dt / self.tau_m)
        self.gain = 1.0 - self.decay

    def simulate_batch(self, stimuli, steps=50):
        B = stimuli.shape[0]
        v = np.full((B, self.N), self.v_rest, dtype=np.float32)
        spikes_sum = np.zeros((B, self.N), dtype=np.int32)
        
        for _ in range(steps):
            # En matrix mul BxN @ NxN (sparse)
            input_current = stimuli + spikes_sum @ self.W
            v = self.v_rest + (v - self.v_rest) * self.decay + input_current * self.gain
            fired = v > self.v_th
            spikes_sum += fired
            v[fired] = self.v_reset
            
        return spikes_sum / (steps * self.dt / 1000.0)

def run_bench():
    path = "/run/media/dolfius/Juegos/mosca_ajedrez_data/subgraph_laptop.npz"
    if not os.path.exists(path):
        return
        
    W = sp.load_npz(path)
    N = W.shape[0]
    engine = FastLIFEngine(W, N)
    
    B = 128
    stimuli = np.random.randn(B, N).astype(np.float32) * 5.0
    
    # Warmup
    _ = engine.simulate_batch(stimuli, steps=5)
    
    print("Iniciando benchmark (Batch 128, 50 steps)...")
    t0 = time.time()
    _ = engine.simulate_batch(stimuli, steps=50)
    t1 = time.time()
    
    time_per_batch = t1 - t0
    total_positions = 50000
    total_batches = total_positions / B
    total_time_sec = total_batches * time_per_batch
    
    print(f"Tiempo por lote (128 pos): {time_per_batch:.2f} s")
    print(f"Tiempo estimado para {total_positions} posiciones: {total_time_sec / 60:.1f} minutos ({total_time_sec:.0f} s)")

if __name__ == "__main__":
    run_bench()
