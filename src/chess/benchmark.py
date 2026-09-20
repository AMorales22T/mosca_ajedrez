import time
import numpy as np
import scipy.sparse as sp
from .config import PROFILES
from .lif import LIFEngine

def make_mock_connectome(units: int, seed: int = 42):
    np.random.seed(seed)
    W = sp.random(units, units, density=0.01, format="csr", dtype=np.float32)
    W.setdiag(0)
    W.eliminate_zeros()
    return W

def run_benchmark():
    for name, profile in PROFILES.items():
        print(f"\nBenchmarking profile '{name}': {profile.max_neurons} neurons, batch_size {profile.batch_size}")
        try:
            W = make_mock_connectome(profile.max_neurons)
            engine = LIFEngine(W, tau=20.0, v_th=1.0, v_reset=0.0, dt=1.0)
            
            stimuli = np.random.rand(profile.batch_size, profile.max_neurons).astype(np.float32) * 2.0
            
            # Warmup
            engine.simulate_batch(stimuli[0:1], steps=10)
            
            # Real test
            start = time.time()
            steps = 50
            engine.simulate_batch(stimuli, steps=steps)
            elapsed = time.time() - start
            
            print(f"  Time for {steps} steps: {elapsed:.4f}s")
            print(f"  Throughput: {profile.batch_size * steps / elapsed:.2f} positions*steps / sec")
            print(f"  Positions per sec: {profile.batch_size / elapsed:.2f}")
        except MemoryError:
            print(f"  Failed: Out of Memory for profile {name}")

if __name__ == "__main__":
    run_benchmark()
