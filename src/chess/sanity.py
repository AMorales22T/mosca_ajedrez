import time
import numpy as np
import scipy.sparse as sp
from .config import get_profile
from .lif import LIFEngine

def make_mock_connectome(units: int, seed: int = 42):
    np.random.seed(seed)
    W = sp.random(units, units, density=0.01, format="csr", dtype=np.float32)
    W.setdiag(0)
    W.eliminate_zeros()
    return W

def sanity_feeding_test():
    profile = get_profile("tiny")
    units = profile.max_neurons
    
    W = make_mock_connectome(units)
    # mock feeding: sugar sensors (0..9) -> extend proboscis (10..19)
    # bitter sensors (20..29) -> retract (30..39)
    for i in range(10):
        W[i, 10 + i] = 50.0
        W[20 + i, 30 + i] = 50.0
    
    engine = LIFEngine(W, tau=20.0, v_th=1.0, v_reset=0.0, dt=1.0)
    
    stimuli = np.zeros((2, units), dtype=np.float32)
    stimuli[0, 0:10] = 5.0
    stimuli[1, 20:30] = 5.0
    
    start = time.time()
    rates = engine.simulate_batch(stimuli, steps=100)
    elapsed = time.time() - start
    
    print(f"Sanity test on 'tiny' profile ({units} neurons) took {elapsed:.4f}s")
    
    sugar_extend = rates[0, 10:20].mean()
    sugar_retract = rates[0, 30:40].mean()
    bitter_extend = rates[1, 10:20].mean()
    bitter_retract = rates[1, 30:40].mean()
    
    print(f"Sugar input -> extend rate: {sugar_extend:.3f}, retract rate: {sugar_retract:.3f}")
    print(f"Bitter input -> extend rate: {bitter_extend:.3f}, retract rate: {bitter_retract:.3f}")
    
    assert sugar_extend > 0 and sugar_retract == 0
    assert bitter_retract > 0 and bitter_extend == 0
    print("Sanity checks passed.")

if __name__ == "__main__":
    sanity_feeding_test()
