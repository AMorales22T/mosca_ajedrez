import numpy as np
import scipy.sparse as sp
from src.chess.lif import LIFEngine

def test_lif_basic():
    # 3 neurons: A -> B -> C
    # Weights
    W = sp.csr_matrix([
        [0.0, 20.0, 0.0],
        [0.0, 0.0, 20.0],
        [0.0, 0.0, 0.0]
    ])
    
    engine = LIFEngine(W, tau=10.0, v_th=1.0, v_reset=0.0, dt=1.0)
    
    # Stimulate neuron 0 (A) heavily
    stimuli = np.array([
        [2.0, 0.0, 0.0]
    ])
    
    rates = engine.simulate_batch(stimuli, steps=50)
    
    # A should fire, which fires B, which fires C
    assert rates[0, 0] > 0
    assert rates[0, 1] > 0
    assert rates[0, 2] > 0

def test_lif_batching():
    W = sp.csr_matrix(np.eye(2) * 0) # No connections
    engine = LIFEngine(W, tau=10.0, v_th=1.0, v_reset=0.0, dt=1.0)
    
    # Batch of 2
    # Item 0: stimulate N0
    # Item 1: stimulate N1
    stimuli = np.array([
        [2.0, 0.0],
        [0.0, 2.0]
    ])
    
    rates = engine.simulate_batch(stimuli, steps=20)
    
    assert rates[0, 0] > 0
    assert rates[0, 1] == 0
    assert rates[1, 0] == 0
    assert rates[1, 1] > 0

