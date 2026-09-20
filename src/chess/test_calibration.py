import numpy as np
import scipy.sparse as sp
from .encoding import encode_board, map_features_to_stimuli
import chess

def test():
    path = "/run/media/dolfius/Juegos/mosca_ajedrez_data/subgraph_laptop.npz"
    W = sp.load_npz(path)
    # Calibración: aumentar peso sináptico x10 como en el Hito 2 para mantener la red viva
    W = W.multiply(10.0)
    N = W.shape[0]
    
    b1 = chess.Board()
    b2 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    
    f1 = encode_board(b1)
    f2 = encode_board(b2)
    
    stim_nodes = list(range(1546))
    
    # Probamos una corriente base de 20.0 mV para forzar el disparo sensorial
    s1 = map_features_to_stimuli(np.array([f1]), stim_nodes, N, current=20.0)
    s2 = map_features_to_stimuli(np.array([f2]), stim_nodes, N, current=20.0)
    
    def sim(stim):
        v = np.full((1, N), -52.0, dtype=np.float32)
        spikes = np.zeros((1, N), dtype=np.int32)
        gain = 1.0 - np.exp(-1.0 / 20.0)
        
        for _ in range(50):
            v = -52.0 + (v - -52.0) * np.exp(-1.0/20.0) + (stim + spikes @ W) * gain
            fired = v > -45.0
            spikes += fired
            v[fired] = -52.0
        return spikes[0]

    r1 = sim(s1)
    r2 = sim(s2)
    
    print(f"Posición 1 (Inicial): {np.sum(r1 > 0)} neuronas activas")
    print(f"Posición 2 (1. e4)  : {np.sum(r2 > 0)} neuronas activas")
    diff = np.sum(np.abs(r1 - r2))
    print(f"Diferencia en spikes: {diff}")
    
    if diff > 10 and np.sum(r1 > 0) > 1000 and np.sum(r1 > 0) < N*0.8:
        print("Calibración SUPERADA. La red ni satura ni muere, y separa patrones.")
    else:
        print("Calibración FALLIDA.")
        
test()
