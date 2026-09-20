import torch
import chess
import numpy as np
import scipy.sparse as sp

from src.chess.pipeline import CudaLIFEngine
from src.chess.readout import ChessReadout
from src.chess.encoding import encode_board, map_features_to_stimuli
from src.chess.viz.server import get_best_legal_move

DATA_DIR = "/run/media/dolfius/Juegos/mosca_ajedrez_data"
N = 30000
stim_nodes = list(range(1546))

print("Cargando Mosca...")
W_sub = sp.load_npz(f"{DATA_DIR}/subgraph_laptop.npz")
engine = CudaLIFEngine(W_sub, device="cuda")
model = ChessReadout(in_features=N).to("cuda")
model.load_state_dict(torch.load("checkpoints/FlyWire_Readout.pt"))
model.eval()

wins, draws, losses = 0, 0, 0
num_games = 50

print(f"Simulando {num_games} partidas rápidas: Mosca (Blancas) vs Aleatorio (Negras)...")

for i in range(num_games):
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            features = encode_board(board)
            stim_np = map_features_to_stimuli(np.array([features]), stim_nodes, N, current=20.0)
            stim_t = torch.tensor(stim_np, dtype=torch.float32, device="cuda")
            rates_t = engine.simulate_batch(stim_t, steps=50)
            
            with torch.no_grad():
                p_src, p_dst, p_promo, _ = model(rates_t)
            
            best_move, _ = get_best_legal_move(board, p_src, p_dst)
            board.push(best_move)
        else:
            board.push(np.random.choice(list(board.legal_moves)))
            
    res = board.result()
    if res == "1-0": wins += 1
    elif res == "0-1": losses += 1
    else: draws += 1
    
print(f"--- RESULTADO DE {num_games} PARTIDAS ---")
print(f"Mosca Gana: {wins}")
print(f"Oponente Gana: {losses}")
print(f"Empates: {draws}")
