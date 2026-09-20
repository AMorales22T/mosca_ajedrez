import chess
import chess.engine
import numpy as np
import torch
from .encoding import encode_board, map_features_to_stimuli
from .lif import LIFEngine
from .readout import ChessReadout

def get_best_legal_move(board: chess.Board, src_logits, dst_logits, promo_logits):
    best_move = None
    best_score = -float('inf')
    
    for move in board.legal_moves:
        score = src_logits[move.from_square].item() + dst_logits[move.to_square].item()
        if move.promotion:
            # move.promotion is 2 (Knight) to 5 (Queen). Offset by 2 to get 0..3
            promo_idx = move.promotion - 2
            score += promo_logits[promo_idx].item()
            
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move

def play_game(lif_engine: LIFEngine, readout: ChessReadout, sensory_nodes, opponent="random", sf_engine=None):
    """
    Juega una partida completa. La mosca juega con Blancas.
    """
    board = chess.Board()
    readout.eval()
    moves_played = 0
    
    while not board.is_game_over() and moves_played < 200:
        if board.turn == chess.WHITE: # Turno de la Mosca
            feat = encode_board(board)
            stim = map_features_to_stimuli(np.array([feat]), sensory_nodes, lif_engine.N, current=5.0)
            rates = lif_engine.simulate_batch(stim, steps=50)
            
            with torch.no_grad():
                rates_t = torch.tensor(rates, dtype=torch.float32)
                src, dst, promo, val = readout(rates_t)
                
            move = get_best_legal_move(board, src[0], dst[0], promo[0])
            board.push(move)
            
        else: # Turno del Oponente
            if opponent == "random":
                move = np.random.choice(list(board.legal_moves))
                board.push(move)
            elif opponent == "stockfish" and sf_engine:
                result = sf_engine.play(board, chess.engine.Limit(time=0.01))
                board.push(result.move)
        moves_played += 1
        
    return board.result(), moves_played

