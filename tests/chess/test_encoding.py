import chess
import numpy as np
from src.chess.encoding import encode_board, map_features_to_stimuli, NUM_FEATURES

def test_encode_board():
    board = chess.Board() # starting pos
    features = encode_board(board)
    
    assert len(features) == NUM_FEATURES
    
    # Check turn
    assert features[772] == 1.0 # White's turn
    
    # Check castling rights
    assert features[768] == 1.0
    assert features[771] == 1.0
    
    # Check e2 pawn (White Pawn)
    # e2 is square 12. White Pawn is offset 0. 12 * 12 + 0 = 144
    assert features[144] == 1.0
    
def test_map_features():
    features = np.zeros((2, NUM_FEATURES), dtype=np.float32)
    features[0, 0] = 1.0
    
    sensory_nodes = list(range(NUM_FEATURES * 2)) # 2 nodes per feature
    
    stimuli = map_features_to_stimuli(features, sensory_nodes, N=2000, current=3.0)
    
    # Shape should be (2, 2000)
    assert stimuli.shape == (2, 2000)
    
    # First batch, feature 0 (nodes 0, 1) should have 3.0
    assert stimuli[0, 0] == 3.0
    assert stimuli[0, 1] == 3.0
    # Other nodes should be 0
    assert stimuli[0, 2] == 0.0
    
    # Second batch should be all 0
    assert stimuli[1].sum() == 0.0

