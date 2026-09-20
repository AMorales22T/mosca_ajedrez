import numpy as np
import chess

# Total de características booleanas del tablero:
# 64 casillas * 12 tipos de piezas = 768
# + 4 derechos de enroque + 1 turno = 773
NUM_FEATURES = 773

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Convierte el estado del tablero de python-chess en un vector de características de tamaño 773.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    
    # Piezas (0 a 767)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Color: White = True (1), Black = False (0)
            # Piece type: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
            # Offset: White pieces: 0..5, Black pieces: 6..11
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = piece.piece_type - 1 + color_offset
            
            idx = square * 12 + piece_idx
            features[idx] = 1.0
            
    # Enroques (768 a 771)
    features[768] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[769] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[770] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[771] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # Turno (772)
    features[772] = float(board.turn == chess.WHITE)
    
    return features

def map_features_to_stimuli(features_batch: np.ndarray, sensory_nodes: list[int], N: int, current: float = 5.0) -> np.ndarray:
    """
    Mapea el vector de características a la matriz de estímulos de la red completa.
    Agrupa los nodos sensoriales equitativamente entre las características.
    
    Args:
        features_batch: (Batch, NUM_FEATURES)
        sensory_nodes: Lista de índices de neuronas en la matriz.
        N: Tamaño de la matriz neuronal.
        current: Corriente inyectada por cada característica activa.
        
    Returns:
        stimuli: (Batch, N)
    """
    Batch = features_batch.shape[0]
    stimuli = np.zeros((Batch, N), dtype=np.float32)
    
    # Dividimos las neuronas sensoriales en NUM_FEATURES grupos
    nodes_per_feature = max(1, len(sensory_nodes) // NUM_FEATURES)
    
    for i in range(NUM_FEATURES):
        group = sensory_nodes[i * nodes_per_feature : (i + 1) * nodes_per_feature]
        # Multiplicar (Batch,) * current y asignarlo a las columnas del grupo
        # features_batch[:, i] tiene forma (Batch,)
        # Queremos sumarlo a stimuli[:, group]
        val = features_batch[:, i:i+1] * current  # (Batch, 1)
        stimuli[:, group] = val
        
    return stimuli
