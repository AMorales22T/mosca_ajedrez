import chess
import chess.pgn
import pandas as pd
import io

def generate_mock_games(num_games=10):
    """
    Genera juegos aleatorios para pruebas del pipeline.
    """
    games = []
    for _ in range(num_games):
        board = chess.Board()
        game = []
        for _ in range(20): # 20 movimientos aleatorios
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            # Pick first legal move for mock
            move = legal_moves[0]
            game.append((board.copy(), move))
            board.push(move)
        games.append(game)
    return games

def get_positions_from_puzzles(csv_path: str, limit: int = 1000):
    """
    Lee puzzles desde el CSV de Lichess.
    El formato de lichess puzzles es: PuzzleId, FEN, Moves, Rating, ...
    """
    try:
        df = pd.read_csv(csv_path, nrows=limit)
        boards = []
        for fen in df['FEN']:
            boards.append(chess.Board(fen))
        return boards
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Returning mock positions.")
        return [board for game in generate_mock_games(limit // 20) for board in game]
