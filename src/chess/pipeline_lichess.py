import requests
import zstandard as zstd
import io
import chess
import chess.pgn
import os
import numpy as np

DATA_DIR = "/run/media/dolfius/Juegos/mosca_ajedrez_data"

def download_and_extract_positions(target_positions=50000, min_elo=2000):
    print(f"Descargando en streaming partidas de Lichess (Filtro Elo > {min_elo})...")
    # Usamos un mes antiguo pequeño para la prueba
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
    
    response = requests.get(url, stream=True)
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(response.raw)
    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
    
    positions_extracted = 0
    games_processed = 0
    
    dataset = []
    
    while positions_extracted < target_positions:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
            
        games_processed += 1
        
        # Filtrar Elo
        try:
            w_elo = int(game.headers.get("WhiteElo", "0"))
            b_elo = int(game.headers.get("BlackElo", "0"))
            if w_elo < min_elo or b_elo < min_elo:
                continue
        except ValueError:
            continue
            
        board = game.board()
        for move in game.mainline_moves():
            if positions_extracted >= target_positions:
                break
                
            dataset.append({
                "fen": board.fen(),
                "move": move.uci()
            })
            board.push(move)
            positions_extracted += 1
            
        if games_processed % 500 == 0:
            print(f"Partidas procesadas: {games_processed}. Posiciones: {positions_extracted}/{target_positions}")
            
    print(f"Extracción completada: {positions_extracted} posiciones.")
    
    # Dividir en Train (80%), Val (10%), Test (10%) sin mezclar dentro de las partidas (ya vienen secuenciales).
    # Como las partidas son secuenciales, cortar el array es seguro contra data leakage.
    train_idx = int(0.8 * len(dataset))
    val_idx = int(0.9 * len(dataset))
    
    train_data = dataset[:train_idx]
    val_data = dataset[train_idx:val_idx]
    test_data = dataset[val_idx:]
    
    print(f"Splits: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")
    
    # Guardar FENs y Moves para simular luego
    os.makedirs(os.path.join(DATA_DIR, "dataset"), exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = os.path.join(DATA_DIR, "dataset", f"{name}.npy")
        np.save(path, data)
        print(f"Guardado {path}")

if __name__ == "__main__":
    download_and_extract_positions(50000, 2000)
