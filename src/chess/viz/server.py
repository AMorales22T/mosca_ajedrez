from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import os
import json
import numpy as np
try:
    import torch
    import scipy.sparse as sp
    import chess
    import chess.engine
    from src.chess.pipeline import CudaLIFEngine
    from src.chess.readout import ChessReadout
    from src.chess.encoding import encode_board, map_features_to_stimuli
except ImportError as exc:
    # La vista procedural puede arrancar sin el stack de entrenamiento.
    # Si está instalado, el bloque de inicialización inferior usa el modelo real.
    torch = None
    sp = None
    chess = None
    print(f"Stack neuronal no disponible; se sirve la vista demo: {exc}")

app = FastAPI()
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# --- Inicialización Global de los Modelos (Se carga 1 vez) ---
DATA_DIR = "/run/media/dolfius/Juegos/mosca_ajedrez_data"
N = 30000
stim_nodes = list(range(1546))

print("Cargando Conectoma Biológico en GPU...")
try:
    W_sub = sp.load_npz(os.path.join(DATA_DIR, "subgraph_laptop.npz"))
    engine = CudaLIFEngine(W_sub, device="cuda")
    
    model_readout = ChessReadout(in_features=N).to("cuda")
    model_readout.load_state_dict(torch.load("checkpoints/FlyWire_Readout.pt"))
    model_readout.eval()
    print("Modelos cargados. Listo para jugar.")
except Exception as e:
    print("Error cargando modelos:", e)
    engine = None

def get_best_legal_move(board, p_src, p_dst):
    best_move = None
    best_score = -999999
    
    # p_src, p_dst son logits (Batch, 64)
    # Convertir a log_softmax
    log_src = torch.nn.functional.log_softmax(p_src, dim=1).squeeze().cpu().numpy()
    log_dst = torch.nn.functional.log_softmax(p_dst, dim=1).squeeze().cpu().numpy()
    
    candidates = []
    for move in board.legal_moves:
        src = move.from_square
        dst = move.to_square
        score = log_src[src] + log_dst[dst]
        candidates.append({"move": move.uci(), "score": float(score)})
        if score > best_score:
            best_score = score
            best_move = move
            
    # Ordenar candidatos
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:5]
    
    # Normalizar scores para UI mock prob
    if candidates:
        max_s = candidates[0]["score"]
        for c in candidates:
            c["prob"] = round(np.exp(c["score"] - max_s), 3) # Pseudo probabilidad
            
    return best_move, candidates

@app.websocket("/ws/replay")
async def websocket_replay(websocket: WebSocket):
    await websocket.accept()
    if engine is None:
        # Demo reproducible: conserva la interfaz de metadatos aunque no haya GPU/conectoma.
        metadata = [{"id": i, "region": "central_brain" if i < 8 else "optic_lobe" if i < 12 else "vnc", "type": "demo_neuron"} for i in range(12)]
        try:
            for frame in range(180):
                spikes = np.flatnonzero(np.random.rand(2000) > 0.96).tolist()
                await websocket.send_text(json.dumps({
                    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                    "rates": (np.random.rand(2000) > 0.96).astype(int).tolist(),
                    "spikes": spikes, "synapses": [[int(i), int((i + 17) % 2000)] for i in spikes[:24]],
                    "neural_meta": metadata, "desc_rates": 0.0, "frame": frame,
                    "phase": "DEMO / SIN ENTRENAR · PROPAGACIÓN PROCEDURAL",
                    "action_from": 12, "action_to": 28, "candidates": [], "sf_move": "e2e4"
                }))
                await asyncio.sleep(1.0 / 30.0)
        except WebSocketDisconnect:
            pass
        return
        
    try:
        board = chess.Board()
        
        while not board.is_game_over():
            # TURNO DE LA MOSCA (Blancas)
            if board.turn == chess.WHITE:
                # 1. VISUAL
                features = encode_board(board)
                stim_np = map_features_to_stimuli(np.array([features]), stim_nodes, N, current=20.0)
                stim_t = torch.tensor(stim_np, dtype=torch.float32, device="cuda")
                
                # Para la animación, enviamos rates aleatorios temporales pero al final los reales
                rates_t = engine.simulate_batch(stim_t, steps=50)
                real_rates = rates_t[0].cpu().numpy()
                
                # 2. READOUT
                with torch.no_grad():
                    p_src, p_dst, p_promo, _ = model_readout(rates_t)
                
                best_move, candidates = get_best_legal_move(board, p_src, p_dst)
                
                # Animación por fases para la Mosca (3 segundos totales)
                for step in range(120):
                    if step < 30:
                        phase = "1. ESTIMULACIÓN VISUAL"
                        desc_rates = 0.0
                    elif step < 80:
                        phase = "2. PROPAGACIÓN BIOLÓGICA"
                        desc_rates = 0.0
                    elif step < 100:
                        phase = "3. DECISIÓN DEL READOUT"
                        desc_rates = 0.0
                    else:
                        phase = "4. ACTIVACIÓN MOTORA (IK)"
                        desc_rates = 1.0
                        
                    payload = {
                        "fen": board.fen(),
                        "rates": real_rates[:2000].tolist(), # Enviamos subconjunto para Three.js
                        "desc_rates": desc_rates,
                        "frame": step,
                        "phase": phase,
                        "action_from": best_move.from_square if best_move else -1,
                        "action_to": best_move.to_square if best_move else -1,
                        "candidates": candidates,
                        "sf_move": "--"
                    }
                    await websocket.send_text(json.dumps(payload))
                    await asyncio.sleep(1.0 / 60.0)
                
                if best_move:
                    board.push(best_move)
                    
            # TURNO ALEATORIO (Negras)
            else:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    enemy_move = np.random.choice(legal_moves)
                    
                    # Pequeña animación de espera para el turno enemigo
                    for step in range(30):
                        payload = {
                            "fen": board.fen(),
                            "rates": np.zeros(2000).tolist(),
                            "desc_rates": 0.0,
                            "frame": step,
                            "phase": "TURNO DEL OPONENTE (ALEATORIO)",
                            "action_from": -1,
                            "action_to": -1,
                            "candidates": [],
                            "sf_move": "--"
                        }
                        await websocket.send_text(json.dumps(payload))
                        await asyncio.sleep(1.0 / 60.0)
                        
                    board.push(enemy_move)
            
        # Fin de la partida
        payload = {
            "fen": board.fen(),
            "rates": np.zeros(2000).tolist(),
            "desc_rates": 0.0,
            "frame": 0,
            "phase": f"FIN DE PARTIDA: {board.result()}",
            "action_from": -1,
            "action_to": -1,
            "candidates": [],
            "sf_move": "--"
        }
        await websocket.send_text(json.dumps(payload))
        
    except WebSocketDisconnect:
        pass
