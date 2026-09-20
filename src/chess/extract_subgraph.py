import os
import numpy as np
import scipy.sparse as sp

# Aseguramos FLYPOKE_DATA
os.environ["FLYPOKE_DATA"] = "/run/media/dolfius/Juegos/mosca_ajedrez_data/flypoke"
from flypoke.data import build_network

DATA_DIR = "/run/media/dolfius/Juegos/mosca_ajedrez_data"

def extract_and_save(target_size=30000):
    print("Cargando cerebro completo (FlyWire 783)...")
    net = build_network(min_syn=5)
    N = net.n
    W = net.W
    
    # 1. Identificar nodos ancla (Sensoriales visuales y descendentes/motores)
    # Para el ajedrez usaremos el sistema visual.
    # En FlyWire, los fotoreceptores o neuronas visuales (visual_projection, etc).
    # Vamos a coger una muestra representativa de visual, central y descendentes.
    print("Identificando anclas (visual, central, descending)...")
    visual_idx = net.select("super_class=visual_projection|visual_centrifugal")
    if len(visual_idx) == 0:
        # Fallback si no hay suficientes visuales claras
        visual_idx = net.select("cell_class=optic_lobe_neuron")
        if len(visual_idx) == 0:
            # Fallback 2: coger nodos de más alto grado
            degrees = np.array(W.sum(axis=1)).flatten() + np.array(W.sum(axis=0)).flatten()
            visual_idx = np.argsort(degrees)[-1000:]
            
    descending_idx = net.select("super_class=descending")
    
    anchors = np.unique(np.concatenate([visual_idx, descending_idx]))
    print(f"Anclas seleccionadas: {len(anchors)}")
    
    # 2. Expansión (Random Walk o Grados) para llegar a target_size
    # Forma rápida: coger los N-len(anchors) nodos con más sinapsis (grado total)
    degrees = np.array(np.abs(W).sum(axis=1)).flatten() + np.array(np.abs(W).sum(axis=0)).flatten()
    
    # Excluir anclas para no repetirlas
    degrees[anchors] = -1 
    
    remaining = target_size - len(anchors)
    if remaining > 0:
        top_nodes = np.argsort(degrees)[-remaining:]
        subgraph_nodes = np.concatenate([anchors, top_nodes])
    else:
        subgraph_nodes = anchors[:target_size]
        
    subgraph_nodes = np.sort(subgraph_nodes)
    print(f"Nodos del subgrafo: {len(subgraph_nodes)}")
    
    # Extraer submatriz
    print("Extrayendo submatriz...")
    W_sub = W[subgraph_nodes, :][:, subgraph_nodes]
    print(f"Submatriz extraída. Sinapsis: {W_sub.nnz}")
    
    # Guardar en disco
    out_path = os.path.join(DATA_DIR, "subgraph_laptop.npz")
    sp.save_npz(out_path, W_sub)
    
    # Guardar los índices originales por si los necesitamos
    idx_path = os.path.join(DATA_DIR, "subgraph_laptop_indices.npy")
    np.save(idx_path, subgraph_nodes)
    print(f"Subgrafo guardado en {out_path}")

if __name__ == "__main__":
    extract_and_save(30000)
