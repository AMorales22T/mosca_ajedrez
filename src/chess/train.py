import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from src.chess.readout import ChessReadout
from src.chess.controls import BaselineLinear, make_shuffled_connectome, make_random_sparse
from src.chess.pipeline import CudaLIFEngine
from src.chess.encoding import encode_board, map_features_to_stimuli
import chess
import scipy.sparse as sp

DATA_DIR = "/run/media/dolfius/Juegos/mosca_ajedrez_data"

def uci_to_tensors(moves):
    src_list, dst_list, promo_list = [], [], []
    promo_map = {'q': 0, 'r': 1, 'n': 2, 'b': 3}
    for m in moves:
        src = chess.parse_square(m[:2])
        dst = chess.parse_square(m[2:4])
        promo = promo_map[m[4]] if len(m) == 5 else 0
        src_list.append(src)
        dst_list.append(dst)
        promo_list.append(promo)
    return torch.tensor(src_list), torch.tensor(dst_list), torch.tensor(promo_list)

class MmapDataset(Dataset):
    def __init__(self, mmap_array, src_t, dst_t, promo_t):
        self.mmap = mmap_array
        self.src = src_t
        self.dst = dst_t
        self.promo = promo_t
        
    def __len__(self):
        return len(self.src)
        
    def __getitem__(self, idx):
        # Read from disk mmap
        return torch.tensor(self.mmap[idx], dtype=torch.float32), self.src[idx], self.dst[idx], self.promo[idx]

def train_model(model, loader, name, epochs=5):
    print(f"\n--- Entrenando {name} ---")
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        t0 = time.time()
        for x, src, dst, promo in loader:
            x, src, dst = x.to("cuda"), src.to("cuda"), dst.to("cuda")
            
            optimizer.zero_grad()
            p_src, p_dst, p_promo, _ = model(x)
            
            loss_src = F.cross_entropy(p_src, src)
            loss_dst = F.cross_entropy(p_dst, dst)
            loss = loss_src + loss_dst
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred_src = p_src.argmax(dim=1)
            pred_dst = p_dst.argmax(dim=1)
            correct += ((pred_src == src) & (pred_dst == dst)).sum().item()
            total += src.size(0)
            
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f} - Acc: {correct/total*100:.2f}% - {elapsed:.1f}s")
        
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{name}.pt")
    
    # Liberar memoria de GPU
    model.cpu()
    torch.cuda.empty_cache()

def cache_and_train_control(engine_name, engine, moves_train, dataset_train, stim_nodes, N, src_t, dst_t, promo_t):
    print(f"\nGenerando caché para {engine_name}...")
    
    # Para ahorrar memoria, guardamos directamente a un mmap file
    out_path = os.path.join(DATA_DIR, "dataset", f"train_rates_{engine_name}.npy")
    mmap_out = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(len(dataset_train), N))
    
    batch_size = 256
    for i in range(0, len(dataset_train), batch_size):
        batch_d = dataset_train[i:i+batch_size]
        batch_f = np.array([encode_board(chess.Board(d["fen"])) for d in batch_d])
        stim_np = map_features_to_stimuli(batch_f, stim_nodes, N, current=20.0)
        stim_t = torch.tensor(stim_np, dtype=torch.float32, device="cuda")
        rates_t = engine.simulate_batch(stim_t, steps=50)
        mmap_out[i:i+batch_size] = rates_t.cpu().numpy()
        
    mmap_out.flush()
    # Reabrir en modo lectura
    mmap_read = np.load(out_path, mmap_mode='r')
    
    dataset = MmapDataset(mmap_read, src_t, dst_t, promo_t)
    # Num_workers=0 para no duplicar mmap en memoria
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
    
    model = ChessReadout(in_features=N)
    train_model(model, loader, engine_name)

def run_step6():
    print("Cargando datos con mmap...")
    rates_train = np.load(os.path.join(DATA_DIR, "dataset", "train_rates.npy"), mmap_mode='r')
    moves_train = np.load(os.path.join(DATA_DIR, "dataset", "train_moves.npy"))
    dataset_train = np.load(os.path.join(DATA_DIR, "dataset", "train.npy"), allow_pickle=True)
    
    src_t, dst_t, promo_t = uci_to_tensors(moves_train)
    
    # 1. FlyWire Connectome
    dataset_real = MmapDataset(rates_train, src_t, dst_t, promo_t)
    loader_real = DataLoader(dataset_real, batch_size=512, shuffle=True, num_workers=0)
    model_real = ChessReadout(in_features=rates_train.shape[1])
    target_params = sum(p.numel() for p in model_real.parameters())
    train_model(model_real, loader_real, "FlyWire_Readout")
    
    # 2. BaselineLinear
    print(f"\nPreparando BaselineLinear...")
    features_train = np.array([encode_board(chess.Board(d["fen"])) for d in dataset_train])
    loader_base = DataLoader(TensorDataset(torch.tensor(features_train, dtype=torch.float32), src_t, dst_t, promo_t), batch_size=512, shuffle=True)
    model_base = BaselineLinear(target_params=target_params)
    train_model(model_base, loader_base, "BaselineLinear")
    
    # Preparamos controles topológicos
    W_sub = sp.load_npz(os.path.join(DATA_DIR, "subgraph_laptop.npz"))
    N = W_sub.shape[0]
    stim_nodes = list(range(1546))
    
    # 3. ShuffledConnectome
    W_shuf = make_shuffled_connectome(W_sub)
    engine_shuf = CudaLIFEngine(W_shuf, device="cuda")
    cache_and_train_control("ShuffledConnectome", engine_shuf, moves_train, dataset_train, stim_nodes, N, src_t, dst_t, promo_t)
    
    # 4. RandomSparse
    density = W_sub.nnz / (N * N)
    W_rand = make_random_sparse(N, density)
    engine_rand = CudaLIFEngine(W_rand, device="cuda")
    cache_and_train_control("RandomSparse", engine_rand, moves_train, dataset_train, stim_nodes, N, src_t, dst_t, promo_t)

    print("\n¡PASO 6 COMPLETADO! Todos los modelos entrenados y guardados.")

if __name__ == "__main__":
    run_step6()
