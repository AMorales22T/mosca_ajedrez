import numpy as np
import scipy.sparse as sp

class LIFEngine:
    def __init__(self, W_sparse: sp.csr_matrix, tau=20.0, v_th=1.0, v_reset=0.0, dt=1.0, gain=1.0, base_current=0.0):
        """
        Motor de simulación LIF por lotes usando numpy y scipy.sparse.
        
        Args:
            W_sparse: scipy.sparse.csr_matrix (N, N) - Pesos sinápticos. W[i,j] es la conexión de i a j.
            tau: Constante de tiempo de la membrana.
            v_th: Umbral de disparo.
            v_reset: Voltaje de reseteo post-disparo.
            dt: Paso de tiempo.
            gain: Multiplicador global de ganancia sináptica (para calibración).
            base_current: Corriente base de fondo (para evitar red muerta).
        """
        # Asegurarnos de que está en formato CSR para multiplicaciones eficientes
        if not isinstance(W_sparse, sp.csr_matrix):
            W_sparse = sp.csr_matrix(W_sparse)
            
        self.W = W_sparse.multiply(gain)
        self.tau = tau
        self.v_th = v_th
        self.v_reset = v_reset
        self.dt = dt
        self.N = W_sparse.shape[0]
        self.base_current = base_current
        
        self.decay = np.exp(-dt / tau)

    def simulate_batch(self, stimuli: np.ndarray, steps: int) -> np.ndarray:
        """
        Simula las dinámicas LIF para un lote de posiciones/estímulos en paralelo.
        
        Args:
            stimuli: np.ndarray (Batch, N) - Corriente externa de entrada constante.
            steps: int - Número de pasos de simulación.
            
        Returns:
            rates: np.ndarray (Batch, N) - Tasa de disparo (firing rate) promedio de cada neurona.
        """
        Batch = stimuli.shape[0]
        # (Batch, N)
        v = np.full((Batch, self.N), self.v_reset, dtype=np.float32)
        spikes = np.zeros((Batch, self.N), dtype=np.float32)
        total_spikes = np.zeros((Batch, self.N), dtype=np.float32)
        
        for _ in range(steps):
            # Producto disperso @ denso: (Batch, N) = (Batch, N) @ (N, N)
            # Nota: scipy.sparse sobrecarga @ para matrices. spikes @ self.W es eficiente.
            syn_input = spikes @ self.W
            
            # Dinámica LIF discreta
            v = v * self.decay + (syn_input + stimuli + self.base_current) * (1.0 - self.decay)
            
            # Disparo
            spikes = (v > self.v_th).astype(np.float32)
            total_spikes += spikes
            
            # Reseteo duro
            v[spikes > 0] = self.v_reset
            
        return total_spikes / steps
