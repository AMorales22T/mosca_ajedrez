import numpy as np
import scipy.sparse as sp

def extract_subgraph(W: sp.csr_matrix, sensory_nodes: list[int], motor_nodes: list[int], max_neurons: int):
    """
    Extrae un subgrafo de tamaño máximo `max_neurons` que incluye los nodos 
    sensoriales y motores, rellenando el resto con las neuronas de mayor grado de conectividad.
    
    Args:
        W: sp.csr_matrix - Matriz de conectividad original.
        sensory_nodes: list[int] - Índices de las neuronas de entrada (visuales).
        motor_nodes: list[int] - Índices de las neuronas de salida (descendentes).
        max_neurons: int - Límite de neuronas del perfil.
        
    Returns:
        W_sub: sp.csr_matrix - Matriz del subgrafo.
        keep_list: list[int] - Mapeo de índices del subgrafo a la matriz original.
    """
    N = W.shape[0]
    if max_neurons >= N:
        return W, list(range(N))
        
    # Inicializar el conjunto con los nodos críticos
    keep = set(sensory_nodes) | set(motor_nodes)
    
    # Calcular el grado (in-degree + out-degree) para cada neurona
    # Como W puede tener pesos no unitarios, usamos binarización para contar conexiones reales
    W_bin = (W != 0).astype(int)
    degrees = np.array(W_bin.sum(axis=0)).flatten() + np.array(W_bin.sum(axis=1)).flatten()
    
    # Ordenar de mayor a menor grado
    sorted_nodes = np.argsort(degrees)[::-1]
    
    idx = 0
    while len(keep) < max_neurons and idx < len(sorted_nodes):
        keep.add(sorted_nodes[idx])
        idx += 1
        
    keep_list = sorted(list(keep))
    
    # Extraer el subgrafo
    W_sub = W[keep_list, :][:, keep_list]
    return W_sub, keep_list
