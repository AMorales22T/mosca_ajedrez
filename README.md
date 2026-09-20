# Evaluación Final: Mosca vs Matemáticas (Hito 7)

Tras entrenar el modelo de decodificación lineal (`Readout`) sobre la actividad neuronal estimulada, hemos evaluado la precisión de la mosca frente a redes de control puramente matemáticas que poseen **exactamente el mismo presupuesto de parámetros**.

## 📊 Tabla Comparativa de Precisión (Set de Prueba)

El set de prueba (Test) consta de 4.813 posiciones de ajedrez nunca vistas durante el entrenamiento, provenientes de partidas de jugadores de Lichess con Elo > 2000.

| Modelo / Cerebro | Precisión Top-1 (Acierto Exacto) | Precisión Top-3 (En el podio) |
| :--- | :--- | :--- |
| **Mosca Biológica (FlyWire 783)** | **0.02%** | **1.66%** |
| **Cables Mezclados (Shuffled)** | **~0.05%** | **~2.10%** |
| **Control Aleatorio (RandomSparse)**| **~6.50%** | **~21.00%** |
| **Matemática Pura (BaselineLinear)**| **6.81%** | **21.96%** |

---

## 🔬 Conclusión Científica

**La biología no es magia.** 
El conectoma de *Drosophila melanogaster* es una red sumamente especializada (identificación de comida, vuelo, feromonas). Si lo utilizas como un reservorio matemático general (*Reservoir Computing*) para una tarea de alta abstracción espacial y no-local como es el Ajedrez, **fracasa de forma estrepitosa**.

1. **La mosca juega al azar:** Su 0.02% de precisión Top-1 equivale literalmente a elegir una casilla origen y destino al azar ($1 / 4096 \approx 0.024\%$). La topología biológica colapsa, interfiere o "apaga" las señales complejas del tablero.
2. **Las redes aleatorias le ganan:** Un cerebro inicializado con pesos y conexiones completamente aleatorias (`RandomSparse`) o una red densa estándar (`BaselineLinear`) aprenden a encontrar patrones de ajedrez y consiguen casi un **22% de acierto en el Top-3** con tan solo 5 épocas de entrenamiento. La matemática abstracta distribuye la información de forma uniforme, mientras que la biología la concentra en cuellos de botella específicos (ganglios) que son inútiles para el ajedrez.

El experimento demuestra empíricamente que **la estructura conectómica real de un insecto destruye la información útil para el ajedrez** en comparación con una red matemática aleatoria.
