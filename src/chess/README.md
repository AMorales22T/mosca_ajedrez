# Mosca Ajedrez (LIF Simulator)

Este directorio contiene el simulador LIF (Leaky Integrate-and-Fire) en CPU puro (usando `scipy.sparse`) del conectoma de Drosophila, diseñado para aprender ajedrez.

## Hito 1: Motor y Sanidad
* `config.py`: Definición de perfiles de hardware (tiny, laptop, full).
* `lif.py`: Motor LIF vectorizado por lotes.
* `sanity.py`: Experimento de alimentación mockeado para verificar las dinámicas del modelo.
* `benchmark.py`: Medición de rendimiento en CPU.

## Hito 2: Subgrafo y Ajedrez
* `subgraph.py`: Extracción del subgrafo basado en grados y neuronas sensoriales/motores.
* `encoding.py`: Codificación del tablero (773 características) y mapeo retinotópico simulado a neuronas.
* `data_chess.py`: Streaming de puzzles Lichess.
* `pipeline.py`: Ejecuta la simulación por lotes y guarda la caché de *firing rates* en disco superando pruebas de separabilidad.

## Hito 3: Entrenamiento y Controles
* `readout.py`: Red de lectura (MLP en PyTorch) con cabezas de origen, destino, promoción y valor.
* `controls.py`: Controles empíricos (`BaselineLinear`, `ShuffledConnectome`, `RandomSparse`).
* `evaluate_controls.py`: Verifica que el presupuesto de parámetros sea el mismo y que los controles topológicos conserven los grados/densidad.
* `train.py`: Script de entrenamiento del readout sobre los datos cacheados.

## Hito 4: Evaluación y Partidas Completas
* `eval_metrics.py`: Computa las métricas de acierto (Top-1 y Top-3).
* `play.py`: Contiene el bucle de la partida. Extrae los logits de la red, enmascara las jugadas ilegales de ajedrez, e interactúa con el oponente.
* `evaluate.py`: Juega partidas completas evaluativas de forma automática (mosca vs aleatorio / mosca vs Stockfish).

## Hito 5 y 6: Visualización (En vivo y Replay)
* `viz/server.py`: Servidor FastAPI con endpoints para archivos estáticos y WebSockets de telemetría.
* `viz/static/index.html` & `app.js`: Interfaz de usuario estructurada en 3 paneles que combina HTML clásico (Tablero 2D), **Three.js** (Mosca 3D estilizada respondiendo a los estímulos) y **Canvas 2D** (Renderizado hiper-rápido de *spikes* en el subgrafo neuronal con Tooltips interactivos).

### Reproducir Visualización
Arranca el servidor local:
```bash
./venv/bin/python -m uvicorn src.chess.viz.server:app --reload --port 8000
```
Y abre [http://127.0.0.1:8000](http://127.0.0.1:8000) en tu navegador.
