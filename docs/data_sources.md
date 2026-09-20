# Fuentes de datos

La fuente oficial es `https://database.lichess.org/`. Lichess publica partidas mensuales estándar en `.pgn.zst`, la base `lichess_db_eval.jsonl.zst` con FEN y evaluaciones Stockfish y `lichess_db_puzzle.csv.zst` con puzzles. La página indica CC0 para las exportaciones y documenta sus esquemas.

El código no descarga estos archivos al crear el proyecto. `src/data_download.py` exige `confirm=True`, escribe un `.part` reanudable y guarda metadatos de URL/tamaño. El PGN se procesa con `zstandard` y `python-chess` en streaming.
