# Mosca ajedrez

Proyecto reproducible en Python/PyTorch para comparar una `FlyBrainNet` recurrente enmascarada con controles honestos en ajedrez. El notebook está pensado para Google Colab Free; no es necesario ejecutar el entrenamiento en el PC.

## Estado de fase 0

Esta fase no descarga bases grandes. `make smoke` genera posiciones sintéticas pequeñas, entrena cuatro modelos y escribe `data/processed/smoke_report.json`. El modo del conectoma queda registrado como `synthetic`; no se presenta como Hemibrain real.

La política usa el espacio AlphaZero de 4.672 acciones: 64 casillas de origen por 73 tipos (56 movimientos de dama, 8 caballos y 9 subpromociones). La promoción a dama usa el movimiento de dama correspondiente. Las jugadas ilegales se enmascaran tanto al entrenar como al elegir una jugada.

## Colab Free

Abre `colab_mosca_ajedrez.ipynb`. La primera celda instala dependencias en la sesión de Colab; la segunda permite clonar/subir este directorio. Los datos se guardan en `/content/mosca_ajedrez` durante la sesión o en `MyDrive/mosca_ajedrez` si montas Drive. La sesión gratuita puede quedarse sin RAM/GPU: empieza con `configs/smoke.yaml`.

La celda de descarga real está deliberadamente bloqueada hasta que confirmes el tamaño. La base estándar mensual de Lichess se procesa en streaming y se parte en Parquet; no se carga el PGN completo en RAM. Las fuentes oficiales son la base de partidas, evaluaciones y puzzles de `database.lichess.org`.

## Conectoma y controles

`synthetic` crea un grafo reproducible de depuración. `hemibrain` exige `NEUPRINT_TOKEN` en `.env` y falla explícitamente mientras no se configure el dataset/subgrafo MB/CX; nunca cambia de modo en silencio. Los controles tienen la misma arquitectura de cabezas y se comprueba que el número de parámetros quede dentro de ±5%.

## Comandos locales solo para validar fase 0

```bash
python -m pip install -r requirements.txt
make test
make smoke
```

No ejecutes una descarga real hasta revisar el tamaño y confirmar espacio en Colab/Drive.
