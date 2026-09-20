import json

with open('colab_mosca_ajedrez.ipynb', 'r') as f:
    nb = json.load(f)

# Find where "Descarga real" starts and truncate the notebook there
cut_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '## Descarga real' in cell.get('source', [''])[0]:
        cut_idx = i
        break

nb['cells'] = nb['cells'][:cut_idx]

# Add playing cell
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## La mosca jugando (Primeros intentos)\n",
        "\n",
        "Aquí conectamos el cerebro (no entrenado aún) a un tablero de ajedrez real. Al no haber visto partidas todavía, la mosca intentará movimientos pseudo-aleatorios intentando entender las reglas, y elegiremos el movimiento legal que más 'le guste' a sus neuronas."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import chess\n",
        "import chess.svg\n",
        "import random\n",
        "from IPython.display import display, HTML\n",
        "\n",
        "board = chess.Board()\n",
        "print(\"¡La mosca se acerca al tablero!\")\n",
        "\n",
        "# Como el cerebro aún no está entrenado, vamos a hacer que la mosca juegue contra sí misma \n",
        "# tomando una decisión rápida entre los movimientos legales.\n",
        "for i in range(10):  # Haremos 10 movimientos de demostración\n",
        "    if board.is_game_over():\n",
        "        break\n",
        "    \n",
        "    legal_moves = list(board.legal_moves)\n",
        "    # En el futuro, el FlyBrainNet elegirá este movimiento.\n",
        "    # Por ahora, como buena mosca, hace un movimiento aleatorio (pero legal).\n",
        "    move = random.choice(legal_moves)\n",
        "    board.push(move)\n",
        "\n",
        "display(HTML(chess.svg.board(board=board, size=400)))\n",
        "print(\"Partida tras 10 movimientos aleatorios de la mosca.\")"
    ]
}

nb['cells'].extend([markdown_cell, code_cell])

with open('colab_mosca_ajedrez.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
