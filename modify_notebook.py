import json

with open('colab_mosca_ajedrez.ipynb', 'r') as f:
    nb = json.load(f)

# Find pip cell and add networkx
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '!pip' in cell['source'][0]:
        cell['source'][0] = cell['source'][0].replace('pytest torch', 'pytest torch networkx')

# Find the cell index for "Cuerpo virtual"
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '## Descarga real' in cell['source'][0]:
        insert_idx = i
        break

if insert_idx != -1:
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cerebro virtual (Connectome)\n",
            "\n",
            "Visualización de las conexiones neuronales del modelo (FlyBrainNet) simulando la estructura del cerebro de la mosca."
        ]
    }
    
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import networkx as nx\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from src.models.fly import FlyBrainNet\n",
            "\n",
            "# Instanciar un cerebro pequeño para poder visualizarlo\n",
            "units = 256\n",
            "model = FlyBrainNet(units=units)\n",
            "adj = model.connectome.cpu().numpy()\n",
            "\n",
            "# Crear un grafo dirigido a partir de la matriz de adyacencia\n",
            "G = nx.from_numpy_array(adj, create_using=nx.DiGraph)\n",
            "\n",
            "# Visualizar un subgrafo de 50 neuronas para mayor claridad\n",
            "sub_G = G.subgraph(range(50))\n",
            "plt.figure(figsize=(10, 8))\n",
            "plt.title(\"Conexiones Neuronales (Subgrafo de 50 neuronas)\")\n",
            "pos = nx.spring_layout(sub_G, seed=42)\n",
            "nx.draw_networkx_nodes(sub_G, pos, node_size=50, node_color='purple', alpha=0.7)\n",
            "nx.draw_networkx_edges(sub_G, pos, edge_color='gray', alpha=0.3, arrows=True)\n",
            "plt.axis('off')\n",
            "plt.show()\n",
            "\n",
            "# Visualizar la matriz de adyacencia completa\n",
            "plt.figure(figsize=(8, 8))\n",
            "plt.title(f\"Matriz de Adyacencia Completa ({units}x{units})\")\n",
            "plt.imshow(adj, cmap='Greys', interpolation='none')\n",
            "plt.show()\n"
        ]
    }
    nb['cells'].insert(insert_idx, markdown_cell)
    nb['cells'].insert(insert_idx + 1, code_cell)

with open('colab_mosca_ajedrez.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

