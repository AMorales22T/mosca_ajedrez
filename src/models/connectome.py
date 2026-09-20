from __future__ import annotations

import os

import torch


def load_connectome(mode: str, units: int, seed: int = 0) -> torch.Tensor:
    if not 1000 <= units <= 4000:
        raise ValueError("units debe estar entre 1.000 y 4.000")
    if mode == "hemibrain":
        if not os.getenv("NEUPRINT_TOKEN"):
            raise RuntimeError("Modo hemibrain requiere NEUPRINT_TOKEN en .env; no se cambia a synthetic")
        raise RuntimeError("Loader hemibrain pendiente de configurar: especifica dataset neuPrint y subgrafo MB/CX")
    if mode != "synthetic":
        raise ValueError(f"Modo de conectoma desconocido: {mode}")
    generator = torch.Generator().manual_seed(seed)
    types = torch.multinomial(torch.tensor([0.12, 0.48, 0.25, 0.15]), units, replacement=True, generator=generator)
    mask = torch.rand(units, units, generator=generator) < 0.015
    # Keep neuron-type proportions in metadata-compatible deterministic form.
    mask.fill_diagonal_(False)
    return mask.float()


def shuffle_degree_preserving(mask: torch.Tensor, seed: int = 0, swaps: int = 1000) -> torch.Tensor:
    """Directed double-edge swaps preserve every row/column degree."""
    generator = torch.Generator().manual_seed(seed)
    out = mask.bool().clone()
    n = out.shape[0]
    edges = torch.nonzero(out, as_tuple=False).tolist()
    if len(edges) < 2:
        return out.float()
    for _ in range(swaps):
        a, b = edges[torch.randint(len(edges), (1,), generator=generator).item()], edges[torch.randint(len(edges), (1,), generator=generator).item()]
        if a[0] == b[0] or a[1] == b[1] or out[a[0], b[1]] or out[b[0], a[1]]:
            continue
        out[a[0], a[1]] = out[b[0], b[1]] = False
        out[a[0], b[1]] = out[b[0], a[1]] = True
        edges = torch.nonzero(out, as_tuple=False).tolist()
    return out.float()
