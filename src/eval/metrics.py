from __future__ import annotations

import torch


def policy_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    order = logits.argsort(dim=-1, descending=True)
    return {
        "top1": float((order[:, :1] == target[:, None]).any(dim=1).float().mean()),
        "top3": float((order[:, :3] == target[:, None]).any(dim=1).float().mean()),
    }
