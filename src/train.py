from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import RUNS, json_dump
from .device import report_device
from .encoding import board_planes, move_to_index


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def synthetic_dataset(n: int = 512, seed: int = 0):
    import chess
    rng = random.Random(seed)
    boards, actions, values = [], [], []
    for _ in range(n):
        board = chess.Board()
        for _ in range(rng.randint(0, 20)):
            if board.is_game_over(): break
            board.push(rng.choice(list(board.legal_moves)))
        move = rng.choice(list(board.legal_moves))
        boards.append(board_planes(board)); actions.append(move_to_index(board, move))
        values.append(rng.choice([-1.0, 0.0, 1.0]))
    return torch.stack(boards), torch.tensor(actions), torch.tensor(values, dtype=torch.float32)


def _loader(data, batch_size, shuffle):
    return DataLoader(TensorDataset(*data), batch_size=batch_size, shuffle=shuffle)


def sanity_overfit(model: nn.Module, data, device: torch.device, steps: int = 80) -> float:
    model.train(); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    boards, actions, values = [x[: min(32, len(x))].to(device) for x in data]
    initial = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, predicted, _ = model(boards)
        loss = nn.functional.cross_entropy(logits, actions) + 0.2 * nn.functional.mse_loss(predicted, values)
        initial = loss.item() if initial is None else initial
        loss.backward(); optimizer.step()
    if loss.item() > initial * 0.25:
        raise AssertionError(f"El batch no sobreajusta: {initial:.4f} -> {loss.item():.4f}")
    return float(loss.item())


def train_model(model: nn.Module, train_data, val_data, *, epochs=2, batch_size=32, lr=2e-4,
                value_weight=0.2, run_name="run", seed=0, device=None) -> dict:
    seed_everything(seed); device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); sanity = sanity_overfit(model, train_data, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_dir = RUNS / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch in range(1, epochs + 1):
        model.train(); train_losses = []
        for boards, actions, values in _loader(train_data, batch_size, True):
            boards, actions, values = boards.to(device), actions.to(device), values.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, predicted, _ = model(boards)
                loss = nn.functional.cross_entropy(logits, actions) + value_weight * nn.functional.mse_loss(predicted, values)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            train_losses.append(loss.item())
        model.eval(); val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for boards, actions, values in _loader(val_data, batch_size, False):
                logits, predicted, _ = model(boards.to(device))
                val_losses.append((nn.functional.cross_entropy(logits, actions.to(device)) + value_weight * nn.functional.mse_loss(predicted, values.to(device))).item())
                correct += (logits.argmax(-1).cpu() == actions).sum().item(); total += len(actions)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses)), "val_top1": correct / max(1, total)}
        rows.append(row); print(run_name, row)
        torch.save({"model": model.state_dict(), "epoch": epoch, "seed": seed}, run_dir / f"checkpoint_{epoch:03d}.pt")
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    json_dump(run_dir / "metadata.json", {"run": run_name, "seed": seed, "device": str(device), "sanity_final_loss": sanity})
    return rows[-1]
