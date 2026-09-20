from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RUNS = ROOT / "runs"
for path in (RAW, PROCESSED, RUNS):
    path.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")


def stable_split(game_id: str, seed: int = 0) -> str:
    value = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()[0] / 255
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
