from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from .common import RAW, json_dump

BASE = "https://database.lichess.org"


def urls(month: str) -> dict[str, str]:
    return {
        "games": f"{BASE}/standard/lichess_db_standard_rated_{month}.pgn.zst",
        "evals": f"{BASE}/lichess_db_eval.jsonl.zst",
        "puzzles": f"{BASE}/lichess_db_puzzle.csv.zst",
    }


def download(url: str, destination: Path, *, confirm: bool = False, expected_sha256: str | None = None) -> Path:
    """Download only after explicit confirmation; stream to a resumable .part file."""
    if not confirm:
        raise RuntimeError("Descarga bloqueada: llama a download(..., confirm=True) después de confirmar el tamaño.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "mosca-ajedrez/0.1 research"})
    with urllib.request.urlopen(request, timeout=60) as source, partial.open("ab") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    partial.replace(destination)
    if expected_sha256:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(f"SHA256 inesperado para {destination}: {digest}")
    json_dump(destination.with_suffix(destination.suffix + ".meta.json"), {"url": url, "path": str(destination), "size": destination.stat().st_size})
    return destination
