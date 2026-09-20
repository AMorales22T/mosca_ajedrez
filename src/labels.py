from __future__ import annotations

import json
from pathlib import Path

import chess
import pandas as pd
import zstandard as zstd


def result_value(result: str, turn: bool) -> float:
    if result == "1/2-1/2":
        return 0.0
    white = 1.0 if result == "1-0" else -1.0
    return white if turn == chess.WHITE else -white


def attach_result_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["label"] = [result_value(result, chess.Board(fen).turn) for result, fen in zip(out.result, out.fen)]
    out["label_source"] = "game_result"
    return out


def attach_eval_labels(frame: pd.DataFrame, eval_path: Path) -> pd.DataFrame:
    """Stream the huge eval file and retain only FENs present in the shard."""
    if not eval_path.exists():
        return attach_result_labels(frame)
    out = attach_result_labels(frame)
    wanted = set(out.fen)
    found: dict[str, float] = {}
    with eval_path.open("rb") as raw:
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        import io
        for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
            if not wanted:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fen = row.get("fen")
            if fen not in wanted or not row.get("evals"):
                continue
            best = max(row["evals"], key=lambda item: item.get("depth", -1))
            pv = (best.get("pvs") or [{}])[0]
            found[fen] = float(pv["cp"]) / 100.0 if "cp" in pv else (100.0 if pv.get("mate", 0) > 0 else -100.0)
            wanted.discard(fen)
    mask = out.fen.isin(found)
    out.loc[mask, "label"] = out.loc[mask, "fen"].map(found)
    out.loc[mask, "label_source"] = "lichess_stockfish_eval"
    return out
