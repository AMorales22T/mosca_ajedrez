from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess.pgn
import pandas as pd
import zstandard as zstd

from .common import PROCESSED, RAW, json_dump, stable_split


def _elo(headers: chess.pgn.Headers, key: str) -> int:
    try:
        return int(headers.get(key, "0"))
    except ValueError:
        return 0


def parse_month(path: Path, *, min_elo: int = 1800, max_games: int | None = None,
                positions_per_game: int = 10, shard_size: int = 10_000, seed: int = 0) -> dict:
    """Stream PGN and emit parquet shards; no complete PGN is held in memory."""
    if not path.exists():
        raise FileNotFoundError(path)
    out_dir = PROCESSED / "positions"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, accepted, seen = [], 0, 0
    started = time.perf_counter()
    with path.open("rb") as compressed:
        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        text = reader if False else None
        import io
        pgn = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while max_games is None or accepted < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            seen += 1
            h = game.headers
            if h.get("Variant", "Standard") != "Standard" or h.get("Termination", "") in {"abandoned", "timeout"}:
                continue
            white_elo, black_elo = _elo(h, "WhiteElo"), _elo(h, "BlackElo")
            if min(white_elo, black_elo) < min_elo:
                continue
            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < 12:
                continue
            indices = sorted(set(round(i * (len(moves) - 1) / max(1, positions_per_game - 1)) for i in range(positions_per_game)))
            game_id = h.get("Site", "").rstrip("/").split("/")[-1] or f"game_{seen}"
            for index in indices:
                board = game.board()
                for move in moves[:index]:
                    board.push(move)
                actual = moves[index]
                rows.append({"game_id": game_id, "fen": board.fen(), "uci": actual.uci(), "result": h.get("Result", "*"),
                             "ply": index, "white_elo": white_elo, "black_elo": black_elo,
                             "split": stable_split(game_id, seed)})
            accepted += 1
            if len(rows) >= shard_size:
                _write_shard(rows, out_dir, accepted // max(1, shard_size))
                rows = []
    if rows:
        _write_shard(rows, out_dir, (accepted // max(1, shard_size)) + 1)
    report = {"source": str(path), "games_seen": seen, "games_accepted": accepted,
              "seconds": round(time.perf_counter() - started, 2), "positions_per_game": positions_per_game}
    json_dump(PROCESSED / "parse_report.json", report)
    return report


def _write_shard(rows: list[dict], out_dir: Path, shard: int) -> None:
    pd.DataFrame(rows).drop_duplicates(["game_id", "ply"]).to_parquet(out_dir / f"positions_{shard:05d}.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--max-games", type=int)
    args = parser.parse_args()
    print(parse_month(args.pgn, min_elo=args.min_elo, max_games=args.max_games))
