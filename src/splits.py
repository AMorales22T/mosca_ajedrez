from __future__ import annotations

import pandas as pd

from .common import stable_split


def split_by_game(frame: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    out = frame.copy()
    out["split"] = out["game_id"].astype(str).map(lambda game: stable_split(game, seed))
    groups = {name: out[out.split == name].copy() for name in ("train", "val", "test")}
    ids = [set(groups[name].game_id) for name in groups]
    if set.intersection(*ids):
        raise AssertionError("Una partida aparece en más de un split")
    return groups
