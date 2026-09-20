import pandas as pd

from src.splits import split_by_game


def test_game_ids_never_cross_splits():
    frame = pd.DataFrame({"game_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3, "label": range(9)})
    groups = split_by_game(frame)
    memberships = {game: sum(game in set(group.game_id) for group in groups.values()) for game in frame.game_id.unique()}
    assert all(value == 1 for value in memberships.values())
