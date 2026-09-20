import chess
import torch

from src.encoding import POLICY_SIZE, board_planes, index_to_move, legal_action_mask, mask_logits, move_to_index


def test_fen_tensor_roundtrip_shape_and_stability():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 2")
    assert board_planes(board).shape == (19, 8, 8)
    assert torch.equal(board_planes(board), board_planes(chess.Board(board.fen())))


def test_move_index_roundtrip():
    for board in (chess.Board(), chess.Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1")):
        for move in list(board.legal_moves):
            assert index_to_move(board, move_to_index(board, move)) == move


def test_mask_never_selects_illegal_move():
    board = chess.Board(); logits = torch.zeros(POLICY_SIZE)
    selected = int(mask_logits(logits, board).argmax())
    assert index_to_move(board, selected) in board.legal_moves
