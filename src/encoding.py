from __future__ import annotations

"""AlphaZero-style chess encoding.

The policy space is 4,672 = 64 source squares * 73 move types:
56 queen-like moves (8 directions x 7 distances), 8 knight moves and
9 underpromotions (three pieces x three file offsets). Queen promotions are
represented by the corresponding queen-like move. Squares are oriented from
the side to move, so the moving side always advances in the same direction.
"""

from dataclasses import dataclass

import chess
import torch

POLICY_SIZE = 4672
QUEEN_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
KNIGHT_DELTAS = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def _oriented_coords(square: chess.Square, turn: chess.Color) -> tuple[int, int]:
    rank, file = chess.square_rank(square), chess.square_file(square)
    return (rank, file) if turn == chess.WHITE else (7 - rank, 7 - file)


def _board_square(rank: int, file: int, turn: chess.Color) -> chess.Square:
    if turn == chess.BLACK:
        rank, file = 7 - rank, 7 - file
    return chess.square(file, rank)


def square_to_action(square: chess.Square, turn: chess.Color) -> int:
    rank, file = _oriented_coords(square, turn)
    return rank * 8 + file


def action_to_square(action_square: int, turn: chess.Color) -> chess.Square:
    return _board_square(action_square // 8, action_square % 8, turn)


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    sr, sf = _oriented_coords(move.from_square, board.turn)
    dr, df = _oriented_coords(move.to_square, board.turn)
    delta = (dr - sr, df - sf)
    source = sr * 8 + sf
    if move.promotion in UNDERPROMOTIONS:
        if delta[0] != 1 or delta[1] not in (-1, 0, 1):
            raise ValueError(f"Promoción inválida para AlphaZero: {move}")
        return source * 73 + 64 + UNDERPROMOTIONS.index(move.promotion) * 3 + delta[1] + 1
    if delta in KNIGHT_DELTAS:
        return source * 73 + 56 + KNIGHT_DELTAS.index(delta)
    for direction, (vr, vf) in enumerate(QUEEN_DIRECTIONS):
        if vr == 0 and delta[0] != 0 or vf == 0 and delta[1] != 0:
            continue
        distance = delta[0] // vr if vr else delta[1] // vf
        if distance in range(1, 8) and (delta[0], delta[1]) == (vr * distance, vf * distance):
            return source * 73 + direction * 7 + distance - 1
    raise ValueError(f"Jugada fuera del espacio AlphaZero: {move}")


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    if not 0 <= index < POLICY_SIZE:
        raise ValueError(index)
    source, move_type = divmod(index, 73)
    sr, sf = source // 8, source % 8
    if move_type < 56:
        direction, distance = divmod(move_type, 7)
        vr, vf = QUEEN_DIRECTIONS[direction]
        dr, df = sr + vr * (distance + 1), sf + vf * (distance + 1)
        if not (0 <= dr < 8 and 0 <= df < 8):
            raise ValueError("Índice apunta fuera del tablero")
        from_square = action_to_square(source, board.turn)
        to_square = action_to_square(dr * 8 + df, board.turn)
        promotion = chess.QUEEN if board.piece_at(from_square) and board.piece_at(from_square).piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7) else None
        return chess.Move(from_square, to_square, promotion=promotion)
    if move_type < 64:
        dr, df = sr + KNIGHT_DELTAS[move_type - 56][0], sf + KNIGHT_DELTAS[move_type - 56][1]
        if not (0 <= dr < 8 and 0 <= df < 8):
            raise ValueError("Índice apunta fuera del tablero")
        return chess.Move(action_to_square(source, board.turn), action_to_square(dr * 8 + df, board.turn))
    promo = UNDERPROMOTIONS[(move_type - 64) // 3]
    df = (move_type - 64) % 3 - 1
    return chess.Move(action_to_square(source, board.turn), action_to_square((sr + 1) * 8 + sf + df, board.turn), promotion=promo)


def legal_action_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(board, move)] = True
    return mask


def mask_logits(logits: torch.Tensor, board: chess.Board | None = None, legal: torch.Tensor | None = None) -> torch.Tensor:
    legal = legal if legal is not None else legal_action_mask(board)  # type: ignore[arg-type]
    if not legal.any():
        raise ValueError("No hay jugadas legales")
    mask = legal.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        if mask.ndim < logits.ndim:
            mask = mask.expand_as(logits)
        else:
            mask = mask.reshape(logits.shape)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def board_planes(board: chess.Board, repetition_count: int = 1) -> torch.Tensor:
    """Return 19 planes: 12 pieces, turn, 4 castling, en-passant, repetition."""
    planes = torch.zeros(19, 8, 8, dtype=torch.float32)
    own = board.turn
    for square, piece in board.piece_map().items():
        rank, file = _oriented_coords(square, own)
        row, col = 7 - rank, file
        plane = piece.piece_type - 1 + (0 if piece.color == own else 6)
        planes[plane, row, col] = 1.0
    planes[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)
    castling = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for i, has_right in enumerate(castling, start=13):
        if has_right:
            planes[i].fill_(1.0)
    if board.ep_square is not None:
        rank, file = _oriented_coords(board.ep_square, own)
        planes[17, 7 - rank, file] = 1.0
    planes[18].fill_(min(1.0, max(0.0, repetition_count / 3.0)))
    return planes


def fen_to_tensor(fen: str) -> torch.Tensor:
    return board_planes(chess.Board(fen))
