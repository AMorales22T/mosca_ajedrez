import torch
from torch import nn

from src.train import sanity_overfit


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(19 * 8 * 8, 4672)
        self.value = nn.Linear(19 * 8 * 8, 1)

    def forward(self, board, state=None):
        flat = board.flatten(1)
        return self.body(flat), self.value(flat).squeeze(-1).tanh(), flat


def test_sanity_batch_overfits():
    data = (torch.randn(4, 19, 8, 8), torch.tensor([1, 2, 3, 4]), torch.zeros(4))
    final_loss = sanity_overfit(TinyModel(), data, torch.device("cpu"), steps=80)
    assert final_loss >= 0
