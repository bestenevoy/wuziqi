from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import BOARD_AREA, BOARD_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match reference: conv+bn, add skip, then ReLU.
        out = self.bn1(self.conv1(x))
        out = out + x
        return F.relu(out)


def encode_board(
    history: list[list[int]] | torch.Tensor,
    player: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    # Two-plane encoding: current player stones + opponent stones.
    size = BOARD_SIZE
    if isinstance(history, list) and history and isinstance(history[0], np.ndarray):
        history = np.array(history)
    history_t = torch.as_tensor(history, device=device)

    if history_t.ndim == 2:
        if history_t.shape != (size, size):
            raise ValueError(f"Unexpected board shape: {tuple(history_t.shape)}")
        cur = (history_t == player).float()
        opp = ((history_t != 0) & (history_t != player)).float()
        return torch.stack((cur, opp), dim=0)

    if history_t.ndim == 3:
        # Accept [B, H, W] or already [2, H, W].
        if history_t.shape == (2, size, size):
            return history_t.float()
        if history_t.shape[1:] != (size, size):
            raise ValueError(f"Unexpected board batch shape: {tuple(history_t.shape)}")
        cur = (history_t == player).float()
        opp = ((history_t != 0) & (history_t != player)).float()
        return torch.stack((cur, opp), dim=1)

    if history_t.ndim == 4 and history_t.shape[1] == 2:
        return history_t.float()

    raise ValueError(f"Unexpected history shape: {tuple(history_t.shape)}")



class AZNet(nn.Module):
    # Reference-style AlphaZero net: 2-plane input, shallow residual trunk.
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res2 = ResidualBlock(128)

        # policy head
        self.policy_conv = nn.Conv2d(128, 1, kernel_size=1)

        # value head
        self.value_fc1 = nn.Linear(128, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入 x: [B, 2, H, W]
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)

        p = self.policy_conv(x)
        policy_logits = p.view(p.size(0), -1)

        v = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value.squeeze(-1)
