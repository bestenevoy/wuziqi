from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

BOARD_SIZE = 15
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
WIN_LEN = 5


@dataclass
class Gomoku:
    board: np.ndarray = field(
        default_factory=lambda: np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    )
    size: int = BOARD_SIZE
    win_len: int = WIN_LEN
    move_count: int = 0
    last_action: Optional[Tuple[int, int]] = None
    nearby_empty: Set[int] = field(default_factory=set)
    nearby_radius: int = 2

    def __post_init__(self) -> None:
        # 如果是全空初始化，设置初始候选点为中心
        if self.move_count == 0 and not np.any(self.board):
            self.nearby_empty.clear()
            center = (self.size // 2) * self.size + (self.size // 2)
            self.nearby_empty.add(center)

    def reset(self) -> np.ndarray:
        # 清空棋盘并重置候选点为中心。
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.move_count = 0
        self.last_action = None
        self.nearby_empty.clear()
        center = (self.size // 2) * self.size + (self.size // 2)
        self.nearby_empty.add(center)
        return self.board

    def set_board(
        self,
        board: np.ndarray,
        last_action: Optional[Tuple[int, int] | int] = None,
        move_count: Optional[int] = None,
        nearby_empty: Optional[Set[int]] = None,
    ) -> None:
        """
        核心修改：接受完整的状态参数，避免昂贵的重算。
        """
        board_arr = np.asarray(board, dtype=np.int8)
        if board_arr.ndim == 1:
            board_arr = board_arr.reshape(self.size, self.size)
        elif board_arr.ndim != 2:
            raise ValueError(f"Invalid board shape: {board_arr.shape}")
        self.board = board_arr.copy()
        if last_action is None:
            self.last_action = None
        elif isinstance(last_action, int):
            self.last_action = divmod(last_action, self.size)
        else:
            self.last_action = (int(last_action[0]), int(last_action[1]))

        if move_count is not None:
            self.move_count = move_count
        else:
            # 未提供 move_count 时按棋盘统计。
            self.move_count = int(np.count_nonzero(self.board))

        if nearby_empty is not None:
            self.nearby_empty = set(nearby_empty)
        else:
            # 只有在未提供缓存时才重算，MCTS中不应触发此逻辑
            self._rebuild_nearby_empty()

    def step(self, action: int, player: int) -> None:
        r, c = divmod(action, self.size)
        if self.board[r, c] != 0:
            raise ValueError(f"Invalid action {action}")

        # 落子并更新计数与最后一步。
        self.board[r, c] = player
        self.move_count += 1
        self.last_action = (r, c)

        # 增量更新：移除当前点，添加周围新空位
        self.nearby_empty.discard(action)
        self._add_nearby(action, radius=self.nearby_radius)

    def _add_nearby(self, action: int, radius: int = 2) -> None:
        # 将指定半径内的空位加入候选集合。
        size = self.size
        r, c = divmod(action, size)
        r_min, r_max = max(0, r - radius), min(size - 1, r + radius)
        c_min, c_max = max(0, c - radius), min(size - 1, c + radius)

        for nr in range(r_min, r_max + 1):
            for nc in range(c_min, c_max + 1):
                idx = nr * size + nc
                if self.board[nr, nc] == 0:
                    self.nearby_empty.add(idx)

    def _rebuild_nearby_empty(self) -> None:
        # 依据已有落子重建候选空位集合。
        self.nearby_empty.clear()
        stones = np.flatnonzero(self.board)
        if stones.size == 0:
            center = (self.size // 2) * self.size + (self.size // 2)
            self.nearby_empty.add(center)
            return
        for action in stones.tolist():
            self._add_nearby(action, radius=self.nearby_radius)

    def candidate_actions(self) -> List[int]:
        # 直接返回缓存集合
        return list(self.nearby_empty)

    def check_winner(self) -> Optional[int]:
        # 优先检查最后一步
        if self.last_action is not None:
            winner = self._check_at(self.last_action)
            if winner is not None:
                return winner

        # 兜底：未知最后一步或最后一步不在连线中时，扫描全盘。
        if self.move_count > 0:
            for idx, p in enumerate(self.board.ravel()):
                if p != 0:
                    w = self._check_at(divmod(idx, self.size))
                    if w is not None:
                        return w

        if self.move_count >= BOARD_AREA:
            return 0
        return None

    def _check_at(self, action: Tuple[int, int]) -> Optional[int]:
        # 从某一落子点向 4 个方向检查是否连成胜利长度。
        r, c = action
        player = self.board[r, c]
        if player == 0:
            return None
        size = self.size
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            for direction in [1, -1]:
                for k in range(1, self.win_len):
                    nr, nc = r + dr * k * direction, c + dc * k * direction
                    if (
                        0 <= nr < size
                        and 0 <= nc < size
                        and self.board[nr, nc] == player
                    ):
                        count += 1
                    else:
                        break
            if count >= self.win_len:
                return player
        return None
