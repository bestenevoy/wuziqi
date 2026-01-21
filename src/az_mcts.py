from __future__ import annotations

import math
import random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 假设这些模块你已经有了
from az_net import AZNet, encode_board
from env import BOARD_AREA, Gomoku

@dataclass
class MCTSNode:
    """
    MCTS 节点
    player: 当前节点 *轮到* 谁下棋 (1: Black, -1: White)
    """
    player: int
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    # 动作 -> 子节点
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def q(self) -> float:
        """平均价值 (相对于当前节点 player)"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class AZMCTS:
    def __init__(
        self,
        model: AZNet,
        sims: int = 400,         # 每次落子的模拟次数
        c_puct: float = 5.0,     # 探索常数 (五子棋分支多，建议设大一点，如 5.0)
        dir_alpha: float = 0.3,  # Dirichlet 噪声参数
        dir_eps: float = 0.25,   # 噪声占比
        prune_radius: int = 2,   # 剪枝半径
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.sims = sims
        self.c_puct = c_puct
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps
        self.prune_radius = prune_radius
        self.device = device or torch.device("cpu")
        
        # 搜索树状态维护
        self.root: Optional[MCTSNode] = None
        self.root_key: Optional[str] = None
        self.last_root_stats: Optional[dict] = None

    def copy_env(self, env: Gomoku) -> Gomoku:
        """
        快速复制环境用于模拟。
        注意：如果 Gomoku 类支持 undo/revert，可以改为非拷贝式回溯，性能会更好。
        """
        new_env = Gomoku()
        # 必须使用 .copy() 确保 numpy 数组也是深拷贝
        new_env.set_board(
            board=env.board.copy(),
            last_action=env.last_action,
            move_count=env.move_count,
            nearby_empty=list(env.nearby_empty) if hasattr(env, "nearby_empty") else None,
        )
        new_env.nearby_radius = getattr(env, "nearby_radius", 1)
        return new_env

    def run(self, env: Gomoku, player: int, add_noise: bool = True) -> List[float]:
        """
        执行 MCTS 搜索，返回动作概率分布 pi。
        """
        # --- 2. 根节点准备 ---
        current_key = self._state_key(env)
        
        # 尝试复用树：如果有根节点，且棋盘状态对得上，且轮次对得上
        if self.root is None or self.root_key != current_key or self.root.player != player:
            self.root = MCTSNode(player=player)
            # 新建根节点后，必须立即扩展它，算出 Prior
            self._expand_node(self.root, env)
            self.root_key = current_key
        
        # 训练模式下，给根节点加噪声以增加探索多样性
        if add_noise:
            self._apply_dirichlet(self.root)

        # --- 3. 核心模拟循环 (串行) ---
        for _ in range(self.sims):
            self._simulate(env)

        # --- 4. 统计结果生成策略 ---
        visits = [0] * BOARD_AREA
        for action, child in self.root.children.items():
            visits[action] = child.visits
        
        total_visits = sum(visits)
        if total_visits == 0:
            # 异常兜底：如果没有访问过任何节点（极少见），返回均匀分布
            self.last_root_stats = {"total_visits": 0, "children": []}
            return [1.0 / BOARD_AREA] * BOARD_AREA

        self.last_root_stats = self._collect_root_stats(total_visits)
        return [v / total_visits for v in visits]

    def _collect_root_stats(self, total_visits: int) -> dict:
        if self.root is None:
            return {"total_visits": 0, "children": []}
        sqrt_total = math.sqrt(max(1, self.root.visits))
        children = []
        for action, child in self.root.children.items():
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visits)
            q = -child.q
            score = q + u
            children.append(
                {
                    "action": action,
                    "visits": child.visits,
                    "prior": float(child.prior),
                    "q": float(q),
                    "u": float(u),
                    "score": float(score),
                    "pi": float(child.visits / total_visits) if total_visits > 0 else 0.0,
                }
            )
        children.sort(key=lambda x: x["visits"], reverse=True)
        return {"total_visits": total_visits, "children": children}

    def _simulate(self, env: Gomoku) -> None:
        """
        单次 MCTS 模拟过程：Selection -> Expansion -> Backpropagation
        """
        env_copy = self.copy_env(env)
        node = self.root
        path = [node] # 记录路径用于回传

        # (A) Selection: 依据 PUCT 下潜直到叶子节点或终局
        while node.children:
            winner = env_copy.check_winner()
            if winner is not None:
                break
            
            action, child = self._select_child(node)
            env_copy.step(action, node.player) # 当前节点玩家落子
            node = child
            path.append(node)

        # (B) Expansion & Evaluation
        # 检查最终状态
        winner = env_copy.check_winner()
        value = 0.0

        if winner is not None:
            # 游戏结束
            if winner == 0:
                value = 0.0
            else:
                # 如果 winner 等于当前 node 的玩家，说明当前玩家赢了（但在 AlphaZero 逻辑里通常是上一手导致赢）
                # value 是相对于 "node.player" 的价值
                value = 1.0 if winner == node.player else -1.0
        else:
            # 未结束，使用神经网络评估并扩展
            value = self._expand_node(node, env_copy)

        # (C) Backpropagation
        self._backprop(path, value)

    def _expand_node(self, node: MCTSNode, env: Gomoku) -> float:
        """
        扩展节点：网络推理 -> Masking -> 创建子节点
        返回：网络预测的 Value (当前节点视角)
        """
        # 1. 编码输入
        board_tensor = encode_board(env.board, node.player, device=self.device)
        
        # 2. 网络推理
        with torch.no_grad():
            # unsqueeze(0) 增加 batch 维度
            policy_logits, value_tensor = self.model(board_tensor.unsqueeze(0))
        
        value = float(value_tensor.item())
        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # 3. 获取合法动作并 Masking
        available = self._available_actions(env)
        
        valid_probs = []
        prob_sum = 0.0
        for a in available:
            p = policy_probs[a]
            valid_probs.append(p)
            prob_sum += p
            
        # 4. 创建子节点
        if prob_sum > 0:
            # 归一化概率
            for i, action in enumerate(available):
                prior = valid_probs[i] / prob_sum
                # 子节点轮到对手
                node.children[action] = MCTSNode(player=-node.player, prior=prior)
        else:
            # 兜底：如果网络预测所有合法点概率都为0
            prior = 1.0 / len(available)
            for action in available:
                node.children[action] = MCTSNode(player=-node.player, prior=prior)
                
        return value

    def _backprop(self, path: List[MCTSNode], leaf_value: float) -> None:
        """
        反向传播价值。
        leaf_value: 是 path[-1] (叶子节点) 视角的价值。
        """
        current_value = leaf_value
        
        # 从叶子往根反向遍历
        for node in reversed(path):
            node.visits += 1
            node.value_sum += current_value
            # 每往上一层，玩家切换，价值取反
            current_value = -current_value

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """PUCT 选择公式"""
        # node.visits 包含了所有子节点的 visits 之和 (因为 backprop 时父节点也会 +1)
        # 也就是 sqrt(N_parent)
        sqrt_total = math.sqrt(max(1, node.visits))
        
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            # PUCT = Q + U
            # U = c_puct * P * sqrt(N_parent) / (1 + N_child)
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visits)
            
            # 重要：child.q 是子节点视角的期望胜率
            # 当前节点做决策时，希望选让子节点最惨（或者说对自己最好）的分支
            # 在零和游戏中，当前视角的 Q = -child.q
            q_value = -child.q
            
            score = q_value + u
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def select_action(self, pi: List[float], temp: float = 1.0) -> int:
        """
        根据策略 pi 选择最终落子
        temp=0: 贪婪选择 (Argmax)
        temp>0: 按概率采样 (Softmax temperature)
        """
        available = [i for i, p in enumerate(pi) if p > 0]
        if not available:
            return int(np.argmax(pi))

        if temp <= 1e-3:
            # 贪婪模式
            return int(np.argmax(pi))
        
        # 温度缩放
        # pi_i^(1/T)
        pi_vals = np.array(pi)
        # 避免 log(0)
        pi_vals = np.maximum(pi_vals, 1e-10) 
        log_pi = np.log(pi_vals) / temp
        exp_pi = np.exp(log_pi - np.max(log_pi)) # 减最大值防溢出
        probs = exp_pi / np.sum(exp_pi)
        
        return np.random.choice(len(probs), p=probs)

    def advance(self, action: int, env: Gomoku) -> None:
        """
        在真实落子后，将树的根节点移动到对应的子节点，保留搜索统计信息。
        """
        if self.root is None:
            return
            
        if action in self.root.children:
            self.root = self.root.children[action]
            # 更新 key，确保下次 run 时校验通过
            self.root_key = self._state_key(env)
        else:
            # 如果落子不在树中（例如对手下了一个被剪枝掉的奇怪位置），重置树
            self._reset_tree()

    def _reset_tree(self):
        self.root = None
        self.root_key = None

    def _apply_dirichlet(self, node: MCTSNode):
        """添加 Dirichlet 噪声"""
        actions = list(node.children.keys())
        if not actions:
            return
        
        noise = np.random.dirichlet([self.dir_alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - self.dir_eps) * child.prior + self.dir_eps * noise[i]

    def _state_key(self, env: Gomoku) -> str:
        # 简单的棋盘哈希，用于校验
        # 生产环境建议用 hash(env.board.tobytes()) 更快
        return env.board.tobytes().hex()

    def _one_hot_policy(self, action: int) -> List[float]:
        pi = [0.0] * BOARD_AREA
        pi[action] = 1.0
        return pi

    def _available_actions(self, env: Gomoku) -> List[int]:
        if self.prune_radius <= 0:
             return [i for i, v in enumerate(env.board.ravel()) if v == 0]
        return env.candidate_actions()

    # --- 辅助判断逻辑 (同原代码) ---
    def _winning_moves(self, env: Gomoku, player: int, candidates: List[int]) -> List[int]:
        return [a for a in candidates if self._is_winning_move(env, a, player)]

    def _is_winning_move(self, env: Gomoku, action: int, player: int) -> bool:
        # 这里的逻辑建议直接调用 env 内部的高效判断，如果 env 没有，则保留这段 Python 实现
        size = env.size
        win_len = env.win_len
        r, c = divmod(action, size)
        
        if env.board[r, c] != 0: return False

        # 简单的 4 方向判定
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            # 正向
            nr, nc = r + dr, c + dc
            while 0 <= nr < size and 0 <= nc < size and env.board[nr, nc] == player:
                count += 1
                nr += dr; nc += dc
            # 反向
            nr, nc = r - dr, c - dc
            while 0 <= nr < size and 0 <= nc < size and env.board[nr, nc] == player:
                count += 1
                nr -= dr; nc -= dc
            
            if count >= win_len:
                return True
        return False
