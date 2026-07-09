"""
Django 集成示例
===============

将此代码放入 Django 项目中的 utils/az_client.py 或类似位置。
"""

from __future__ import annotations

import httpx
from typing import Optional
from dataclasses import dataclass


@dataclass
class AIMoveResult:
    """AI 落子结果"""
    action: int          # 落子位置 (0-224)
    row: int             # 行
    col: int             # 列
    value: float         # 局势评估 (-1 到 1, 黑子视角)
    confidence: float    # 置信度
    time_ms: float       # 推理耗时


class AlphaZeroClient:
    """
    AlphaZero API 客户端

    使用方法:
    ```python
    # 在 Django settings.py 中配置
    AZERO_API_URL = "http://localhost:8001"
    AZERO_DEFAULT_SIMS = 800

    # 在 views.py 中使用
    client = AlphaZeroClient(settings.AZERO_API_URL)

    # 使用会话模式（推荐）
    session_id = str(uuid.uuid4())
    result = client.get_move(board, player=1, session_id=session_id, reset_session=True)
    # 后续请求...
    result = client.get_move(board, player=-1, session_id=session_id)
    # 游戏结束时清理
    client.delete_session(session_id)
    ```
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    def close(self):
        self._client.close()

    def health_check(self) -> dict:
        """检查服务健康状态"""
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def get_move(
        self,
        board: list[int],
        player: int = 1,
        session_id: Optional[str] = None,
        reset_session: bool = False,
        model_id: Optional[str] = None,
        profile: Optional[str] = None,
        sims: Optional[int] = None,
        temp: float = 0.0,
    ) -> AIMoveResult:
        """
        获取 AI 推荐落子

        参数:
            board: 15x15 棋盘一维数组 (225个元素)
                   1 = 黑子, -1 = 白子, 0 = 空位
            player: 当前玩家 (1=黑子, -1=白子)
            session_id: 会话ID，用于复用 MCTS 搜索树（推荐使用）
            reset_session: 新游戏时设为 True，清空旧状态
            model_id: 指定模型 ID
            profile: 预设配置 (fast/standard/strong/analysis)
            sims: MCTS 模拟次数，None 使用服务端默认值
            temp: 温度参数，0=贪婪，>0=采样

        返回:
            AIMoveResult 对象

        会话模式说明:
            - 使用 session_id 可以复用 MCTS 搜索树，AI 更强更稳定
            - 新游戏时设置 reset_session=True
            - 游戏结束时调用 delete_session() 释放资源
        """
        payload = {
            "board": board,
            "player": player,
            "temp": temp,
        }
        if session_id is not None:
            payload["session_id"] = session_id
            payload["reset_session"] = reset_session
        if model_id is not None:
            payload["model_id"] = model_id
        if profile is not None:
            payload["profile"] = profile
        if sims is not None:
            payload["sims"] = sims

        resp = self._client.post(f"{self.base_url}/api/move", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return AIMoveResult(
            action=data["action"],
            row=data["row"],
            col=data["col"],
            value=data["value"],
            confidence=data["confidence"],
            time_ms=data["time_ms"],
        )

    def delete_session(self, session_id: str) -> dict:
        """
        删除会话，释放资源

        游戏结束时调用此方法释放 MCTS 树占用的内存。
        """
        resp = self._client.delete(f"{self.base_url}/api/session/{session_id}")
        resp.raise_for_status()
        return resp.json()

    def analyze(
        self,
        board: list[int],
        player: int = 1,
        top_n: int = 10,
        model_id: Optional[str] = None,
        profile: Optional[str] = None,
        sims: Optional[int] = None,
    ) -> dict:
        """
        分析当前局面

        返回局势评估和候选着法列表
        """
        payload = {
            "board": board,
            "player": player,
            "top_n": top_n,
        }
        if model_id is not None:
            payload["model_id"] = model_id
        if profile is not None:
            payload["profile"] = profile
        if sims is not None:
            payload["sims"] = sims
        resp = self._client.post(f"{self.base_url}/api/analyze", json=payload)
        resp.raise_for_status()
        return resp.json()

    def quick_evaluate(
        self,
        board: list[int],
        player: int = 1,
        model_id: Optional[str] = None,
    ) -> dict:
        """
        快速评估 (仅神经网络，无 MCTS)

        适合需要快速判断局势的场景
        """
        payload = {
            "board": board,
            "player": player,
        }
        if model_id is not None:
            payload["model_id"] = model_id
        resp = self._client.post(f"{self.base_url}/api/evaluate", json=payload)
        resp.raise_for_status()
        return resp.json()

    def validate_move(self, board: list[int], action: int) -> bool:
        """校验着法是否合法"""
        payload = {"board": board, "action": action}
        resp = self._client.post(f"{self.base_url}/api/validate", json=payload)
        resp.raise_for_status()
        return resp.json()["valid"]

    def get_game_state(self, board: list[int]) -> dict:
        """获取游戏状态 (是否结束、胜者等)"""
        resp = self._client.post(
            f"{self.base_url}/api/state",
            json={"board": board}
        )
        resp.raise_for_status()
        return resp.json()


# ============================================================================
# Django View 示例
# ============================================================================

"""
# views.py

import uuid
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings

from .utils.az_client import AlphaZeroClient
from .models import Game, GameMove


@method_decorator(csrf_exempt, name='dispatch')
class AIGameView(View):
    '''
    对战 API 示例（使用会话模式）
    '''

    def post(self, request):
        import json
        data = json.loads(request.body)

        game_id = data.get('game_id')
        action = data.get('action')  # 玩家落子
        session_id = data.get('session_id')  # 会话ID（前端保存并传回）
        is_new_game = data.get('new_game', False)

        game = Game.objects.get(id=game_id)

        # 生成或复用会话ID
        if session_id is None or is_new_game:
            session_id = str(uuid.uuid4())

        # 更新棋盘
        board = game.board  # 假设存储为 list
        board[action] = 1   # 玩家是黑子

        # 记录玩家落子
        GameMove.objects.create(game=game, player='human', position=action)

        # 检查玩家是否获胜
        client = AlphaZeroClient(settings.AZERO_API_URL)
        state = client.get_game_state(board)
        if state['game_over']:
            game.winner = state['winner']
            game.save()
            # 清理会话
            client.delete_session(session_id)
            return JsonResponse({
                'game_over': True,
                'winner': state['winner'],
                'session_id': session_id,
            })

        # AI 落子（使用会话模式复用 MCTS 树）
        ai_result = client.get_move(
            board,
            player=-1,
            session_id=session_id,
            reset_session=is_new_game,
            sims=settings.AZERO_DEFAULT_SIMS
        )

        # 更新棋盘
        board[ai_result.action] = -1
        game.board = board

        # 记录 AI 落子
        GameMove.objects.create(
            game=game,
            player='ai',
            position=ai_result.action,
            value=ai_result.value,
        )

        # 检查 AI 是否获胜
        state = client.get_game_state(board)
        if state['game_over']:
            game.winner = state['winner']
            game.save()
            # 清理会话
            client.delete_session(session_id)

        game.save()

        return JsonResponse({
            'ai_move': {
                'action': ai_result.action,
                'row': ai_result.row,
                'col': ai_result.col,
            },
            'value': ai_result.value,
            'game_over': state['game_over'],
            'winner': state.get('winner'),
            'session_id': session_id,  # 返回给前端保存
        })
"""


# ============================================================================
# 异步客户端示例 (适用于 Django 异步视图)
# ============================================================================

class AsyncAlphaZeroClient:
    """
    异步客户端，适用于高并发场景
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def get_move(
        self,
        board: list[int],
        player: int = 1,
        session_id: Optional[str] = None,
        reset_session: bool = False,
        model_id: Optional[str] = None,
        profile: Optional[str] = None,
        sims: Optional[int] = None,
        temp: float = 0.0,
    ) -> AIMoveResult:
        payload = {"board": board, "player": player, "temp": temp}
        if session_id is not None:
            payload["session_id"] = session_id
            payload["reset_session"] = reset_session
        if model_id is not None:
            payload["model_id"] = model_id
        if profile is not None:
            payload["profile"] = profile
        if sims is not None:
            payload["sims"] = sims

        resp = await self._client.post(f"{self.base_url}/api/move", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return AIMoveResult(
            action=data["action"],
            row=data["row"],
            col=data["col"],
            value=data["value"],
            confidence=data["confidence"],
            time_ms=data["time_ms"],
        )

    async def delete_session(self, session_id: str) -> dict:
        """删除会话，释放资源"""
        resp = await self._client.delete(f"{self.base_url}/api/session/{session_id}")
        resp.raise_for_status()
        return resp.json()


# ============================================================================
# 简单测试
# ============================================================================

if __name__ == "__main__":
    # 测试客户端
    import time

    client = AlphaZeroClient("http://localhost:8001")

    # 健康检查
    print("Health check:")
    print(client.health_check())

    # 空棋盘
    empty_board = [0] * 225

    # 测试会话模式
    print("\n=== 测试会话模式 ===")
    session_id = "test-session-1"

    # 新游戏
    print("\nAI move on empty board (session mode):")
    start = time.time()
    result = client.get_move(
        empty_board,
        player=1,
        session_id=session_id,
        reset_session=True,  # 新游戏
        sims=200
    )
    print(f"  Action: {result.action} ({result.row}, {result.col})")
    print(f"  Value: {result.value:.4f}")
    print(f"  Time: {result.time_ms:.1f}ms")

    # 模拟对手落子
    board = empty_board.copy()
    board[result.action] = 1  # AI 落子
    board[112] = -1  # 对手中心

    # 后续请求（复用树）
    print("\nAI move after opponent (tree reused):")
    result2 = client.get_move(
        board,
        player=1,
        session_id=session_id,
        sims=200
    )
    print(f"  Action: {result2.action} ({result2.row}, {result2.col})")
    print(f"  Time: {result2.time_ms:.1f}ms")

    # 清理会话
    client.delete_session(session_id)
    print("\nSession deleted.")

    # 测试无状态模式
    print("\n=== 测试无状态模式 ===")
    print("\nAI move (stateless mode):")
    result3 = client.get_move(empty_board, player=1, sims=200)
    print(f"  Action: {result3.action} ({result3.row}, {result3.col})")
    print(f"  Time: {result3.time_ms:.1f}ms")
