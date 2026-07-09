"""
AlphaZero Gomoku API Service
============================

高性能 FastAPI 服务，供 Django 或其他服务调用。

特性:
- 异步 API 支持
- 自动 OpenAPI 文档
- 模型预加载和热重载
- 批量推理支持
- 健康检查端点
- 性能监控

启动方式:
    uv run src/api_service.py --host 0.0.0.0 --port 8001 --model artifacts/az_model_best.pt

或者使用 gunicorn (生产环境):
    gunicorn src.api_service:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from az_mcts import AZMCTS
from az_net import AZNet, encode_board
from env import BOARD_AREA, BOARD_SIZE, Gomoku


# ============================================================================
# 配置和全局状态
# ============================================================================

class ModelConfig:
    """单个模型的配置"""
    sims: int = 800
    c_puct: float = 5.0
    prune_radius: int = 2

    def __init__(self, sims=800, c_puct=5.0, prune_radius=2):
        self.sims = sims
        self.c_puct = c_puct
        self.prune_radius = prune_radius


class GameSession:
    """游戏会话 - 支持树复用"""
    def __init__(self, model: AZNet, config: ModelConfig, device: torch.device):
        self.env = Gomoku()
        self.model = model
        self.config = config
        self.device = device
        self.mcts = AZMCTS(
            model=model,
            sims=config.sims,
            c_puct=config.c_puct,
            prune_radius=config.prune_radius,
            device=device,
        )
        self.last_board: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None  # 上一次 AI 推荐的落子

    def reset(self):
        """重置游戏状态"""
        self.env.reset()
        self.mcts.root = None
        self.mcts.root_key = None
        self.last_board = None
        self.last_action = None

    def sync_board(self, board: np.ndarray, player: int):
        """
        同步棋盘状态，尝试复用MCTS树。

        策略：
        1. 如果 last_board 为空或 reset，重建
        2. 如果 board == last_board，不做任何事（状态已同步）
        3. 如果只有一步差异，advance 复用树
        4. 否则重建
        """
        if self.last_board is None:
            # 新会话，重建
            self._rebuild(board)
            return

        if np.array_equal(board, self.last_board):
            # 状态相同，无需同步
            return

        # 检测差异
        diff = board - self.last_board
        nonzero = np.nonzero(diff.ravel())[0]

        if len(nonzero) == 1:
            # 只有一步差异
            action = nonzero[0]
            # 验证是合法落子（从空变非空）
            if self.last_board.ravel()[action] == 0 and board.ravel()[action] != 0:
                # 这一步是对手下的（因为 AI 推荐的落子已经在 last_board 中了）
                # advance 树
                self.env.step(action, board.ravel()[action])
                self.mcts.advance(action, self.env)
                self.last_board = board.copy()
                return

        # 多步差异或非法变化，重建
        self._rebuild(board)

    def _rebuild(self, board: np.ndarray):
        """重建环境和树"""
        self.env.reset()
        self.mcts.root = None
        self.mcts.root_key = None
        for action in range(BOARD_AREA):
            r, c = divmod(action, BOARD_SIZE)
            if board[r, c] != 0:
                self.env.step(action, board[r, c])
        self.last_board = board.copy()
        self.last_action = None


# 预设配置 profiles
PROFILE_DEFAULTS = {
    "fast": ModelConfig(sims=100),
    "standard": ModelConfig(sims=400),
    "strong": ModelConfig(sims=800),
    "analysis": ModelConfig(sims=1600),
}


class Config:
    """全局配置"""
    model_path: str = "artifacts/az_model_best.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    default_profile: str = "standard"


# 会话管理
class SessionRegistry:
    """游戏会话注册表 - 支持树复用"""
    sessions: dict[str, GameSession] = {}
    max_sessions: int = 1000  # 最大会话数

    @classmethod
    def get_or_create(cls, session_id: str, model_id: str = None) -> GameSession:
        """获取或创建会话"""
        if session_id in cls.sessions:
            return cls.sessions[session_id]

        model = ModelRegistry.get_model(model_id)
        if model is None:
            raise ValueError(f"Model not loaded: {model_id or ModelRegistry.default_model_id}")

        config = ModelRegistry.get_config(model_id)
        device = torch.device(Config.device)

        # 清理过期会话
        if len(cls.sessions) >= cls.max_sessions:
            # 删除最早的会话
            oldest = next(iter(cls.sessions))
            del cls.sessions[oldest]

        session = GameSession(model, config, device)
        cls.sessions[session_id] = session
        return session

    @classmethod
    def delete(cls, session_id: str):
        """删除会话"""
        if session_id in cls.sessions:
            del cls.sessions[session_id]


class ModelRegistry:
    """模型注册表 - 支持多模型"""
    models: dict[str, AZNet] = {}
    model_configs: dict[str, ModelConfig] = {}  # model_id -> config
    loaded_at: dict[str, datetime] = {}
    default_model_id: str = "default"

    @classmethod
    def get_model(cls, model_id: str = None) -> Optional[AZNet]:
        """获取模型"""
        if model_id is None:
            model_id = cls.default_model_id
        return cls.models.get(model_id)

    @classmethod
    def get_config(cls, model_id: str = None) -> ModelConfig:
        """获取模型配置"""
        if model_id is None:
            model_id = cls.default_model_id
        return cls.model_configs.get(model_id, ModelConfig())

    @classmethod
    def load_model(cls, path: str, device: str, model_id: str = None,
                   config: ModelConfig = None) -> bool:
        """加载模型到注册表"""
        if not os.path.exists(path):
            return False
        try:
            model = AZNet().to(torch.device(device))
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()

            if model_id is None:
                model_id = cls.default_model_id

            cls.models[model_id] = model
            cls.model_configs[model_id] = config or ModelConfig()
            cls.loaded_at[model_id] = datetime.now()

            print(f"Model loaded: {model_id} = {path} on {device}")
            return True
        except Exception as e:
            print(f"Model load error: {e}")
            return False

    @classmethod
    def unload_model(cls, model_id: str):
        """卸载模型"""
        if model_id in cls.models:
            del cls.models[model_id]
            if model_id in cls.model_configs:
                del cls.model_configs[model_id]
            if model_id in cls.loaded_at:
                del cls.loaded_at[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def list_models(cls) -> list[dict]:
        """列出所有已加载的模型"""
        result = []
        for model_id, model in cls.models.items():
            config = cls.model_configs.get(model_id, ModelConfig())
            result.append({
                "model_id": model_id,
                "loaded_at": cls.loaded_at.get(model_id).isoformat() if model_id in cls.loaded_at else None,
                "sims": config.sims,
                "c_puct": config.c_puct,
            })
        return result


# 兼容旧代码
class ModelState:
    """模型状态管理 (兼容层)"""
    model: Optional[AZNet] = None
    model_loaded_at: Optional[datetime] = None
    model_file: Optional[str] = None

    @classmethod
    def load_model(cls, path: str, device: str) -> bool:
        """加载模型 (兼容旧接口)"""
        success = ModelRegistry.load_model(path, device, ModelRegistry.default_model_id)
        if success:
            cls.model = ModelRegistry.get_model()
            cls.model_loaded_at = datetime.now()
            cls.model_file = path
        return success


# ============================================================================
# Pydantic 模型 (请求/响应)
# ============================================================================

class MoveRequest(BaseModel):
    """获取 AI 落子建议"""
    board: list[int] = Field(
        ...,
        description="15x15 棋盘一维数组，225个元素。1=黑子，-1=白子，0=空位",
        json_schema_extra={"example": [0] * 225}
    )
    player: int = Field(
        ...,
        ge=-1, le=1,
        description="当前玩家: 1=黑子(X), -1=白子(O)"
    )
    session_id: Optional[str] = Field(
        None,
        description="会话ID，用于复用MCTS搜索树。不传则无状态模式"
    )
    reset_session: bool = Field(
        False,
        description="是否重置会话（新游戏时设为 true）"
    )
    model_id: Optional[str] = Field(
        None,
        description="指定模型 ID，不指定则使用默认模型"
    )
    profile: Optional[str] = Field(
        None,
        description="预设配置: fast(100), standard(400), strong(800), analysis(1600)"
    )
    sims: Optional[int] = Field(
        None,
        ge=1, le=4000,
        description="MCTS 模拟次数，优先级高于 profile"
    )
    temp: float = Field(
        0.0,
        ge=0.0, le=2.0,
        description="温度参数，0=贪婪选择，>0=采样"
    )


class MoveResponse(BaseModel):
    """AI 落子响应"""
    action: int = Field(..., description="推荐的落子位置 (0-224)")
    row: int = Field(..., description="行索引 (0-14)")
    col: int = Field(..., description="列索引 (0-14)")
    value: float = Field(..., description="局势评估值，黑子视角 (-1到1)")
    policy: list[float] = Field(..., description="完整策略分布 (225个概率值)")
    confidence: float = Field(..., description="AI 对推荐着法的置信度")
    top_moves: list[dict] = Field(..., description="Top 5 候选着法")
    sims: int = Field(..., description="实际使用的模拟次数")
    time_ms: float = Field(..., description="推理耗时 (毫秒)")


class AnalyzeRequest(BaseModel):
    """局面分析请求"""
    board: list[int] = Field(..., description="15x15 棋盘一维数组")
    player: int = Field(..., ge=-1, le=1, description="当前玩家")
    top_n: int = Field(10, ge=1, le=50, description="返回的候选着法数量")
    model_id: Optional[str] = Field(None, description="指定模型 ID")
    profile: Optional[str] = Field(None, description="预设配置")
    sims: Optional[int] = Field(None, ge=1, le=4000, description="MCTS 模拟次数")


class AnalyzeResponse(BaseModel):
    """局面分析响应"""
    value: float = Field(..., description="局势评估，黑子视角 (-1到1)")
    policy: list[float] = Field(..., description="完整策略分布")
    top_moves: list[dict] = Field(..., description="候选着法列表")
    game_over: bool = Field(..., description="游戏是否结束")
    winner: Optional[int] = Field(None, description="胜者: 1=黑子, -1=白子, 0=平局, None=未结束")
    time_ms: float = Field(..., description="分析耗时 (毫秒)")


class EvaluateRequest(BaseModel):
    """快速评估请求 (仅神经网络，无 MCTS)"""
    board: list[int] = Field(..., description="15x15 棋盘一维数组")
    player: int = Field(..., ge=-1, le=1, description="当前玩家")
    model_id: Optional[str] = Field(None, description="指定模型 ID")


class EvaluateResponse(BaseModel):
    """快速评估响应"""
    value: float = Field(..., description="局势评估，黑子视角 (-1到1)")
    policy: list[float] = Field(..., description="神经网络输出的策略")
    legal_moves: int = Field(..., description="合法着法数量")
    time_ms: float = Field(..., description="推理耗时 (毫秒)")


class ValidateRequest(BaseModel):
    """校验着法请求"""
    board: list[int] = Field(..., description="15x15 棋盘一维数组")
    action: int = Field(..., ge=0, lt=BOARD_AREA, description="落子位置")


class ValidateResponse(BaseModel):
    """校验着法响应"""
    valid: bool = Field(..., description="着法是否合法")
    reason: Optional[str] = Field(None, description="如果不合法，说明原因")


class GameStateRequest(BaseModel):
    """游戏状态查询"""
    board: list[int] = Field(..., description="15x15 棋盘一维数组")


class GameStateResponse(BaseModel):
    """游戏状态响应"""
    game_over: bool = Field(..., description="游戏是否结束")
    winner: Optional[int] = Field(None, description="胜者")
    move_count: int = Field(..., description="已下棋子数")
    next_player: int = Field(..., description="下一个玩家")


class BatchMoveRequest(BaseModel):
    """批量推理请求"""
    positions: list[MoveRequest] = Field(
        ...,
        max_length=32,
        description="批量位置列表 (最多32个)"
    )
    model_id: Optional[str] = Field(None, description="指定模型 ID")
    profile: Optional[str] = Field(None, description="预设配置")
    sims: Optional[int] = Field(None, ge=1, le=4000, description="MCTS 模拟次数")


class BatchMoveResponse(BaseModel):
    """批量推理响应"""
    results: list[MoveResponse] = Field(..., description="每个位置的推理结果")
    total_time_ms: float = Field(..., description="总耗时")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态: ok/error")
    models_loaded: list[dict] = Field(..., description="已加载的模型列表")
    default_model_id: str = Field(..., description="默认模型 ID")
    device: str = Field(..., description="运行设备")
    gpu_available: bool = Field(..., description="GPU 是否可用")
    gpu_name: Optional[str] = Field(None, description="GPU 名称")


class ModelInfo(BaseModel):
    """模型信息"""
    model_id: str = Field(..., description="模型标识")
    model_path: Optional[str] = Field(None, description="模型文件路径")
    loaded_at: Optional[str] = Field(None, description="加载时间")
    sims: int = Field(..., description="默认模拟次数")
    c_puct: float = Field(..., description="PUCT 探索常数")


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    model_path: Optional[str] = Field(None, description="新模型路径")
    model_id: Optional[str] = Field(None, description="模型标识，不指定则使用默认")
    default_sims: Optional[int] = Field(None, ge=1, le=4000, description="默认模拟次数")
    c_puct: Optional[float] = Field(None, ge=0.1, le=20.0, description="PUCT 探索常数")
    device: Optional[str] = Field(None, description="运行设备")
    set_default: Optional[bool] = Field(None, description="设为默认模型")


class LoadModelRequest(BaseModel):
    """加载模型请求"""
    model_path: str = Field(..., description="模型文件路径")
    model_id: Optional[str] = Field(None, description="模型标识，不指定则生成")
    profile: Optional[str] = Field(None, description="使用预设配置")
    sims: Optional[int] = Field(None, ge=1, le=4000, description="自定义模拟次数")
    c_puct: Optional[float] = Field(None, ge=0.1, le=20.0, description="自定义 PUCT")
    set_default: bool = Field(False, description="设为默认模型")


class ConfigResponse(BaseModel):
    """配置响应"""
    model_path: str
    model_id: str
    models_loaded: list[dict]
    device: str
    default_profile: str
    available_profiles: dict[str, int]


# ============================================================================
# 辅助函数
# ============================================================================

def parse_board(board_data: list[int]) -> np.ndarray:
    """解析棋盘数据为 numpy 数组"""
    if len(board_data) != BOARD_AREA:
        raise ValueError(f"Board must have {BOARD_AREA} elements, got {len(board_data)}")
    return np.array(board_data, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE)


def get_top_moves(policy: list[float], board: np.ndarray, top_n: int = 5) -> list[dict]:
    """获取 Top N 候选着法"""
    # 过滤非法着法
    legal_mask = (board.ravel() == 0)
    legal_policy = [(i, p) for i, p in enumerate(policy) if legal_mask[i]]
    legal_policy.sort(key=lambda x: x[1], reverse=True)

    top_moves = []
    for action, prob in legal_policy[:top_n]:
        row, col = divmod(action, BOARD_SIZE)
        top_moves.append({
            "action": action,
            "row": row,
            "col": col,
            "probability": round(prob, 6),
        })
    return top_moves


def create_env_from_board(board: np.ndarray) -> Gomoku:
    """从棋盘数据创建游戏环境"""
    env = Gomoku()
    env.reset()
    # 重新设置棋盘状态
    for action in range(BOARD_AREA):
        r, c = divmod(action, BOARD_SIZE)
        if board[r, c] != 0:
            env.step(action, board[r, c])
    return env


# ============================================================================
# FastAPI 应用
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载默认模型
    ModelRegistry.load_model(Config.model_path, Config.device)
    print(f"Default model loaded: {Config.model_path} on {Config.device}")
    yield
    # 关闭时清理
    for model_id in list(ModelRegistry.models.keys()):
        ModelRegistry.unload_model(model_id)


app = FastAPI(
    title="AlphaZero Gomoku API",
    description="五子棋 AI 推理服务，提供 MCTS 搜索和神经网络评估",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件 (允许 Django 跨域调用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API 端点
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """服务根路径"""
    return {
        "service": "AlphaZero Gomoku API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """健康检查端点"""
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return HealthResponse(
        status="ok" if ModelRegistry.models else "error",
        models_loaded=ModelRegistry.list_models(),
        default_model_id=ModelRegistry.default_model_id,
        device=Config.device,
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
    )


@app.get("/config", response_model=ConfigResponse, tags=["System"])
async def get_config():
    """获取当前配置"""
    return ConfigResponse(
        model_path=Config.model_path,
        model_id=ModelRegistry.default_model_id,
        models_loaded=ModelRegistry.list_models(),
        device=Config.device,
        default_profile=Config.default_profile,
        available_profiles={k: v.sims for k, v in PROFILE_DEFAULTS.items()},
    )


@app.post("/config", response_model=ConfigResponse, tags=["System"])
async def update_config(req: ConfigUpdateRequest):
    """更新配置"""
    if req.device is not None:
        Config.device = req.device

    # 更新默认模型配置
    if req.model_id and req.model_id in ModelRegistry.model_configs:
        config = ModelRegistry.model_configs[req.model_id]
        if req.default_sims is not None:
            config.sims = req.default_sims
        if req.c_puct is not None:
            config.c_puct = req.c_puct

    if req.set_default and req.model_id:
        ModelRegistry.default_model_id = req.model_id

    return ConfigResponse(
        model_path=Config.model_path,
        model_id=ModelRegistry.default_model_id,
        models_loaded=ModelRegistry.list_models(),
        device=Config.device,
        default_profile=Config.default_profile,
        available_profiles={k: v.sims for k, v in PROFILE_DEFAULTS.items()},
    )


@app.post("/models/load", response_model=ModelInfo, tags=["System"])
async def load_model(req: LoadModelRequest):
    """加载新模型"""
    # 构建 config
    if req.profile and req.profile in PROFILE_DEFAULTS:
        config = ModelConfig(
            sims=PROFILE_DEFAULTS[req.profile].sims,
            c_puct=PROFILE_DEFAULTS[req.profile].c_puct,
        )
    else:
        config = ModelConfig()

    if req.sims is not None:
        config.sims = req.sims
    if req.c_puct is not None:
        config.c_puct = req.c_puct

    # 生成 model_id
    model_id = req.model_id
    if model_id is None:
        # 基于文件名生成
        model_id = Path(req.model_path).stem
        if model_id in ModelRegistry.models:
            # 添加序号避免冲突
            i = 1
            while f"{model_id}_{i}" in ModelRegistry.models:
                i += 1
            model_id = f"{model_id}_{i}"

    # 加载模型
    if not ModelRegistry.load_model(req.model_path, Config.device, model_id, config):
        raise HTTPException(status_code=400, detail=f"Failed to load model: {req.model_path}")

    if req.set_default:
        ModelRegistry.default_model_id = model_id
        Config.model_path = req.model_path

    loaded_at = ModelRegistry.loaded_at.get(model_id)
    return ModelInfo(
        model_id=model_id,
        model_path=req.model_path,
        loaded_at=loaded_at.isoformat() if loaded_at else None,
        sims=config.sims,
        c_puct=config.c_puct,
    )


@app.get("/models", response_model=list[ModelInfo], tags=["System"])
async def list_models():
    """列出所有已加载的模型"""
    result = []
    for model_id in ModelRegistry.models:
        config = ModelRegistry.model_configs.get(model_id, ModelConfig())
        loaded_at = ModelRegistry.loaded_at.get(model_id)
        result.append(ModelInfo(
            model_id=model_id,
            model_path=None,  # 不追踪路径
            loaded_at=loaded_at.isoformat() if loaded_at else None,
            sims=config.sims,
            c_puct=config.c_puct,
        ))
    return result


@app.delete("/models/{model_id}", tags=["System"])
async def unload_model(model_id: str):
    """卸载模型"""
    if model_id not in ModelRegistry.models:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    if model_id == ModelRegistry.default_model_id:
        raise HTTPException(status_code=400, detail="Cannot unload default model")

    ModelRegistry.unload_model(model_id)
    return {"status": "ok", "unloaded": model_id}


@app.post("/api/move", response_model=MoveResponse, tags=["AI"])
async def get_move(req: MoveRequest):
    """
    获取 AI 推荐着法

    这是最核心的 API，Django 调用此接口获取 AI 的落子建议。

    **会话模式 vs 无状态模式**:
    - 提供 session_id 时使用会话模式，MCTS 搜索树会被复用，AI 更强
    - 不提供 session_id 时为无状态模式，每次请求独立搜索

    **模型选择优先级**: model_id > profile > sims

    请求示例:
    ```json
    {
        "board": [0, 0, ..., 0],  // 225个元素
        "player": 1,
        "session_id": "game-123",   // 可选，用于复用MCTS树
        "reset_session": true,      // 新游戏时设为 true
        "model_id": "az_model_v2",  // 可选
        "profile": "strong",        // 可选
        "sims": 600,                // 可选
        "temp": 0.0
    }
    ```

    响应包含推荐位置、局势评估、策略分布等信息。
    """
    # 获取模型
    model = ModelRegistry.get_model(req.model_id)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {req.model_id or ModelRegistry.default_model_id}")

    # 获取配置
    model_config = ModelRegistry.get_config(req.model_id)

    # 确定 sims: 显式指定 > profile > 模型默认
    sims = req.sims
    if sims is None and req.profile:
        if req.profile in PROFILE_DEFAULTS:
            sims = PROFILE_DEFAULTS[req.profile].sims
        else:
            raise HTTPException(status_code=400, detail=f"Unknown profile: {req.profile}. Available: {list(PROFILE_DEFAULTS.keys())}")
    if sims is None:
        sims = model_config.sims

    start_time = time.perf_counter()

    # 解析棋盘
    try:
        board = parse_board(req.board)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 检查游戏是否已结束
    env = create_env_from_board(board)
    winner = env.check_winner()
    if winner is not None:
        raise HTTPException(status_code=400, detail="Game already finished")

    # 获取当前玩家 (从 move_count 推断)
    actual_player = 1 if env.move_count % 2 == 0 else -1
    if req.player != actual_player:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid player: expected {actual_player}, got {req.player}"
        )

    # 会话模式：复用 MCTS 树
    use_session = req.session_id is not None
    mcts = None

    if use_session:
        try:
            session = SessionRegistry.get_or_create(req.session_id, req.model_id)
            # 重置会话或同步棋盘
            if req.reset_session:
                session.reset()
                session.sync_board(board, req.player)
            else:
                # 同步棋盘状态，尝试复用树
                session.sync_board(board, req.player)
            # 更新 sims（如果请求中指定了）
            if req.sims is not None:
                session.mcts.sims = sims
            mcts = session.mcts
            env = session.env
        except ValueError as e:
            raise HTTPException(status_code=503, detail=str(e))

    # 无状态模式：创建临时 MCTS
    if mcts is None:
        mcts = AZMCTS(
            model=model,
            sims=sims,
            c_puct=model_config.c_puct,
            prune_radius=model_config.prune_radius,
            device=torch.device(Config.device),
        )

    pi = mcts.run(env, player=req.player, add_noise=False)
    action = mcts.select_action(pi, temp=req.temp)

    # 会话模式：更新状态以便下次复用
    if use_session and session is not None:
        # 在 env 中落子并 advance 树
        session.env.step(action, req.player)
        session.mcts.advance(action, session.env)
        # 更新 last_board
        new_board = board.copy()
        r, c = divmod(action, BOARD_SIZE)
        new_board[r, c] = req.player
        session.last_board = new_board

    # 获取局势评估 (黑子视角)
    board_tensor = encode_board(board, req.player, device=torch.device(Config.device))
    with torch.no_grad():
        _, value = model(board_tensor.unsqueeze(0))
    value_x = float(value.item()) if req.player == 1 else -float(value.item())

    # 获取 Top moves
    top_moves = get_top_moves(pi, board, top_n=5)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    row, col = divmod(action, BOARD_SIZE)

    return MoveResponse(
        action=action,
        row=row,
        col=col,
        value=value_x,
        policy=pi,
        confidence=pi[action],
        top_moves=top_moves,
        sims=sims,
        time_ms=round(elapsed_ms, 2),
    )


@app.delete("/api/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """删除游戏会话，释放资源"""
    if session_id not in SessionRegistry.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    SessionRegistry.delete(session_id)
    return {"status": "ok", "deleted": session_id}


@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["AI"])
async def analyze_position(req: AnalyzeRequest):
    """
    分析当前局面

    返回局势评估和候选着法，但不给出具体推荐。
    适用于分析模式和教学场景。
    """
    model = ModelRegistry.get_model(req.model_id)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {req.model_id or ModelRegistry.default_model_id}")

    # 确定 sims
    model_config = ModelRegistry.get_config(req.model_id)
    sims = req.sims
    if sims is None and req.profile:
        if req.profile in PROFILE_DEFAULTS:
            sims = PROFILE_DEFAULTS[req.profile].sims
        else:
            raise HTTPException(status_code=400, detail=f"Unknown profile: {req.profile}")
    if sims is None:
        sims = model_config.sims

    start_time = time.perf_counter()

    try:
        board = parse_board(req.board)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    env = create_env_from_board(board)
    winner = env.check_winner()

    # 即使游戏结束也返回策略评估
    board_tensor = encode_board(board, req.player, device=torch.device(Config.device))
    with torch.no_grad():
        policy_logits, value = model(board_tensor.unsqueeze(0))
    policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy().tolist()
    value_x = float(value.item()) if req.player == 1 else -float(value.item())

    top_moves = get_top_moves(policy, board, top_n=req.top_n)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return AnalyzeResponse(
        value=value_x,
        policy=policy,
        top_moves=top_moves,
        game_over=winner is not None,
        winner=winner,
        time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/evaluate", response_model=EvaluateResponse, tags=["AI"])
async def quick_evaluate(req: EvaluateRequest):
    """
    快速评估 (仅神经网络，无 MCTS)

    速度最快，适合需要快速评估大量局面的场景。
    返回神经网络直接输出的策略和局势值。
    """
    model = ModelRegistry.get_model(req.model_id)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {req.model_id or ModelRegistry.default_model_id}")

    start_time = time.perf_counter()

    try:
        board = parse_board(req.board)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    board_tensor = encode_board(board, req.player, device=torch.device(Config.device))
    with torch.no_grad():
        policy_logits, value = model(board_tensor.unsqueeze(0))

    policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy().tolist()
    value_x = float(value.item()) if req.player == 1 else -float(value.item())
    legal_moves = int(np.sum(board == 0))

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return EvaluateResponse(
        value=value_x,
        policy=policy,
        legal_moves=legal_moves,
        time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/validate", response_model=ValidateResponse, tags=["Game"])
async def validate_move(req: ValidateRequest):
    """
    校验着法是否合法

    Django 可以在落子前调用此接口校验。
    """
    try:
        board = parse_board(req.board)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    row, col = divmod(req.action, BOARD_SIZE)

    if board[row, col] != 0:
        return ValidateResponse(valid=False, reason="Position already occupied")

    return ValidateResponse(valid=True, reason=None)


@app.post("/api/state", response_model=GameStateResponse, tags=["Game"])
async def get_game_state(req: GameStateRequest):
    """
    获取游戏状态

    返回游戏是否结束、胜者等信息。
    """
    try:
        board = parse_board(req.board)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    env = create_env_from_board(board)
    winner = env.check_winner()
    next_player = 1 if env.move_count % 2 == 0 else -1

    return GameStateResponse(
        game_over=winner is not None,
        winner=winner,
        move_count=env.move_count,
        next_player=next_player,
    )


@app.post("/api/batch", response_model=BatchMoveResponse, tags=["AI"])
async def batch_move(req: BatchMoveRequest):
    """
    批量推理接口

    适用于需要同时评估多个局面的场景，提高 GPU 利用率。
    最多支持 32 个位置同时推理。
    """
    # 确定全局配置（可被单个请求覆盖）
    global_model_id = req.model_id
    global_sims = req.sims
    if global_sims is None and req.profile:
        if req.profile in PROFILE_DEFAULTS:
            global_sims = PROFILE_DEFAULTS[req.profile].sims
        else:
            raise HTTPException(status_code=400, detail=f"Unknown profile: {req.profile}")

    model = ModelRegistry.get_model(global_model_id)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {global_model_id or ModelRegistry.default_model_id}")

    start_time = time.perf_counter()
    results = []

    # 串行处理 MCTS (MCTS 不适合批处理)
    for pos in req.positions:
        # 单个请求可以覆盖全局设置
        pos_model = ModelRegistry.get_model(pos.model_id) if pos.model_id else model
        if pos_model is None:
            pos_model = model

        # sims 优先级: 请求指定 > 全局指定 > 模型默认
        pos_sims = pos.sims or global_sims
        if pos_sims is None:
            pos_sims = ModelRegistry.get_config(pos.model_id or global_model_id).sims

        # 构建带覆盖参数的请求
        modified_pos = MoveRequest(
            board=pos.board,
            player=pos.player,
            model_id=pos.model_id or global_model_id,
            profile=pos.profile,
            sims=pos_sims,
            temp=pos.temp,
        )
        try:
            result = await get_move(modified_pos)
            results.append(result)
        except HTTPException as e:
            raise e

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return BatchMoveResponse(
        results=results,
        total_time_ms=round(elapsed_ms, 2),
    )


# ============================================================================
# 启动入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Gomoku API Service")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    parser.add_argument("--model", default="artifacts/az_model_best.pt", help="Default model path")
    parser.add_argument("--device", default="auto", help="Device: cpu, cuda, cuda:0, auto")
    parser.add_argument("--sims", type=int, default=800, help="Default MCTS simulations")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")

    # 多模型支持
    parser.add_argument("--models", nargs="*", help="Additional models: model_id=path[:profile] (e.g., strong=artifacts/v2.pt:strong)")
    parser.add_argument("--default-profile", default="standard",
                        choices=list(PROFILE_DEFAULTS.keys()),
                        help="Default profile for auto-loaded models")

    args = parser.parse_args()

    # 设置配置
    Config.model_path = args.model
    Config.default_profile = args.default_profile

    if args.device == "auto":
        Config.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        Config.device = args.device

    # 加载默认模型
    default_config = ModelConfig(sims=args.sims)
    ModelRegistry.load_model(args.model, Config.device, ModelRegistry.default_model_id, default_config)

    # 加载额外模型
    if args.models:
        for model_spec in args.models:
            parts = model_spec.split("=", 1)
            if len(parts) == 2:
                model_id, path_profile = parts
                path_parts = path_profile.split(":")
                model_path = path_parts[0]

                # 确定配置
                config = ModelConfig()
                if len(path_parts) > 1 and path_parts[1] in PROFILE_DEFAULTS:
                    config.sims = PROFILE_DEFAULTS[path_parts[1]].sims
                else:
                    config.sims = PROFILE_DEFAULTS[Config.default_profile].sims

                ModelRegistry.load_model(model_path, Config.device, model_id, config)
            else:
                print(f"Warning: Invalid model spec '{model_spec}', expected format: model_id=path[:profile]")

    # 启动服务
    import uvicorn
    uvicorn.run(
        "api_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
