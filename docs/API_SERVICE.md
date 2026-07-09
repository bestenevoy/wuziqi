# AlphaZero Gomoku API 服务文档

## 概述

这是一个高性能的 FastAPI 服务，为 Django 或其他后端提供五子棋 AI 推理能力。

### 架构图

```
┌─────────────┐     HTTP/REST     ┌─────────────────┐     GPU/CPU     ┌─────────────┐
│   Django    │ ───────────────→ │  FastAPI 服务    │ ─────────────→ │  PyTorch    │
│   (业务层)   │ ←─────────────── │  (AI推理层)      │ ←───────────── │  模型推理    │
└─────────────┘                   └─────────────────┘                 └─────────────┘
     :8000                              :8001
```

### 优势

1. **解耦部署**: AI 服务独立部署，可单独扩展 GPU 资源
2. **FastAPI 高性能**: 异步支持、自动 OpenAPI 文档、类型校验
3. **多级推理**: MCTS 深度搜索 / 快速神经网络评估
4. **热重载**: 支持运行时更换模型
5. **多模型支持**: 同时加载多个模型，按需选择不同强度
6. **会话模式**: 支持 MCTS 树复用，AI 更强更稳定

---

## 快速开始

### 1. 安装依赖

```bash
cd /home/wrz/code/wuziqi
uv sync
```

### 2. 启动服务

```bash
# 开发模式 (单模型)
uv run python src/api_service.py --host 0.0.0.0 --port 8001 --model artifacts/az_model_best.pt --device cuda:3

# 多模型启动
uv run python src/api_service.py --host 0.0.0.0 --port 8001 \
    --model artifacts/az_model_best.pt \
    --models "fast=artifacts/az_model_fast.pt:fast" "strong=artifacts/az_model_v2.pt:strong"

# 生产模式 (使用 gunicorn)
gunicorn src.api_service:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8001 \
    --timeout 120
```

### 3. 访问文档

- OpenAPI 文档: http://localhost:8001/docs
- ReDoc 文档: http://localhost:8001/redoc

---

## 预设配置 (Profiles)

服务内置了四种预设配置，通过 `profile` 参数快速选择 AI 强度：

| Profile | sims | 说明 |
|---------|------|------|
| `fast` | 100 | 快速响应，适合在线对战快速模式 |
| `standard` | 400 | 标准强度，平衡速度和质量 |
| `strong` | 800 | 强力模式，适合重要对局 |
| `analysis` | 1600 | 分析模式，最强 AI |

**注意**: 默认 sims 为 800，相当于 `strong` 级别。

---

## 会话模式（重要）

### 为什么需要会话模式？

MCTS 搜索树可以复用！当对手落子后，之前搜索过的子树仍然有效。使用会话模式：

- **AI 更强**: 累积搜索结果，决策质量更高
- **更稳定**: 复用树避免了从零开始的不确定性
- **推荐使用**: 对于连续对局，强烈建议使用会话模式

### 无状态模式 vs 会话模式

| 特性 | 无状态模式 | 会话模式 |
|------|-----------|---------|
| MCTS 树 | 每次重建 | 复用累积 |
| AI 强度 | 标准 | 更强 |
| 内存占用 | 低 | 每会话约 10-50MB |
| 适用场景 | 单次请求 | 连续对局 |

### 会话模式使用方法

```json
// 新游戏开始
POST /api/move
{
    "board": [0, 0, ..., 0],
    "player": 1,
    "session_id": "game-123",      // 会话ID，任意字符串
    "reset_session": true          // 标记为新游戏
}

// 后续请求（对手落子后）
POST /api/move
{
    "board": [1, 0, ..., 0],       // 包含对手的新落子
    "player": -1,
    "session_id": "game-123"       // 相同的 session_id
    // 不需要 reset_session
}
```

---

## API 接口

### 核心接口

#### POST /api/move - 获取 AI 落子

**最核心的接口**，Django 调用此接口获取 AI 的落子建议。

**请求:**
```json
{
    "board": [0, 0, 0, ..., 0],  // 225个元素，1=黑子，-1=白子，0=空
    "player": 1,                  // 当前玩家: 1=黑子, -1=白子
    "session_id": "game-123",     // 可选，会话ID（推荐使用）
    "reset_session": false,       // 可选，新游戏时设为 true
    "model_id": "strong",         // 可选，指定已加载的模型
    "profile": "strong",          // 可选，预设配置
    "sims": 600,                  // 可选，直接指定模拟次数 (最高优先级)
    "temp": 0.0                   // 可选，温度参数
}
```

**参数优先级**: `sims` > `profile` > 模型默认配置

**响应:**
```json
{
    "action": 112,              // 推荐落子位置 (0-224)
    "row": 7,                   // 行索引 (0-14)
    "col": 7,                   // 列索引 (0-14)
    "value": 0.15,              // 局势评估，黑子视角 (-1到1)
    "policy": [0.001, ...],     // 完整策略分布 (225个)
    "confidence": 0.35,         // AI 对推荐着法的置信度
    "top_moves": [              // Top 5 候选
        {"action": 112, "row": 7, "col": 7, "probability": 0.35},
        {"action": 97, "row": 6, "col": 7, "probability": 0.12},
        ...
    ],
    "sims": 600,
    "time_ms": 152.3
}
```

**棋盘坐标说明:**
- 一维索引: `action = row * 15 + col`
- 二维坐标: `row = action // 15, col = action % 15`

---

#### DELETE /api/session/{session_id} - 删除会话

释放会话资源。游戏结束时建议调用。

**响应:**
```json
{
    "status": "ok",
    "deleted": "game-123"
}
```

---

#### POST /api/analyze - 局面分析

分析当前局面，返回局势评估和候选着法。

**请求:**
```json
{
    "board": [0, 0, 0, ..., 0],
    "player": 1,
    "model_id": "strong",        // 可选
    "profile": "analysis",       // 可选
    "sims": 1600,                // 可选
    "top_n": 10                  // 返回候选着法数量
}
```

**响应:**
```json
{
    "value": 0.25,               // 局势评估，黑子视角
    "policy": [0.001, ...],
    "top_moves": [...],
    "game_over": false,
    "winner": null,
    "time_ms": 5.2
}
```

---

#### POST /api/evaluate - 快速评估

仅使用神经网络评估，不进行 MCTS 搜索。**速度最快**，适合批量评估。

**请求:**
```json
{
    "board": [0, 0, 0, ..., 0],
    "player": 1,
    "model_id": "strong"         // 可选
}
```

**响应:**
```json
{
    "value": 0.12,
    "policy": [0.002, ...],
    "legal_moves": 224,
    "time_ms": 2.1
}
```

---

#### POST /api/validate - 校验着法

校验某个位置是否可以落子。

**请求:**
```json
{
    "board": [0, 0, 0, ..., 0],
    "action": 112
}
```

**响应:**
```json
{
    "valid": true,
    "reason": null
}
```

---

#### POST /api/state - 游戏状态

查询游戏是否结束、胜者等信息。

**请求:**
```json
{
    "board": [1, 0, 0, ..., 0]
}
```

**响应:**
```json
{
    "game_over": false,
    "winner": null,
    "move_count": 1,
    "next_player": -1
}
```

---

### 模型管理接口

#### GET /models - 列出已加载模型

```json
[
    {"model_id": "default", "sims": 800, "c_puct": 5.0, "loaded_at": "2024-01-15T10:30:00"},
    {"model_id": "strong", "sims": 800, "c_puct": 5.0, "loaded_at": "2024-01-15T10:30:01"}
]
```

#### POST /models/load - 加载新模型

**请求:**
```json
{
    "model_path": "artifacts/az_model_v2.pt",
    "model_id": "strong",        // 可选，不指定则自动生成
    "profile": "strong",         // 可选，使用预设 sims
    "sims": 800,                 // 可选，自定义 sims
    "c_puct": 5.0,               // 可选
    "set_default": false         // 是否设为默认模型
}
```

**响应:**
```json
{
    "model_id": "strong",
    "model_path": "artifacts/az_model_v2.pt",
    "loaded_at": "2024-01-15T10:35:00",
    "sims": 800,
    "c_puct": 5.0
}
```

#### DELETE /models/{model_id} - 卸载模型

```json
{"status": "ok", "unloaded": "strong"}
```

---

### 系统接口

#### GET /health - 健康检查

```json
{
    "status": "ok",
    "models_loaded": [
        {"model_id": "default", "sims": 800, "c_puct": 5.0}
    ],
    "default_model_id": "default",
    "device": "cuda",
    "gpu_available": true,
    "gpu_name": "NVIDIA RTX 4090"
}
```

#### GET /config - 获取配置

#### POST /config - 更新配置

```json
{
    "model_id": "strong",        // 更新指定模型的配置
    "default_sims": 800,
    "c_puct": 5.0,
    "set_default": true          // 设为默认模型
}
```

---

## 命令行参数

```bash
uv run python src/api_service.py [options]

选项:
  --host HOST           绑定地址 (默认: 127.0.0.1)
  --port PORT           端口 (默认: 8001)
  --model PATH          默认模型路径 (默认: artifacts/az_model_best.pt)
  --device DEVICE       设备: cpu, cuda, cuda:0, auto (默认: auto)
  --sims N              默认模拟次数 (默认: 800)
  --default-profile     默认配置: fast/standard/strong/analysis (默认: standard)
  --models MODEL...     额外模型，格式: model_id=path[:profile]
  --reload              开发模式，自动重载
```

**多模型启动示例:**
```bash
uv run python src/api_service.py \
    --model artifacts/az_model_best.pt \
    --models "fast=artifacts/az_model_fast.pt:fast" "strong=artifacts/v2.pt:strong"
```

---

## Django 集成示例

### 1. 客户端封装

将 `django_client.py` 复制到 Django 项目中。

### 2. settings.py 配置

```python
# AlphaZero AI 服务配置
AZERO_API_URL = "http://localhost:8001"
AZERO_DEFAULT_SIMS = 800
AZERO_TIMEOUT = 30.0
```

### 3. 视图示例（使用会话模式）

```python
# views.py
import uuid
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
import json

from .utils.az_client import AlphaZeroClient


class GameView(View):
    def __init__(self):
        self.az_client = AlphaZeroClient(settings.AZERO_API_URL)

    def post(self, request):
        data = json.loads(request.body)
        board = data['board']        # 当前棋盘
        player_move = data.get('move')  # 玩家落子
        session_id = data.get('session_id') or str(uuid.uuid4())  # 获取或生成会话ID
        is_new_game = data.get('new_game', False)

        # 1. 应用玩家落子
        if player_move is not None:
            board[player_move] = 1  # 玩家执黑

        # 2. 检查游戏状态
        state = self.az_client.get_game_state(board)
        if state['game_over']:
            # 游戏结束，删除会话
            self.az_client.delete_session(session_id)
            return JsonResponse({
                'game_over': True,
                'winner': state['winner'],
                'session_id': session_id,
            })

        # 3. AI 落子（使用会话模式复用 MCTS 树）
        ai_result = self.az_client.get_move(
            board,
            player=-1,  # AI 执白
            session_id=session_id,
            reset_session=is_new_game,
            sims=settings.AZERO_DEFAULT_SIMS
        )

        # 4. 更新棋盘
        board[ai_result.action] = -1

        # 5. 检查 AI 是否获胜
        state = self.az_client.get_game_state(board)
        if state['game_over']:
            self.az_client.delete_session(session_id)

        return JsonResponse({
            'board': board,
            'ai_move': {
                'action': ai_result.action,
                'row': ai_result.row,
                'col': ai_result.col,
            },
            'value': ai_result.value,
            'game_over': state['game_over'],
            'winner': state.get('winner'),
            'session_id': session_id,  # 返回会话ID供客户端保存
        })
```

### 4. 客户端方法更新

`django_client.py` 需要添加会话支持：

```python
def get_move(self, board, player, session_id=None, reset_session=False,
             model_id=None, profile=None, sims=None, temp=0.0):
    """获取 AI 落子建议"""
    payload = {
        "board": board,
        "player": player,
        "temp": temp,
    }
    if session_id:
        payload["session_id"] = session_id
        payload["reset_session"] = reset_session
    if model_id:
        payload["model_id"] = model_id
    if profile:
        payload["profile"] = profile
    if sims:
        payload["sims"] = sims

    response = requests.post(f"{self.base_url}/api/move", json=payload, timeout=self.timeout)
    response.raise_for_status()
    return response.json()


def delete_session(self, session_id):
    """删除会话，释放资源"""
    response = requests.delete(f"{self.base_url}/api/session/{session_id}", timeout=self.timeout)
    response.raise_for_status()
    return response.json()
```

---

## 性能优化建议

### 1. 模拟次数选择

| 场景 | 推荐 sims | 说明 |
|------|----------|------|
| 在线对战 (快速) | 200-400 | 响应快，AI 较强 |
| 在线对战 (标准) | 400-800 | 平衡性能和强度 |
| 分析模式 | 800-2000 | 最强 AI |
| 批量评估 | 使用 /api/evaluate | 最快速度 |

### 2. GPU 加速

```bash
# 使用 GPU
uv run python src/api_service.py --device cuda --model artifacts/az_model_best.pt

# 指定 GPU
uv run python src/api_service.py --device cuda:3
```

### 3. 会话模式最佳实践

- **始终使用会话模式**：对于连续对局，会话模式能复用 MCTS 树，AI 更强
- **新游戏时设置 `reset_session: true`**：清空旧状态
- **游戏结束时删除会话**：调用 `DELETE /api/session/{session_id}` 释放内存
- **会话 ID 管理**：使用 UUID 或用户 ID + 游戏时间戳

### 4. 生产部署

推荐使用 gunicorn + uvicorn:

```bash
gunicorn src.api_service:app \
    --workers 1 \                    # 模型是全局状态，建议单 worker
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8001 \
    --timeout 120 \                  # MCTS 可能需要较长时间
    --keep-alive 5
```

### 5. Nginx 反向代理

```nginx
upstream azero_api {
    server 127.0.0.1:8001;
}

server {
    location /api/azero/ {
        proxy_pass http://azero_api/api/;
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
    }
}
```

---

## 错误处理

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 (无效棋盘、游戏已结束等) |
| 404 | 会话或模型不存在 |
| 503 | 模型未加载 |

示例错误响应:
```json
{
    "detail": "Game already finished"
}
```

---

## 文件说明

```
src/
├── api_service.py      # FastAPI 服务主文件
├── django_client.py    # Django 客户端封装
├── web_app.py          # 原有 Flask Web UI (保留)
├── az_mcts.py          # MCTS 实现
├── az_net.py           # 神经网络
└── env.py              # 游戏环境
```

---

## 更新日志

### v1.1.0
- **新增会话模式**: 支持 MCTS 树复用，AI 更强更稳定
- **新增 DELETE /api/session/{session_id} 接口**: 释放会话资源
- **默认 sims 改为 800**: 与 web_app.py 保持一致
- **优化内存管理**: 自动清理过期会话（最大 1000 个）
