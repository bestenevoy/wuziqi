# AlphaZero 五子棋 API 调用说明文档

## 目录

1. [快速开始](#快速开始)
2. [会话模式（重要）](#会话模式重要)
3. [接口详解](#接口详解)
4. [棋盘数据格式](#棋盘数据格式)
5. [完整调用示例](#完整调用示例)
6. [错误处理](#错误处理)
7. [性能调优](#性能调优)
8. [常见问题](#常见问题)

---

## 快速开始

### 启动服务

```bash
cd /home/wrz/code/wuziqi

# 开发模式
uv run python src/api_service.py --host 127.0.0.1 --port 8001

# 生产模式 (GPU 加速)
uv run python src/api_service.py --host 0.0.0.0 --port 8001 --device cuda:3 --model artifacts/az_model_best.pt
```

### 验证服务

```bash
# 健康检查
curl http://localhost:8001/health

# 查看文档
open http://localhost:8001/docs
```

---

## 会话模式（重要）

### 为什么需要会话模式？

MCTS 搜索树可以复用！当对手落子后，之前搜索过的子树仍然有效。使用会话模式：

- **AI 更强**: 累积搜索结果，决策质量更高
- **更稳定**: 复用树避免了从零开始的不确定性
- **推荐使用**: 对于连续对局，**强烈建议**使用会话模式

### 无状态模式 vs 会话模式

| 特性 | 无状态模式 | 会话模式 |
|------|-----------|---------|
| MCTS 树 | 每次重建 | 复用累积 |
| AI 强度 | 标准 | **更强** |
| 内存占用 | 低 | 每会话约 10-50MB |
| 适用场景 | 单次请求 | **连续对局** |

### 会话模式使用示例

```bash
# 新游戏开始 - 设置 reset_session=true
curl -X POST http://localhost:8001/api/move \
  -H "Content-Type: application/json" \
  -d '{
    "board": [0,0,0,...],
    "player": 1,
    "session_id": "game-123",
    "reset_session": true
  }'

# 后续请求 - 使用相同的 session_id
curl -X POST http://localhost:8001/api/move \
  -H "Content-Type: application/json" \
  -d '{
    "board": [1,0,0,...],  # 包含对手的新落子
    "player": -1,
    "session_id": "game-123"
  }'

# 游戏结束 - 删除会话释放资源
curl -X DELETE http://localhost:8001/api/session/game-123
```

---

## 接口详解

### 1. 获取 AI 落子 (核心接口)

**端点:** `POST /api/move`

**用途:** 获取 AI 推荐的下一步落子位置，这是最主要的接口。

**请求参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| board | array[225] | 是 | 棋盘一维数组 |
| player | int | 是 | 当前玩家: 1=黑子, -1=白子 |
| session_id | string | 否 | 会话ID，用于复用 MCTS 树（推荐） |
| reset_session | bool | 否 | 新游戏时设为 true |
| sims | int | 否 | MCTS 模拟次数，默认 800 |
| temp | float | 否 | 温度参数，默认 0 (贪婪选择) |
| profile | string | 否 | 预设配置: fast/standard/strong/analysis |
| model_id | string | 否 | 指定已加载的模型 ID |

**请求示例:**

```bash
curl -X POST http://localhost:8001/api/move \
  -H "Content-Type: application/json" \
  -d '{
    "board": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
              0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "player": 1,
    "session_id": "game-123",
    "reset_session": true,
    "sims": 800,
    "temp": 0.0
  }'
```

**响应示例:**

```json
{
  "action": 112,
  "row": 7,
  "col": 7,
  "value": 0.05,
  "policy": [0.0001, 0.0002, ...],
  "confidence": 0.45,
  "top_moves": [
    {"action": 112, "row": 7, "col": 7, "probability": 0.45},
    {"action": 98, "row": 6, "col": 8, "probability": 0.12},
    {"action": 126, "row": 8, "col": 6, "probability": 0.08},
    {"action": 97, "row": 6, "col": 7, "probability": 0.06},
    {"action": 127, "row": 8, "col": 7, "probability": 0.05}
  ],
  "sims": 800,
  "time_ms": 156.32
}
```

**响应字段说明:**

| 字段 | 类型 | 说明 |
|------|------|------|
| action | int | 推荐落子位置 (0-224) |
| row | int | 行坐标 (0-14) |
| col | int | 列坐标 (0-14) |
| value | float | 局势评估，黑子视角 (-1 到 1) |
| policy | array | 完整策略分布 (225 个概率值) |
| confidence | float | AI 对推荐着法的置信度 |
| top_moves | array | Top 5 候选着法列表 |
| sims | int | 实际模拟次数 |
| time_ms | float | 推理耗时 (毫秒) |

---

### 2. 删除会话

**端点:** `DELETE /api/session/{session_id}`

**用途:** 游戏结束时删除会话，释放内存资源。

**请求示例:**

```bash
curl -X DELETE http://localhost:8001/api/session/game-123
```

**响应:**

```json
{
  "status": "ok",
  "deleted": "game-123"
}
```

---

### 3. 局面分析

**端点:** `POST /api/analyze`

**用途:** 分析当前局面，返回局势评估和候选着法，不给出具体推荐。适合分析模式或教学场景。

**请求参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| board | array[225] | 是 | 棋盘数组 |
| player | int | 是 | 当前玩家 |
| top_n | int | 否 | 返回候选数，默认 10 |
| sims | int | 否 | MCTS 模拟次数 |
| profile | string | 否 | 预设配置 |

**请求示例:**

```bash
curl -X POST http://localhost:8001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "board": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, ...],
    "player": 1,
    "top_n": 5,
    "profile": "analysis"
  }'
```

**响应示例:**

```json
{
  "value": -0.25,
  "policy": [0.001, 0.002, ...],
  "top_moves": [
    {"action": 98, "row": 6, "col": 8, "probability": 0.35},
    ...
  ],
  "game_over": false,
  "winner": null,
  "time_ms": 5.21
}
```

---

### 4. 快速评估

**端点:** `POST /api/evaluate`

**用途:** 仅使用神经网络评估，不进行 MCTS 搜索。速度最快，适合批量评估场景。

**请求参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| board | array[225] | 是 | 棋盘数组 |
| player | int | 是 | 当前玩家 |
| model_id | string | 否 | 指定模型 ID |

**响应示例:**

```json
{
  "value": 0.12,
  "policy": [0.002, 0.001, ...],
  "legal_moves": 223,
  "time_ms": 1.85
}
```

**性能对比:**

| 接口 | 耗时 (GPU) | 耗时 (CPU) | 适用场景 |
|------|-----------|-----------|----------|
| /api/evaluate | ~2ms | ~5ms | 快速评估、批量处理 |
| /api/move (sims=200) | ~80ms | ~300ms | 快速对战 |
| /api/move (sims=800) | ~300ms | ~1.2s | 标准对战（默认） |
| /api/move (sims=1600) | ~600ms | ~2.4s | 分析模式 |

---

### 5. 校验着法

**端点:** `POST /api/validate`

**用途:** 校验某个位置是否可以落子。

**请求示例:**

```bash
curl -X POST http://localhost:8001/api/validate \
  -H "Content-Type: application/json" \
  -d '{
    "board": [0,0,0,...],
    "action": 112
  }'
```

**响应:**

```json
{"valid": true, "reason": null}
// 或
{"valid": false, "reason": "Position already occupied"}
```

---

### 6. 查询游戏状态

**端点:** `POST /api/state`

**用途:** 查询游戏是否结束、胜者是谁。

**请求示例:**

```bash
curl -X POST http://localhost:8001/api/state \
  -H "Content-Type: application/json" \
  -d '{"board": [1,1,1,1,1,0,0,...]}'
```

**响应:**

```json
{
  "game_over": true,
  "winner": 1,
  "move_count": 5,
  "next_player": -1
}
```

**winner 字段说明:**
- `1`: 黑子获胜
- `-1`: 白子获胜
- `0`: 平局
- `null`: 游戏未结束

---

### 7. 健康检查

**端点:** `GET /health`

**响应:**

```json
{
  "status": "ok",
  "models_loaded": [
    {"model_id": "default", "sims": 800, "c_puct": 5.0}
  ],
  "default_model_id": "default",
  "device": "cuda",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090"
}
```

---

### 8. 配置管理

**获取配置:** `GET /config`

**更新配置:** `POST /config`

```bash
curl -X POST http://localhost:8001/config \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "artifacts/az_model_v1.pt",
    "default_sims": 800,
    "device": "cuda:0"
  }'
```

---

## 棋盘数据格式

### 一维数组表示

棋盘使用长度为 225 的一维数组表示 (15×15 = 225):

```
索引 = row * 15 + col

位置映射:
  0   1   2   3   4  ...  14   (col 0-14, row 0)
 15  16  17  18  19  ...  28   (row 1)
 30  31  32  33  34  ...  44   (row 2)
 ...
210 211 212 213 214  ... 224   (row 14)
```

### 数值含义

| 值 | 含义 |
|----|------|
| 0 | 空位 |
| 1 | 黑子 (X) |
| -1 | 白子 (O) |

### 示例

空棋盘:
```python
board = [0] * 225
```

黑子下在中心 (7, 7):
```python
board[7 * 15 + 7] = 1  # board[112] = 1
```

坐标转换:
```python
# 一维索引 -> 二维坐标
action = 112
row, col = divmod(action, 15)  # row=7, col=7

# 二维坐标 -> 一维索引
row, col = 7, 7
action = row * 15 + col  # action=112
```

---

## 完整调用示例

### Python (requests) - 会话模式（推荐）

```python
import requests
import json
import uuid

API_URL = "http://localhost:8001"

def get_ai_move(board: list, player: int = 1, session_id: str = None,
                reset_session: bool = False, sims: int = 800) -> dict:
    """获取 AI 落子（支持会话模式）"""
    payload = {
        "board": board,
        "player": player,
        "sims": sims,
    }
    if session_id:
        payload["session_id"] = session_id
        payload["reset_session"] = reset_session

    response = requests.post(f"{API_URL}/api/move", json=payload)
    response.raise_for_status()
    return response.json()

def delete_session(session_id: str):
    """删除会话，释放资源"""
    response = requests.delete(f"{API_URL}/api/session/{session_id}")
    response.raise_for_status()
    return response.json()

def play_game_with_session():
    """完整对局示例（使用会话模式）"""
    board = [0] * 225
    current_player = 1  # 黑子先手
    session_id = str(uuid.uuid4())  # 生成会话 ID

    print(f"会话 ID: {session_id}")
    move_count = 0

    try:
        while True:
            # 新游戏时设置 reset_session=True
            is_new_game = (move_count == 0)

            # 获取 AI 着法（会话模式，复用 MCTS 树）
            result = get_ai_move(
                board, current_player,
                session_id=session_id,
                reset_session=is_new_game
            )

            # 应用着法
            board[result["action"]] = current_player
            print(f"玩家 {current_player} 落子: ({result['row']}, {result['col']})")
            print(f"局势评估: {result['value']:.3f}, 耗时: {result['time_ms']:.1f}ms")

            # 检查游戏状态
            state = requests.post(
                f"{API_URL}/api/state",
                json={"board": board}
            ).json()

            if state["game_over"]:
                if state["winner"] == 0:
                    print("平局!")
                else:
                    print(f"{'黑子' if state['winner'] == 1 else '白子'} 获胜!")
                break

            current_player = -current_player
            move_count += 1

    finally:
        # 游戏结束，删除会话
        delete_session(session_id)
        print("会话已释放")

if __name__ == "__main__":
    play_game_with_session()
```

### Python (requests) - 无状态模式

```python
import requests

API_URL = "http://localhost:8001"

def get_ai_move(board: list, player: int = 1) -> dict:
    """获取 AI 落子（无状态模式）"""
    response = requests.post(
        f"{API_URL}/api/move",
        json={"board": board, "player": player, "sims": 800}
    )
    response.raise_for_status()
    return response.json()

def play_game():
    """完整对局示例（无状态模式）"""
    board = [0] * 225
    current_player = 1  # 黑子先手

    while True:
        # 获取 AI 着法
        result = get_ai_move(board, current_player)

        # 应用着法
        board[result["action"]] = current_player
        print(f"玩家 {current_player} 落子: ({result['row']}, {result['col']})")
        print(f"局势评估: {result['value']:.3f}")

        # 检查游戏状态
        state = requests.post(
            f"{API_URL}/api/state",
            json={"board": board}
        ).json()

        if state["game_over"]:
            if state["winner"] == 0:
                print("平局!")
            else:
                print(f"{'黑子' if state['winner'] == 1 else '白子'} 获胜!")
            break

        current_player = -current_player

if __name__ == "__main__":
    play_game()
```

### Python (httpx 异步)

```python
import httpx
import asyncio
import uuid

API_URL = "http://localhost:8001"

async def get_ai_move_async(board: list, player: int = 1,
                            session_id: str = None,
                            reset_session: bool = False) -> dict:
    """获取 AI 落子（异步，支持会话模式）"""
    payload = {"board": board, "player": player}
    if session_id:
        payload["session_id"] = session_id
        payload["reset_session"] = reset_session

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/api/move", json=payload)
        response.raise_for_status()
        return response.json()

async def main():
    board = [0] * 225
    session_id = str(uuid.uuid4())

    # 新游戏
    result = await get_ai_move_async(
        board, player=1,
        session_id=session_id,
        reset_session=True
    )
    print(f"AI 推荐: ({result['row']}, {result['col']})")

    # 清理会话
    async with httpx.AsyncClient() as client:
        await client.delete(f"{API_URL}/api/session/{session_id}")

asyncio.run(main())
```

### JavaScript (fetch)

```javascript
const API_URL = 'http://localhost:8001';

async function getAIMove(board, player = 1, sessionId = null, resetSession = false) {
    const payload = { board, player };
    if (sessionId) {
        payload.session_id = sessionId;
        payload.reset_session = resetSession;
    }

    const response = await fetch(`${API_URL}/api/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    return response.json();
}

async function deleteSession(sessionId) {
    const response = await fetch(`${API_URL}/api/session/${sessionId}`, {
        method: 'DELETE'
    });
    return response.json();
}

// 使用示例（会话模式）
async function playGame() {
    const sessionId = crypto.randomUUID();
    const board = new Array(225).fill(0);

    try {
        // 新游戏
        let result = await getAIMove(board, 1, sessionId, true);
        console.log(`AI 推荐: (${result.row}, ${result.col})`);

        // 后续请求...
        // result = await getAIMove(newBoard, -1, sessionId, false);
    } finally {
        // 清理会话
        await deleteSession(sessionId);
    }
}

// 无状态模式
async function quickMove(board, player = 1) {
    const result = await getAIMove(board, player);
    console.log(`AI 推荐: (${result.row}, ${result.col})`);
}
```

### cURL

```bash
# 获取 AI 落子
curl -X POST http://localhost:8001/api/move \
  -H "Content-Type: application/json" \
  -d '{"board": [0,0,0,...], "player": 1}'

# 健康检查
curl http://localhost:8001/health
```

---

## 错误处理

### HTTP 状态码

| 状态码 | 说明 | 处理建议 |
|--------|------|----------|
| 200 | 成功 | 正常处理响应 |
| 400 | 请求参数错误 | 检查请求格式、棋盘有效性 |
| 503 | 模型未加载 | 等待模型加载或检查模型路径 |

### 错误响应格式

```json
{
  "detail": "Game already finished"
}
```

### 常见错误

**1. 棋盘长度错误**
```json
{"detail": "Board must have 225 elements, got 200"}
```
解决: 确保传入 225 个元素。

**2. 无效玩家**
```json
{"detail": "Invalid player: expected -1, got 1"}
```
解决: `player` 必须与棋盘状态匹配，轮到谁下谁就是 `player`。

**3. 游戏已结束**
```json
{"detail": "Game already finished"}
```
解决: 检查游戏状态后再请求 AI 着法。

**4. 位置已被占用**
```json
{"valid": false, "reason": "Position already occupied"}
```
解决: 先调用 `/api/validate` 校验。

---

## 性能调优

### 模拟次数选择

```python
# 快速对战 (适合在线实时对战)
sims = 200  # ~80ms GPU / ~300ms CPU

# 标准对战 (平衡性能和强度)
sims = 400  # ~150ms GPU / ~600ms CPU

# 分析模式 (最强 AI)
sims = 800  # ~300ms GPU / ~1.2s CPU
```

### GPU 加速

```bash
# 启动时指定 GPU
uv run python src/api_service.py --device cuda

# 指定 GPU 编号
uv run python src/api_service.py --device cuda:0
```

### 生产部署

```bash
# 使用 gunicorn + uvicorn
gunicorn src.api_service:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8001 \
    --timeout 120
```

---

## 常见问题

### Q: 如何判断当前轮到谁下棋?

A: 黑子先手，统计棋盘上非零元素个数:
```python
move_count = sum(1 for x in board if x != 0)
current_player = 1 if move_count % 2 == 0 else -1
```

### Q: value 字段如何解读?

A: `value` 是从黑子视角的局势评估:
- `value > 0`: 黑子优势
- `value < 0`: 白子优势
- `value ≈ 0`: 局势均衡
- 范围: -1 到 1

### Q: 会话模式和無状态模式有什么区别？

A:
| 特性 | 无状态模式 | 会话模式 |
|------|-----------|---------|
| MCTS 树 | 每次重建 | **复用累积** |
| AI 强度 | 标准 | **更强** |
| 内存占用 | 低 | 每会话 10-50MB |
| 使用场景 | 单次请求 | **连续对局（推荐）** |

会话模式复用 MCTS 搜索树，AI 会越来越强。**对于连续对局，强烈推荐使用会话模式。**

### Q: 如何实现悔棋?

A: 使用会话模式时，悔棋需要重置会话：

```python
# 方式1: 重新设置 reset_session=True
result = get_ai_move(board_after_undo, player, session_id, reset_session=True)

# 方式2: 删除旧会话，创建新会话
delete_session(old_session_id)
new_session_id = str(uuid.uuid4())
result = get_ai_move(board_after_undo, player, new_session_id, reset_session=True)
```

无状态模式下，直接发送悔棋后的棋盘即可：
```python
# 悔棋
board[last_action] = 0

# 重新请求 AI
result = get_ai_move(board, current_player)
```

### Q: 多个对局同时进行怎么办?

A: 使用不同的 `session_id` 区分对局：

```python
# 对局 1
result1 = get_ai_move(board1, player1, session_id="game-1")

# 对局 2
result2 = get_ai_move(board2, player2, session_id="game-2")

# 游戏结束时清理
delete_session("game-1")
delete_session("game-2")
```

### Q: 如何提升 AI 强度?

A:
1. **使用会话模式**: 复用 MCTS 树，AI 更强更稳定
2. 增加模拟次数 (`sims` 或 `profile=strong/analysis`)
3. 使用训练更久的模型
4. 启用 GPU 加速

### Q: 默认的模拟次数是多少？

A: 默认 `sims=800`，相当于 `strong` 级别。可以通过以下方式调整：

```python
# 方式1: 直接指定 sims
result = get_ai_move(board, player, sims=1600)

# 方式2: 使用 profile
result = get_ai_move(board, player, profile="analysis")
```

| Profile | sims | 说明 |
|---------|------|------|
| fast | 100 | 快速响应 |
| standard | 400 | 标准强度 |
| strong | 800 | 默认，强力模式 |
| analysis | 1600 | 分析模式，最强 |

---

## 联系与支持

- API 文档: http://localhost:8001/docs
- 源码: /home/wrz/code/wuziqi/src/api_service.py
- 客户端: /home/wrz/code/wuziqi/src/django_client.py
