// Canvas board + UI elements.
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("statusText");
const statusRow = document.querySelector(".row.status");
const evalText = document.getElementById("evalText");
const evalChart = document.getElementById("evalChart");
const simsInput = document.getElementById("sims");
const newGameBtn = document.getElementById("newGame");
const humanSide = document.getElementById("humanSide");
const analysisOnlyInput = document.getElementById("analysisOnly");
const deviceInput = document.getElementById("deviceInput");
const deviceInfo = document.getElementById("deviceInfo");
const azModelInput = document.getElementById("azModel");
const applyConfigBtn = document.getElementById("applyConfig");
const modelInfo = document.getElementById("modelInfo");
const modelTime = document.getElementById("modelTime");
const reloadModelBtn = document.getElementById("reloadModel");
const undoMoveBtn = document.getElementById("undoMove");
const policyListEl = document.getElementById("policyList");
const mctsListEl = document.getElementById("mctsList");
const analysisPanel = document.querySelector(".analysis-panel");

// Client-side state mirrored from the server.
let gameId = null;
let board = Array(225).fill(0);
let lastMove = null;
let human = "x";
let resultState = "idle";
let humanPlayer = 1;
let policy = null;
let policyMax = 0;
let lastBoard = board.slice();
let lastAiMove = null;
let lastMcts = null;
let showPolicyOverlay = false;
let evalHistory = [];
let analysisOnly = false;
let hoverMove = null;
let pendingMove = false;

const size = 15;
const pad = 34;
const labelOffset = 14;
let cell = 32;
const resultModal = document.getElementById("resultModal");
const resultTitle = document.getElementById("resultTitle");
const resultBody = document.getElementById("resultBody");
const closeModal = document.getElementById("closeModal");
const newFromModal = document.getElementById("newFromModal");
const evalCtx = evalChart ? evalChart.getContext("2d") : null;

function resizeBoard() {
  // Scale the canvas on small screens.
  const target = Math.min(window.innerWidth * 0.92, 560);
  canvas.width = target;
  canvas.height = target;
  cell = (target - pad * 2) / (size - 1);
  drawBoard();
}

function resizeEvalChart() {
  if (!evalChart || !evalCtx) return;
  const rect = evalChart.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  evalChart.width = Math.max(1, Math.floor(rect.width * dpr));
  evalChart.height = Math.max(1, Math.floor(rect.height * dpr));
  evalCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawEvalChart();
}

function setPolicy(probList) {
  if (Array.isArray(probList) && probList.length === board.length) {
    policy = probList;
    policyMax = probList.reduce((m, v) => (v > m ? v : m), 0);
  } else {
    policy = null;
    policyMax = 0;
  }
  renderPolicyList(probList, policyListEl, 6);
  drawBoard();
}

function drawEvalChart() {
  if (!evalChart || !evalCtx) return;
  const width = evalChart.getBoundingClientRect().width;
  const height = evalChart.getBoundingClientRect().height;
  evalCtx.clearRect(0, 0, width, height);
  evalCtx.fillStyle = "#f6ead8";
  evalCtx.fillRect(0, 0, width, height);

  const padding = 14;
  const chartW = width - padding * 2;
  const chartH = height - padding * 2;
  const originX = padding;
  const originY = padding;

  // Center line (0 advantage).
  evalCtx.strokeStyle = "rgba(120, 95, 70, 0.35)";
  evalCtx.lineWidth = 1;
  evalCtx.beginPath();
  evalCtx.moveTo(originX, originY + chartH / 2);
  evalCtx.lineTo(originX + chartW, originY + chartH / 2);
  evalCtx.stroke();

  const values = evalHistory.filter((v) => typeof v === "number");
  if (values.length === 0) return;

  const maxAbs = Math.max(1, ...values.map((v) => Math.abs(v)));
  const stepX = chartW / Math.max(1, values.length - 1);

  evalCtx.strokeStyle = "#1f6f63";
  evalCtx.lineWidth = 2;
  evalCtx.beginPath();
  values.forEach((v, i) => {
    const x = originX + i * stepX;
    const y = originY + chartH / 2 - (v / maxAbs) * (chartH / 2);
    if (i === 0) {
      evalCtx.moveTo(x, y);
    } else {
      evalCtx.lineTo(x, y);
    }
  });
  evalCtx.stroke();

  // Dots on each move.
  evalCtx.fillStyle = "#f08a24";
  values.forEach((v, i) => {
    const x = originX + i * stepX;
    const y = originY + chartH / 2 - (v / maxAbs) * (chartH / 2);
    evalCtx.beginPath();
    evalCtx.arc(x, y, 2.5, 0, Math.PI * 2);
    evalCtx.fill();
  });
}

function drawBoard() {
  // Draw grid and stones each update.
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#8b7b6b";
  ctx.lineWidth = 1;
  for (let i = 0; i < size; i++) {
    const x = pad + i * cell;
    const y = pad + i * cell;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(pad + cell * (size - 1), y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, pad);
    ctx.lineTo(x, pad + cell * (size - 1));
    ctx.stroke();
  }

  // Row/column labels.
  ctx.fillStyle = "#6a5c4d";
  ctx.font = "12px 'IBM Plex Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < size; i++) {
    const x = pad + i * cell;
    ctx.fillText(String(i + 1), x, pad - labelOffset);
  }
  ctx.textAlign = "right";
  for (let i = 0; i < size; i++) {
    const y = pad + i * cell;
    ctx.fillText(String(i + 1), pad - labelOffset, y);
  }

  const drawStone = (x, y, player, radius) => {
    const gradient = ctx.createRadialGradient(
      x - radius * 0.35,
      y - radius * 0.35,
      radius * 0.2,
      x,
      y,
      radius
    );
    if (player === 1) {
      gradient.addColorStop(0, "#4b4b4b");
      gradient.addColorStop(0.5, "#1e1e1e");
      gradient.addColorStop(1, "#050505");
      ctx.strokeStyle = "#0f0f0f";
    } else {
      gradient.addColorStop(0, "#ffffff");
      gradient.addColorStop(0.5, "#f1f1f1");
      gradient.addColorStop(1, "#cfcfcf");
      ctx.strokeStyle = "#a7a7a7";
    }
    ctx.save();
    ctx.shadowColor = "rgba(0, 0, 0, 0.25)";
    ctx.shadowBlur = 6;
    ctx.shadowOffsetY = 3;
    ctx.beginPath();
    ctx.fillStyle = gradient;
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(x, y, radius - 0.4, 0, Math.PI * 2);
    ctx.stroke();
  };

  board.forEach((v, idx) => {
    if (v === 0) return;
    const row = Math.floor(idx / size);
    const col = idx % size;
    const x = pad + col * cell;
    const y = pad + row * cell;
    drawStone(x, y, v, 12);
  });

  if (hoverMove !== null && board[hoverMove] === 0 && resultState === "ongoing") {
    const row = Math.floor(hoverMove / size);
    const col = hoverMove % size;
    const x = pad + col * cell;
    const y = pad + row * cell;
    const toPlay = board.filter((v) => v !== 0).length % 2 === 0 ? 1 : -1;
    const hoverPlayer = analysisOnly ? toPlay : humanPlayer;
    drawStone(x, y, hoverPlayer, 13);
  }

  if (lastMove !== null) {
    const row = Math.floor(lastMove / size);
    const col = lastMove % size;
    const x = pad + col * cell;
    const y = pad + row * cell;
    ctx.beginPath();
    ctx.strokeStyle = "#f08a24";
    ctx.lineWidth = 2;
    ctx.arc(x, y, 14, 0, Math.PI * 2);
    ctx.stroke();
  }

  if (lastAiMove !== null && lastAiMove !== lastMove) {
    const row = Math.floor(lastAiMove / size);
    const col = lastAiMove % size;
    const x = pad + col * cell;
    const y = pad + row * cell;
    ctx.beginPath();
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 2;
    ctx.arc(x, y, 13, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Policy heatmap and labels on top to avoid being covered by stones.
  if (showPolicyOverlay && policy && policyMax > 0) {
    policy.forEach((p, idx) => {
      if (!p) return;
      const row = Math.floor(idx / size);
      const col = idx % size;
      const x = pad + col * cell;
      const y = pad + row * cell;
      const strength = Math.sqrt(p / policyMax);
      const alpha = Math.min(0.75, 0.18 + strength * 0.65);
      const radius = 6 + strength * 12;
      ctx.beginPath();
      ctx.fillStyle = `rgba(239, 111, 0, ${alpha.toFixed(3)})`;
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      // Probability label (percentage).
      const percent = Math.round(p * 1000) / 10; // 0.1% resolution
      if (percent > 0) {
        const text = `${percent}%`;
        ctx.font = "11px 'IBM Plex Mono', monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const labelY = board[idx] === 0 ? y : y - 16;
        const metrics = ctx.measureText(text);
        const padX = 4;
        const padY = 2;
        const boxW = metrics.width + padX * 2;
        const boxH = 14 + padY * 2;
        ctx.fillStyle = "rgba(255, 250, 240, 0.92)";
        ctx.strokeStyle = "rgba(40, 30, 20, 0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(x - boxW / 2, labelY - boxH / 2, boxW, boxH);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#1d130b";
        ctx.fillText(text, x, labelY);
      }
    });
  }
}

function renderPolicyList(probList, targetEl, limit) {
  if (!targetEl) return;
  if (!Array.isArray(probList) || probList.length !== board.length) {
    targetEl.textContent = "暂无";
    return;
  }
  const entries = probList
    .map((p, idx) => ({ p, idx }))
    .filter((e) => e.p > 0)
    .sort((a, b) => b.p - a.p);
  const slice = typeof limit === "number" ? entries.slice(0, limit) : entries;
  if (entries.length === 0) {
    targetEl.textContent = "暂无";
    return;
  }
  targetEl.innerHTML = "";
  slice.forEach(({ p, idx }) => {
    const row = Math.floor(idx / size) + 1;
    const col = (idx % size) + 1;
    const item = document.createElement("div");
    item.className = "policy-item";
    const coord = document.createElement("span");
    coord.className = "coord";
    coord.textContent = `(${row}, ${col})`;
    const prob = document.createElement("span");
    prob.className = "prob";
    prob.textContent = `${(p * 100).toFixed(2)}%`;
    item.appendChild(coord);
    item.appendChild(prob);
    targetEl.appendChild(item);
  });
}

function renderMctsList(mcts, isEmptyBoard = false) {
  if (!mctsListEl) return;
  mctsListEl.classList.toggle("mcts-placeholder", isEmptyBoard);
  if (!mcts || !Array.isArray(mcts.children) || mcts.children.length === 0) {
    mctsListEl.textContent = isEmptyBoard ? "" : "暂无";
    return;
  }
  const top = mcts.children.slice(0, 8);
  mctsListEl.innerHTML = "";
  top.forEach((c) => {
    const row = Math.floor(c.action / size) + 1;
    const col = (c.action % size) + 1;
    const item = document.createElement("div");
    item.className = "policy-item";
    const coord = document.createElement("span");
    coord.className = "coord";
    coord.textContent = `(${row}, ${col})`;
    const detail = document.createElement("span");
    detail.className = "prob";
    detail.textContent = `v:${c.visits} p:${c.prior.toFixed(2)} q:${c.q.toFixed(
      2
    )} u:${c.u.toFixed(2)} s:${c.score.toFixed(2)} pi:${(
      c.pi * 100
    ).toFixed(1)}%`;
    item.appendChild(coord);
    item.appendChild(detail);
    mctsListEl.appendChild(item);
  });
}

function setEval(value) {
  if (value === null || value === undefined) {
    evalText.textContent = "AZ 评估：暂无";
    drawEvalChart();
    return;
  }
  evalText.textContent = `AZ 评估（黑方视角）：${value.toFixed(2)}`;
  drawEvalChart();
}

function setStatus(text) {
  statusText.textContent = text;
}

function setPending(isPending) {
  if (!statusRow) return;
  statusRow.classList.toggle("pending", isPending);
}

function detectAiMove(prevBoard, nextBoard, aiPlayer) {
  let found = null;
  for (let i = 0; i < nextBoard.length; i++) {
    if (prevBoard[i] === 0 && nextBoard[i] === aiPlayer) {
      if (found !== null) return null;
      found = i;
    }
  }
  return found;
}

function getHumanSide() {
  const active = humanSide.querySelector(".pill.active");
  return active?.dataset.side || "x";
}

function updateHumanSideButtons() {
  humanSide.querySelectorAll(".pill").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.side === human);
  });
}

humanSide.addEventListener("click", (e) => {
  const btn = e.target.closest(".pill");
  if (!btn) return;
  human = btn.dataset.side;
  updateHumanSideButtons();
});

function updateHumanPlayer(side) {
  humanPlayer = side === "x" ? 1 : -1;
}

function syncEvalHistory(history, currentEval) {
  if (Array.isArray(history)) {
    evalHistory = history.slice();
  } else if (currentEval !== null && currentEval !== undefined) {
    evalHistory = evalHistory.concat([currentEval]);
  }
  drawEvalChart();
}

function getBestPolicyMove(probList) {
  if (!Array.isArray(probList) || probList.length === 0) return null;
  let bestIdx = null;
  let best = -1;
  probList.forEach((p, idx) => {
    if (p > best) {
      best = p;
      bestIdx = idx;
    }
  });
  return bestIdx;
}

function updateStatusForMode(result, nextPlayer) {
  if (!result || result.state === "ongoing") {
    if (analysisOnly) {
      const side = nextPlayer === 1 ? "黑方（X）" : "白方（O）";
      setStatus(`${side}落子。`);
    } else {
      setStatus("轮到你了。");
    }
    return;
  }
  if (result.state === "draw") {
    setStatus("平局。");
    return;
  }
  if (analysisOnly) {
    const winner = result.winner === 1 ? "黑方（X）" : "白方（O）";
    setStatus(`${winner}胜。`);
  } else if (result.winner === humanPlayer) {
    setStatus("你赢了！");
  } else {
    setStatus("AI 获胜。");
  }
}

function showResultModal(result) {
  if (!result || result.state === "ongoing") return;
  resultTitle.textContent = "对局结束";
  if (result.state === "draw") {
    resultBody.textContent = "平局。";
  } else if (analysisOnly) {
    resultBody.textContent = result.winner === 1 ? "黑方胜。" : "白方胜。";
  } else if (result.winner === humanPlayer) {
    resultBody.textContent = "你赢了！";
  } else {
    resultBody.textContent = "AI 获胜。";
  }
  resultModal.classList.remove("hidden");
}

function hideResultModal() {
  resultModal.classList.add("hidden");
}

async function newGame() {
  // Create a new server-side game.
  setStatus("正在创建对局...");
  const sims = parseInt(simsInput.value, 10) || 800;
  lastBoard = board.slice();
  const res = await fetch("/api/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sims,
      device: deviceInput.value.trim() || "cpu",
      az_model: azModelInput.value.trim(),
      human: getHumanSide(),
      analysis_only: analysisOnlyInput?.value === "1",
    }),
  });
  const data = await res.json();
  gameId = data.game_id;
  board = data.board;
  resultState = data.result.state;
  updateHumanPlayer(getHumanSide());
  analysisOnly = !!data.analysis_only;
  if (analysisOnlyInput) analysisOnlyInput.value = analysisOnly ? "1" : "0";
  lastAiMove = analysisOnly
    ? getBestPolicyMove(data.policy)
    : detectAiMove(lastBoard, board, -humanPlayer);
  lastMove = null;
  setPolicy(data.policy);
  syncEvalHistory(data.eval_history, data.eval);
  setEval(data.eval);
  lastMcts = data.mcts || null;
  renderMctsList(lastMcts, board.every((v) => v === 0));
  updateStatusForMode(data.result, data.next_player);
  hideResultModal();
  showResultModal(data.result);
  if (data.config) {
    syncConfigUi(data.config);
  }
  drawBoard();
  lastBoard = board.slice();
}

async function playMove(idx) {
  // Send human move to the backend and render response.
  if (!gameId || resultState !== "ongoing") return;
  if (board[idx] !== 0) return;
  if (pendingMove) return;
  pendingMove = true;
  setPending(true);
  lastBoard = board.slice();
  board[idx] = analysisOnly
    ? (board.filter((v) => v !== 0).length % 2 === 0 ? 1 : -1)
    : humanPlayer;
  lastMove = idx;
  hoverMove = null;
  drawBoard();
  setStatus("思考中...");
  const res = await fetch("/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId, action: idx }),
  });
  if (!res.ok) {
    board = lastBoard.slice();
    pendingMove = false;
    setPending(false);
    drawBoard();
    return;
  }
  const data = await res.json();
  board = data.board;
  resultState = data.result.state;
  lastMove = idx;
  analysisOnly = !!data.analysis_only;
  if (analysisOnlyInput) analysisOnlyInput.value = analysisOnly ? "1" : "0";
  lastAiMove = analysisOnly
    ? getBestPolicyMove(data.policy)
    : detectAiMove(lastBoard, board, -humanPlayer);
  setPolicy(data.policy);
  syncEvalHistory(data.eval_history, data.eval);
  setEval(data.eval);
  lastMcts = data.mcts || null;
  renderMctsList(lastMcts, board.every((v) => v === 0));
  updateStatusForMode(data.result, data.next_player);
  showResultModal(data.result);
  drawBoard();
  lastBoard = board.slice();
  pendingMove = false;
  setPending(false);
}

async function undoMove() {
  if (!gameId) return;
  const steps = analysisOnly ? 1 : 2;
  const res = await fetch("/api/undo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId, steps }),
  });
  if (!res.ok) return;
  const data = await res.json();
  board = data.board;
  resultState = data.result.state;
  lastMove = null;
  hoverMove = null;
  analysisOnly = !!data.analysis_only;
  if (analysisOnlyInput) analysisOnlyInput.checked = analysisOnly;
  lastAiMove = analysisOnly ? getBestPolicyMove(data.policy) : null;
  setPolicy(data.policy);
  setAiMoveProb(data.policy, lastAiMove);
  syncEvalHistory(data.eval_history, data.eval);
  setEval(data.eval);
  lastMcts = data.mcts || null;
  renderMctsList(lastMcts, board.every((v) => v === 0));
  updateStatusForMode(data.result, data.next_player);
  if (resultState === "ongoing") {
    hideResultModal();
  }
  showResultModal(data.result);
  drawBoard();
  lastBoard = board.slice();
}

if (analysisPanel) {
  analysisPanel.addEventListener("mouseenter", () => {
    showPolicyOverlay = true;
    drawBoard();
  });
  analysisPanel.addEventListener("mouseleave", () => {
    showPolicyOverlay = false;
    drawBoard();
  });
}

canvas.addEventListener("click", (event) => {
  // Translate click position to nearest intersection.
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  const col = Math.round((x - pad) / cell);
  const row = Math.round((y - pad) / cell);
  if (row < 0 || row >= size || col < 0 || col >= size) return;
  const idx = row * size + col;
  playMove(idx);
});

canvas.addEventListener("mousemove", (event) => {
  if (resultState !== "ongoing") {
    if (hoverMove !== null) {
      hoverMove = null;
      drawBoard();
    }
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  const col = Math.round((x - pad) / cell);
  const row = Math.round((y - pad) / cell);
  if (row < 0 || row >= size || col < 0 || col >= size) {
    if (hoverMove !== null) {
      hoverMove = null;
      drawBoard();
    }
    return;
  }
  const idx = row * size + col;
  if (board[idx] !== 0) {
    if (hoverMove !== null) {
      hoverMove = null;
      drawBoard();
    }
    return;
  }
  if (hoverMove !== idx) {
    hoverMove = idx;
    drawBoard();
  }
});

canvas.addEventListener("mouseleave", () => {
  if (hoverMove !== null) {
    hoverMove = null;
    drawBoard();
  }
});

async function loadConfig() {
  const res = await fetch("/api/config");
  if (!res.ok) return;
  const data = await res.json();
  syncConfigUi(data);
}

async function applyConfig() {
  const sims = parseInt(simsInput.value, 10) || 800;
  const payload = {
    sims,
    device: deviceInput.value.trim() || "cpu",
    az_model: azModelInput.value.trim(),
  };
  const res = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) return;
  const data = await res.json();
  syncConfigUi(data);
}

async function reloadModel() {
  await applyConfig();
}

function syncConfigUi(data) {
  if (data.sims) simsInput.value = data.sims;
  if (data.device) {
    deviceInput.value = data.device;
    deviceInfo.textContent = `服务端：${data.device}`;
  }
  if (data.az_model) azModelInput.value = data.az_model;
  if (typeof data.model_loaded === "boolean") {
    modelInfo.textContent = data.model_loaded ? "模型：已加载" : "模型：未加载";
  }
  if (data.model_file && data.model_file.created) {
    const created = data.model_file.created;
    const modifiedText = data.model_file.modified ? `，修改于 ${data.model_file.modified}` : "";
    modelTime.textContent = `模型创建于 ${created}${modifiedText}`;
  } else {
    modelTime.textContent = "模型文件：暂无";
  }
}

newGameBtn.addEventListener("click", newGame);
applyConfigBtn.addEventListener("click", applyConfig);
reloadModelBtn.addEventListener("click", reloadModel);
closeModal.addEventListener("click", hideResultModal);
newFromModal.addEventListener("click", () => {
  hideResultModal();
  newGame();
});
undoMoveBtn?.addEventListener("click", undoMove);
window.addEventListener("resize", resizeBoard);
window.addEventListener("resize", resizeEvalChart);

resizeBoard();
resizeEvalChart();
loadConfig();
