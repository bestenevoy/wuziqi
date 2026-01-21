// Canvas board + UI elements.
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("statusText");
const evalText = document.getElementById("evalText");
const evalFill = document.getElementById("evalFill");
const simsInput = document.getElementById("sims");
const newGameBtn = document.getElementById("newGame");
const humanSide = document.getElementById("humanSide");
const deviceInput = document.getElementById("deviceInput");
const deviceInfo = document.getElementById("deviceInfo");
const azModelInput = document.getElementById("azModel");
const applyConfigBtn = document.getElementById("applyConfig");
const modelInfo = document.getElementById("modelInfo");
const modelTime = document.getElementById("modelTime");
const reloadModelBtn = document.getElementById("reloadModel");
const policyListEl = document.getElementById("policyList");
const aiProbEl = document.getElementById("aiProb");
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

const size = 15;
const pad = 34;
const labelOffset = 14;
let cell = 32;
const resultModal = document.getElementById("resultModal");
const resultTitle = document.getElementById("resultTitle");
const resultBody = document.getElementById("resultBody");
const closeModal = document.getElementById("closeModal");
const newFromModal = document.getElementById("newFromModal");

function resizeBoard() {
  // Scale the canvas on small screens.
  const target = Math.min(window.innerWidth * 0.92, 560);
  canvas.width = target;
  canvas.height = target;
  cell = (target - pad * 2) / (size - 1);
  drawBoard();
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

  board.forEach((v, idx) => {
    if (v === 0) return;
    const r = 12;
    const row = Math.floor(idx / size);
    const col = idx % size;
    const x = pad + col * cell;
    const y = pad + row * cell;
    ctx.beginPath();
    ctx.fillStyle = v === 1 ? "#1f1f1f" : "#f5f5f5";
    ctx.strokeStyle = v === 1 ? "#111" : "#888";
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });

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
    targetEl.textContent = "N/A";
    return;
  }
  const entries = probList
    .map((p, idx) => ({ p, idx }))
    .filter((e) => e.p > 0)
    .sort((a, b) => b.p - a.p);
  const slice = typeof limit === "number" ? entries.slice(0, limit) : entries;
  if (entries.length === 0) {
    targetEl.textContent = "N/A";
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

function renderMctsList(mcts) {
  if (!mctsListEl) return;
  if (!mcts || !Array.isArray(mcts.children) || mcts.children.length === 0) {
    mctsListEl.textContent = "N/A";
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
  // Update advantage bar.
  if (value === null || value === undefined) {
    evalText.textContent = "AZ eval: N/A";
    evalFill.style.left = "50%";
    evalFill.style.width = "0%";
    return;
  }
  const youValue = -value;
  const youText = humanPlayer === -1 ? ` | 你方: ${youValue.toFixed(2)}` : "";
  evalText.textContent = `AZ eval (黑): ${value.toFixed(2)}${youText}`;
  const clamped = Math.max(-1, Math.min(1, value));
  const pct = Math.abs(clamped) * 50;
  const left = clamped >= 0 ? 50 : 50 - pct;
  evalFill.style.left = `${left}%`;
  evalFill.style.width = `${pct}%`;
}

function setStatus(text) {
  statusText.textContent = text;
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

function setAiMoveProb(probList, aiMoveIdx) {
  if (!aiProbEl) return;
  if (!Array.isArray(probList) || aiMoveIdx === null || aiMoveIdx === undefined) {
    aiProbEl.textContent = "AI move prob: N/A";
    return;
  }
  const p = probList[aiMoveIdx] || 0;
  const row = Math.floor(aiMoveIdx / size) + 1;
  const col = (aiMoveIdx % size) + 1;
  aiProbEl.textContent = `AI move prob: ${(p * 100).toFixed(2)}% @ (${row}, ${col})`;
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

function showResultModal(result) {
  if (!result || result.state === "ongoing") return;
  resultTitle.textContent = "Game Over";
  if (result.state === "draw") {
    resultBody.textContent = "Draw game.";
  } else if (result.winner === humanPlayer) {
    resultBody.textContent = "You win!";
  } else {
    resultBody.textContent = "AI wins.";
  }
  resultModal.classList.remove("hidden");
}

function hideResultModal() {
  resultModal.classList.add("hidden");
}

async function newGame() {
  // Create a new server-side game.
  setStatus("Creating game...");
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
    }),
  });
  const data = await res.json();
  gameId = data.game_id;
  board = data.board;
  resultState = data.result.state;
  updateHumanPlayer(getHumanSide());
  lastAiMove = detectAiMove(lastBoard, board, -humanPlayer);
  lastMove = null;
  setPolicy(data.policy);
  setAiMoveProb(data.policy, lastAiMove);
  setEval(data.eval);
  lastMcts = data.mcts || null;
  renderMctsList(lastMcts);
  setStatus(resultState === "ongoing" ? "Your move." : "Game over.");
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
  lastBoard = board.slice();
  const res = await fetch("/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId, action: idx }),
  });
  if (!res.ok) return;
  const data = await res.json();
  board = data.board;
  resultState = data.result.state;
  lastMove = idx;
  lastAiMove = detectAiMove(lastBoard, board, -humanPlayer);
  setPolicy(data.policy);
  setAiMoveProb(data.policy, lastAiMove);
  setEval(data.eval);
  lastMcts = data.mcts || null;
  renderMctsList(lastMcts);
  if (resultState === "ongoing") {
    setStatus("Your move.");
  } else if (data.result.state === "draw") {
    setStatus("Draw.");
  } else if (data.result.winner === humanPlayer) {
    setStatus("You win!");
  } else {
    setStatus("AI wins.");
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
    deviceInfo.textContent = `Server: ${data.device}`;
  }
  if (data.az_model) azModelInput.value = data.az_model;
  if (typeof data.model_loaded === "boolean") {
    modelInfo.textContent = data.model_loaded ? "Model: loaded" : "Model: not loaded";
  }
  if (data.model_file && data.model_file.created) {
    const created = data.model_file.created;
    const modified = data.model_file.modified ? `, modified ${data.model_file.modified}` : "";
    modelTime.textContent = `Model file created ${created}${modified}`;
  } else {
    modelTime.textContent = "Model file: N/A";
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
window.addEventListener("resize", resizeBoard);

resizeBoard();
loadConfig();
