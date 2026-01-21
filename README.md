# Gomoku Reinforcement Learning (with Visualization)

This project provides a Gomoku (15x15, 5-in-a-row) environment with an AlphaZero-like training loop and a Tkinter GUI.

## Setup

Create a virtual environment (optional) and install dependencies:

```bash
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Tkinter ships with most Python distributions on Windows/macOS/Linux. If it is missing on Linux, install your distro's `python3-tk` package.

## Train

## Visualize (GUI)

```bash
uv run src/gui.py
```

Click a grid intersection to play as X. The AI plays O using the AlphaZero-style search (falls back to random if no model is loaded).

## AlphaZero-like Training

This version uses a small policy/value network plus MCTS (PUCT) and self-play.

Train (AlphaZero-like):

```bash
uv run src/az_train.py --iterations 30 --games-per-iter 30 --sims 200 --device cuda:0
```

Single GPU on Linux:

```bash
uv run src/az_train.py --device cuda
```

Parallel self-play on CPU (keeps GPU for training):

```bash
uv run src/az_train.py --workers 4 --selfplay-device cpu --device cuda --eval-batch 32
```


Resume training from a saved model:

```bash
uv run src/az_train.py --resume --model-path artifacts/az_model.pt
```

When resuming, new checkpoints are saved as `*_r_iterXXXX.pt` and the lineage is
logged to `artifacts/az_model_lineage.jsonl`. You can override the prefix:

```bash
uv run src/az_train.py --resume --model-path artifacts/az_model.pt --save-prefix artifacts/az_model_run2
```

To save periodic checkpoints, use:

```bash
uv run src/az_train.py --save-every 10
```

To keep the best model (lowest training loss), use:

```bash
uv run src/az_train.py --save-best --best-path artifacts/az_model_best.pt
```

MCTS pruning and progress:

```bash
uv run src/az_train.py --prune-radius 2
```

Training uses `tqdm` for a progress bar when installed.

To enable a simple win/block heuristic in MCTS:

```bash
uv run src/az_train.py --use-heuristic
```


Play with the trained model:

```bash
uv run src/gui.py --ai az --az-model artifacts/az_model.pt
```

The GUI shows a simple advantage bar using the AlphaZero value head (X perspective).

## Web GUI

Run the web app:

```bash
uv run src/web_app.py --sims 800 --device cpu
```

Open `http://127.0.0.1:5000` and choose whether you play first (X) or second (O).
The UI shows the server's current sims and device settings, and lets you change
device/model path with Apply.
