# Gomoku Reinforcement Learning (Web UI)

This project provides a Gomoku (15x15, 5-in-a-row) environment with an AlphaZero-like training loop and a Flask-based web UI.

## Setup

Create a virtual environment (optional) and install dependencies:

```bash
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Train

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
uv run src/az_train.py --workers 4 --selfplay-device cpu --device cuda
```

Resume training from a saved model:

```bash
uv run src/az_train.py --resume --model-path artifacts/az_model.pt
```

When resuming, new checkpoints are saved as `*_r_iterXXXX.pt`. You can override the prefix:

```bash
uv run src/az_train.py --resume --model-path artifacts/az_model.pt --save-prefix artifacts/az_model_run2
```

To save periodic checkpoints, use:

```bash
uv run src/az_train.py --save-every 10
```

To keep the best model during evaluation rounds, use:

```bash
uv run src/az_train.py --save-best --best-path artifacts/az_model_best.pt
```

MCTS pruning and progress:

```bash
uv run src/az_train.py --prune-radius 2
```

Training uses `tqdm` for a progress bar when installed.

## Web UI

```bash
uv run src/web_app.py --sims 800 --device cpu
```

Open `http://127.0.0.1:5000` and choose whether you play first (X) or second (O).
The UI shows the server's current sims and device settings, and lets you change
device/model path with Apply.

Play with a trained model:

```bash
uv run src/web_app.py --az-model artifacts/az_model.pt
```

The UI shows a simple advantage bar using the AlphaZero value head (X perspective).
