from __future__ import annotations

import argparse
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

# 引用新的串行 MCTS
from az_mcts import AZMCTS
from az_net import AZNet, encode_board
from env import BOARD_AREA, Gomoku
import torch # 确保 import torch

def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="web",
        static_url_path="/static",
        template_folder="web",
    )
    return app


class GameState:
    def __init__(
        self,
        game_id: str,
        human_player: int,
        sims: int,
        az_model: Optional[AZNet],
        analysis_only: bool = False,
    ) -> None:
        self.game_id = game_id
        self.env = Gomoku()
        self.human_player = human_player
        self.ai_player = -human_player
        self.sims = sims
        self.az_model = az_model
        self.analysis_only = analysis_only
        self.history: list[tuple[int, int]] = []
        self.eval_history: list[float] = []
        self.last_eval: Optional[float] = None
        
        # 修正：初始化 MCTS 时不传 eval_batch_size
        if az_model:
            device = next(az_model.parameters()).device
            self.az_mcts = AZMCTS(
                model=az_model,
                sims=sims,
                device=device,
            )
        else:
            self.az_mcts = None

    def reset(self) -> None:
        self.env.reset()
        if self.az_mcts:
            self.az_mcts.root = None # 重置搜索树
        self.history = []
        self.eval_history = []
        self.last_eval = None

    def apply_move(
        self,
        action: int,
        player: int,
        track_history: bool = True,
        advance_mcts: bool = True,
    ) -> None:
        self.env.step(action, player)
        if track_history:
            self.history.append((action, player))
        if advance_mcts and self.az_mcts is not None:
            self.az_mcts.advance(action, self.env)
        self.record_eval()

    def undo(self, steps: int) -> bool:
        if steps <= 0 or steps > len(self.history):
            return False
        del self.history[-steps:]
        self.env.reset()
        if self.az_mcts is not None:
            self.az_mcts.root = None
        self.eval_history = []
        self.last_eval = None
        for action, player in self.history:
            self.env.step(action, player)
            self.record_eval()
        return True

    def record_eval(self) -> Optional[float]:
        if self.az_model is None:
            self.last_eval = None
            return None
        to_play = 1 if (self.env.move_count % 2 == 0) else -1
        board_t = encode_board(
            self.env.board,
            to_play,
            device=next(self.az_model.parameters()).device,
        )
        with torch.no_grad():
            _, value = self.az_model(board_t.unsqueeze(0))
        v_to_play = float(value.item())
        v_x = v_to_play if to_play == 1 else -v_to_play
        self.last_eval = v_x
        self.eval_history.append(v_x)
        return v_x


def load_az_model(path: str, device: str) -> Optional[AZNet]:
    if not os.path.exists(path):
        return None
    model = AZNet().to(torch.device(device))
    # 增加 map_location 防止 CUDA 模型在 CPU 机器加载报错
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def get_model_file_info(path: str) -> Optional[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return None
    created = datetime.fromtimestamp(os.path.getctime(path)).isoformat(timespec="seconds")
    modified = datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    return {"created": created, "modified": modified}


MODEL_EXTS = {".pt", ".pth"}


def normalize_model_path(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        return p.as_posix()
    return p.as_posix()


def list_available_models(artifacts_dir: str, extra_paths: Optional[list[str]] = None) -> list[Dict]:
    models: list[Dict] = []
    seen: set[str] = set()
    base = Path(artifacts_dir)
    cwd = Path.cwd()
    if base.exists() and base.is_dir():
        for root, _, files in os.walk(base):
            for name in files:
                if Path(name).suffix.lower() not in MODEL_EXTS:
                    continue
                full_path = Path(root) / name
                try:
                    rel_path = full_path.relative_to(cwd)
                    path_str = rel_path.as_posix()
                except ValueError:
                    path_str = full_path.as_posix()
                if path_str in seen:
                    continue
                seen.add(path_str)
                models.append(
                    {
                        "path": path_str,
                        "exists": True,
                        "file": get_model_file_info(str(full_path)),
                    }
                )
    if extra_paths:
        for path in extra_paths:
            if not path:
                continue
            norm_path = normalize_model_path(path)
            if norm_path in seen:
                continue
            exists = os.path.exists(path)
            models.append(
                {
                    "path": norm_path,
                    "exists": bool(exists),
                    "file": get_model_file_info(path) if exists else None,
                }
            )
            seen.add(norm_path)
    models.sort(key=lambda item: item["path"])
    return models


def encode_result(result: Optional[int]) -> Dict:
    # Convert winner into a simple JSON payload.
    if result is None:
        return {"state": "ongoing", "winner": 0}
    if result == 0:
        return {"state": "draw", "winner": 0}
    return {"state": "win", "winner": int(result)}


def append_move(game: GameState, action: int, player: int) -> None:
    # Apply move and update tree state.
    game.apply_move(action, player, track_history=True, advance_mcts=True)


def ai_choose_action(game: GameState) -> tuple[int, Optional[list[float]], Optional[dict]]:
    # AlphaZero move (falls back to random if no model). Returns (action, policy, mcts).
    available = game.env.candidate_actions()
    if not available:
        available = [i for i, v in enumerate(game.env.board.ravel()) if v == 0]
    pi: Optional[list[float]] = None
    mcts_stats: Optional[dict] = None
    if game.az_mcts is not None:
        pi = game.az_mcts.run(game.env, player=game.ai_player, add_noise=False)
        mcts_stats = game.az_mcts.last_root_stats
        action = game.az_mcts.select_action(pi, temp=0.0)
        r, c = divmod(action, game.env.size)
        if game.env.board[r, c] != 0:
            action = random.choice(available)
        return action, pi, mcts_stats
    return random.choice(available), pi, mcts_stats


def evaluate_board(game: GameState) -> Optional[float]:
    # Use AZ value head for the advantage chart (X perspective).
    if game.last_eval is not None:
        return game.last_eval
    if game.env.move_count == 0:
        return None
    return game.record_eval()


def analyze_position(game: GameState) -> tuple[Optional[list[float]], Optional[dict]]:
    # Return policy and MCTS stats for the side to move without placing a stone.
    if game.az_mcts is None:
        return None, None
    to_play = 1 if (game.env.move_count % 2 == 0) else -1
    pi = game.az_mcts.run(game.env, player=to_play, add_noise=False)
    return pi, game.az_mcts.last_root_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Gomoku Web GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--az-model", default=os.path.join("artifacts", "az_model.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    app = create_app()
    games: Dict[str, GameState] = {}
    config = {
        "sims": args.sims,
        "device": args.device,
        "az_model": args.az_model,
    }
    az_model = load_az_model(config["az_model"], config["device"])

    @app.route("/")
    def index() -> str:
        # Serve the web UI.
        return render_template("index.html")

    @app.route("/assets/<path:filename>")
    def assets(filename: str):
        # Optional asset endpoint if you add images later.
        return send_from_directory(os.path.join(app.root_path, "web", "assets"), filename)

    @app.route("/api/new", methods=["POST"])
    def api_new():
        # Start a new game; AI can move first if human plays O.
        data = request.get_json(silent=True) or {}
        sims = int(data.get("sims", config["sims"]))
        device = data.get("device")
        az_path = data.get("az_model")
        analysis_only = bool(data.get("analysis_only", False))
        if sims > 0:
            config["sims"] = sims
        if isinstance(device, str) and device:
            config["device"] = device
        if isinstance(az_path, str) and az_path:
            config["az_model"] = az_path
        nonlocal az_model
        az_model = load_az_model(config["az_model"], config["device"])
        human = data.get("human", "x")
        human_player = 1 if str(human).lower() == "x" else -1

        game_id = uuid.uuid4().hex
        game = GameState(
            game_id=game_id,
            human_player=human_player,
            sims=sims,
            az_model=az_model,
            analysis_only=analysis_only,
        )
        games[game_id] = game

        if not analysis_only and human_player == -1:
            action, pi, mcts_stats = ai_choose_action(game)
            append_move(game, action, game.ai_player)
        else:
            pi, mcts_stats = analyze_position(game) if analysis_only else (None, None)

        result = game.env.check_winner()
        next_player = 1 if (game.env.move_count % 2 == 0) else -1
        return jsonify(
            {
                "game_id": game_id,
                "board": game.env.board.astype(int).ravel().tolist(),
                "result": encode_result(result),
                "next_player": next_player,
                "eval": evaluate_board(game),
                "eval_history": game.eval_history,
                "policy": pi,
                "mcts": mcts_stats,
                "analysis_only": analysis_only,
                "config": {
                    "sims": config["sims"],
                    "device": config["device"],
                    "az_model": config["az_model"],
                    "model_loaded": az_model is not None,
                    "model_file": get_model_file_info(config["az_model"]),
                    "models": list_available_models("artifacts", [config["az_model"]]),
                },
            }
        )

    @app.route("/api/config", methods=["GET"])
    def api_config():
        # Expose server defaults for the web UI.
        return jsonify(
            {
                "sims": config["sims"],
                "device": config["device"],
                "az_model": config["az_model"],
                "model_loaded": az_model is not None,
                "model_file": get_model_file_info(config["az_model"]),
                "models": list_available_models("artifacts", [config["az_model"]]),
            }
        )

    @app.route("/api/config", methods=["POST"])
    def api_config_update():
        # Update server defaults and reload model if needed.
        nonlocal az_model
        data = request.get_json(silent=True) or {}
        sims = data.get("sims")
        device = data.get("device")
        az_path = data.get("az_model")

        if isinstance(sims, int) and sims > 0:
            config["sims"] = sims
        if isinstance(device, str) and device:
            config["device"] = device
        if isinstance(az_path, str) and az_path:
            config["az_model"] = az_path

        az_model = load_az_model(config["az_model"], config["device"])
        return jsonify(
            {
                "sims": config["sims"],
                "device": config["device"],
                "az_model": config["az_model"],
                "model_loaded": az_model is not None,
                "model_file": get_model_file_info(config["az_model"]),
                "models": list_available_models("artifacts", [config["az_model"]]),
            }
        )

    @app.route("/api/move", methods=["POST"])
    def api_move():
        # Apply a human move, then let the AI respond.
        data = request.get_json(silent=True) or {}
        game_id = data.get("game_id")
        action = data.get("action")
        if game_id not in games or action is None:
            return jsonify({"error": "invalid"}), 400

        game = games[game_id]
        try:
            action = int(action)
        except ValueError:
            return jsonify({"error": "invalid"}), 400
        if action < 0 or action >= BOARD_AREA:
            return jsonify({"error": "invalid"}), 400
        r, c = divmod(action, game.env.size)
        if game.env.board[r, c] != 0:
            return jsonify({"error": "invalid"}), 400

        if game.analysis_only:
            to_play = 1 if (game.env.move_count % 2 == 0) else -1
            append_move(game, action, to_play)
        else:
            append_move(game, action, game.human_player)
        result = game.env.check_winner()
        policy = None
        mcts_stats = None
        if result is None and not game.analysis_only:
            ai_action, policy, mcts_stats = ai_choose_action(game)
            append_move(game, ai_action, game.ai_player)
            result = game.env.check_winner()
        elif result is None and game.analysis_only:
            policy, mcts_stats = analyze_position(game)

        next_player = 1 if (game.env.move_count % 2 == 0) else -1
        return jsonify(
            {
                "game_id": game_id,
                "board": game.env.board.astype(int).ravel().tolist(),
                "result": encode_result(result),
                "next_player": next_player if game.analysis_only else game.human_player,
                "eval": evaluate_board(game),
                "eval_history": game.eval_history,
                "policy": policy,
                "mcts": mcts_stats,
                "analysis_only": game.analysis_only,
            }
        )

    @app.route("/api/undo", methods=["POST"])
    def api_undo():
        data = request.get_json(silent=True) or {}
        game_id = data.get("game_id")
        steps = data.get("steps", 1)
        if game_id not in games:
            return jsonify({"error": "invalid"}), 400
        try:
            steps = int(steps)
        except ValueError:
            return jsonify({"error": "invalid"}), 400
        game = games[game_id]
        if steps > len(game.history):
            steps = len(game.history)
        if steps == 0:
            return jsonify({"error": "invalid"}), 400
        if not game.undo(steps):
            return jsonify({"error": "invalid"}), 400

        result = game.env.check_winner()
        policy = None
        mcts_stats = None
        if result is None and game.analysis_only:
            policy, mcts_stats = analyze_position(game)

        next_player = 1 if (game.env.move_count % 2 == 0) else -1
        return jsonify(
            {
                "game_id": game_id,
                "board": game.env.board.astype(int).ravel().tolist(),
                "result": encode_result(result),
                "next_player": next_player if game.analysis_only else game.human_player,
                "eval": evaluate_board(game),
                "eval_history": game.eval_history,
                "policy": policy,
                "mcts": mcts_stats,
                "analysis_only": game.analysis_only,
            }
        )

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
