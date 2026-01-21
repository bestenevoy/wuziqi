from __future__ import annotations

import argparse
import os
import random
import uuid
from datetime import datetime
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
    ) -> None:
        self.game_id = game_id
        self.env = Gomoku()
        self.human_player = human_player
        self.ai_player = -human_player
        self.sims = sims
        self.az_model = az_model
        
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


def encode_result(result: Optional[int]) -> Dict:
    # Convert winner into a simple JSON payload.
    if result is None:
        return {"state": "ongoing", "winner": 0}
    if result == 0:
        return {"state": "draw", "winner": 0}
    return {"state": "win", "winner": int(result)}


def append_move(game: GameState, action: int, player: int) -> None:
    # Apply move and update tree state.
    game.env.step(action, player)
    if game.az_mcts is not None:
        game.az_mcts.advance(action, game.env)


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
        return game.az_mcts.select_action(pi, temp=0.0), pi, mcts_stats
    return random.choice(available), pi, mcts_stats


def evaluate_board(game: GameState) -> Optional[float]:
    # Use AZ value head for the advantage bar (X perspective).
    if game.az_model is None:
        return None
    import torch

    to_play = 1 if (game.env.move_count % 2 == 0) else -1
    board_t = encode_board(
        game.env.board,
        to_play,
        device=next(game.az_model.parameters()).device,
    )
    with torch.no_grad():
        _, value = game.az_model(board_t.unsqueeze(0))
    v_to_play = float(value.item())
    v_x = v_to_play if to_play == 1 else -v_to_play
    return v_x


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
        )
        games[game_id] = game

        if human_player == -1:
            action, pi, mcts_stats = ai_choose_action(game)
            append_move(game, action, game.ai_player)
        else:
            pi = None
            mcts_stats = None

        result = game.env.check_winner()
        return jsonify(
            {
                "game_id": game_id,
                "board": game.env.board.astype(int).ravel().tolist(),
                "result": encode_result(result),
                "next_player": 1,
                "eval": evaluate_board(game),
                "policy": pi,
                "mcts": mcts_stats,
                "config": {
                    "sims": config["sims"],
                    "device": config["device"],
                    "az_model": config["az_model"],
                    "model_loaded": az_model is not None,
                    "model_file": get_model_file_info(config["az_model"]),
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

        append_move(game, action, game.human_player)
        result = game.env.check_winner()
        policy = None
        mcts_stats = None
        if result is None:
            ai_action, policy, mcts_stats = ai_choose_action(game)
            append_move(game, ai_action, game.ai_player)
            result = game.env.check_winner()

        return jsonify(
            {
                "game_id": game_id,
                "board": game.env.board.astype(int).ravel().tolist(),
                "result": encode_result(result),
                "next_player": game.human_player,
                "eval": evaluate_board(game),
                "policy": policy,
                "mcts": mcts_stats,
            }
        )

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
