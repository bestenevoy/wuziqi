from __future__ import annotations

import argparse
import os
import random
import multiprocessing as mp
import queue
import itertools
import time
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from az_mcts import AZMCTS
from az_net import AZNet, encode_board
from env import BOARD_AREA, BOARD_SIZE, Gomoku

BATCH_SIZE = 128

@dataclass
class Sample:
    board: np.ndarray
    player: int
    pi: np.ndarray
    value: float

    def __getstate__(self) -> tuple[np.ndarray, np.ndarray, int, float]:
        return (self.board, self.pi, int(self.player), float(self.value))

    def __setstate__(self, state: tuple[np.ndarray, np.ndarray, int, float]) -> None:
        board, pi, player, value = state
        self.board = board
        self.player = int(player)
        self.pi = pi
        self.value = float(value)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self.data: deque[Sample] = deque(maxlen=self.capacity or None)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, samples: List[Sample]) -> None:
        if self.capacity <= 0:
            return
        for sample in samples:
            self.data.append(sample)

    def sample(self, count: int) -> List[Sample]:
        if count <= 0 or count >= len(self.data):
            return list(self.data)
        return random.sample(list(self.data), count)


class GomokuDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        size: int = BOARD_SIZE,
        full_augment: bool = True,
    ) -> None:
        self.samples = samples
        self.size = size
        self.full_augment = full_augment

    def __len__(self) -> int:
        if self.full_augment:
            return len(self.samples) * 8
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, float]:
        if self.full_augment:
            sample_idx = idx // 8
            aug_idx = idx % 8
        else:
            sample_idx = idx
            aug_idx = random.randint(0, 7)
        sample = self.samples[sample_idx]
        board = sample.board.reshape(self.size, self.size)
        pi = sample.pi.reshape(self.size, self.size)

        rot = aug_idx % 4
        flip = aug_idx // 4 == 1
        if flip:
            board = np.flip(board, axis=1)
            pi = np.flip(pi, axis=1)
        if rot:
            board = np.rot90(board, k=rot)
            pi = np.rot90(pi, k=rot)

        board = np.ascontiguousarray(board)
        pi = np.ascontiguousarray(pi, dtype=np.float32).reshape(-1)
        return (
            torch.tensor(board),
            torch.tensor(pi),
            sample.player,
            float(sample.value),
        )


def self_play(
    model: AZNet,
    sims: int,
    device: torch.device,
    add_noise: bool,
    prune_radius: int,
    temp_steps: int,
    temp: float,
) -> List[Sample]:
    env = Gomoku()
    env.nearby_radius = prune_radius
    was_training = model.training
    model.eval()
    
    mcts = AZMCTS(
        model=model,
        sims=sims,
        device=device,
        prune_radius=prune_radius,
    )
    history: List[Tuple[np.ndarray, int, List[float]]] = []
    player = 1

    def _apply_discount(samples: List[Sample], winner: int) -> None:
        if winner == 0:
            return
        reward = 1.0
        for s in reversed(samples):
            if s.player == winner:
                s.value = min(1.0, s.value + reward)
            else:
                s.value = max(-1.0, s.value - reward)
            reward *= 0.95
            reward = max(0.2, reward)

    try:
        while True:
            pi = mcts.run(env, player=player, add_noise=add_noise)
            
            if sum(pi) <= 1e-12:
                available = env.candidate_actions()
                if not available:
                    available = [i for i, v in enumerate(env.board.ravel()) if v == 0]
                if not available:
                    result = 0
                    samples = []
                    for board, p, pi_hist in history:
                        samples.append(Sample(board=board, player=p, pi=np.array(pi_hist), value=0.0))
                    return samples
                fallback_action = random.choice(available)
                pi = [0.0] * BOARD_AREA
                pi[fallback_action] = 1.0
            else:
                fallback_action = None
                
            step_temp = temp if env.move_count < temp_steps else 0.0
            if fallback_action is not None:
                action = fallback_action
            else:
                action = mcts.select_action(pi, temp=step_temp)
            
            if step_temp <= 1e-3:
                pi_store = [0.0] * BOARD_AREA
                pi_store[action] = 1.0
            else:
                total = sum(pi)
                pi_store = [p/total for p in pi]
                
            history.append((env.board.copy(), player, pi_store))
            env.step(action, player)
            mcts.advance(action, env)

            result = env.check_winner()
            if result is not None:
                samples = []
                for board, p, pi_hist in history:
                    if result == 0:
                        value = 0.0
                    else:
                        value = 1.0 if result == p else -1.0
                    samples.append(Sample(board=board, player=p, pi=np.array(pi_hist), value=value))
                _apply_discount(samples, result)
                return samples

            player *= -1
    finally:
        if was_training:
            model.train()


def train_epoch(
    model: AZNet,
    samples: Sequence[Sample],
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    value_weight: float,
    full_augment: bool,
) -> float:
    dataset = GomokuDataset(samples, full_augment=full_augment)
    data_workers = min(4, (os.cpu_count() or 1))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    total_loss = 0.0
    for history, target_pi, player, target_v in dataloader:
        legal_mask = (history.reshape(history.size(0), -1) == 0).to(device)
        boards = torch.stack(
            [encode_board(b, int(p), device=device) for b, p in zip(history, player)]
        )
        target_pi = target_pi.to(device=device, dtype=torch.float32)
        target_v = target_v.to(device=device, dtype=torch.float32)
        
        policy_logits, values = model(boards)
        
        masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=1)
        target_pi = target_pi * legal_mask.float()
        target_pi = target_pi / target_pi.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values, target_v)
        loss = policy_loss + value_weight * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(dataloader))


def _self_play_batch(args: tuple) -> List[Sample]:
    (
        state_dict,
        sims,
        device_str,
        prune_radius,
        temp_steps,
        temp,
        games,
        seed,
    ) = args
    if games <= 0:
        return []
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    
    # 恢复 GPU 支持
    device = torch.device(device_str) 
    model = AZNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    buffer: List[Sample] = []
    for _ in range(games):
        buffer.extend(
            self_play(
                model,
                sims=sims,
                device=device,
                add_noise=True,
                prune_radius=prune_radius,
                temp_steps=temp_steps,
                temp=temp,
            )
        )
    return buffer


def _self_play_worker(
    shared: "mp.managers.DictProxy",
    out_queue: mp.Queue,
    stop_event: mp.Event,
    sims: int,
    device_str: str,
    prune_radius: int,
    temp_steps: int,
    temp: float,
    seed: int,
) -> None:
    # 路径修复保持，防止 ModuleNotFound
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    pid = os.getpid()
    # 打印启动日志
    print(f"[Worker {pid}] Started on {device_str}. Loading model...", flush=True)

    try:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        
        # 恢复 GPU 支持: 使用传入的 device_str
        worker_device = torch.device(device_str)
        model = AZNet().to(worker_device)
        model.eval()

        local_version = -1
        games_played = 0

        while not stop_event.is_set():
            # 降低同步频率：每10局同步一次
            if games_played % 10 == 0:
                try:
                    version = shared.get("version", -1)
                    if version > local_version:
                        state = shared.get("state")
                        if state:
                            # 确保 state 加载到正确的 worker_device
                            model.load_state_dict(state)
                            local_version = version
                except Exception:
                    pass

            try:
                samples = self_play(
                    model,
                    sims=sims,
                    device=worker_device,
                    add_noise=True,
                    prune_radius=prune_radius,
                    temp_steps=temp_steps,
                    temp=temp,
                )
                out_queue.put(samples, timeout=10.0)
                games_played += 1
                
            except queue.Full:
                time.sleep(0.1)
            except Exception as e:
                print(f"[Worker {pid}] Game Error: {e}", flush=True)
                traceback.print_exc()
                time.sleep(1.0)
                
    except Exception as e:
        print(f"[Worker {pid}] FATAL INIT ERROR: {e}", flush=True)
        traceback.print_exc()


def _play_eval_game(
    model_x: AZNet,
    model_o: AZNet,
    sims: int,
    device: torch.device,
    prune_radius: int,
    eval_temp: float,
    eval_noise: bool,
    eval_temp_steps: int,
) -> int:
    env = Gomoku()
    env.nearby_radius = prune_radius
    
    mcts_x = AZMCTS(
        model=model_x,
        sims=sims,
        device=device,
        prune_radius=prune_radius,
    )
    mcts_o = AZMCTS(
        model=model_o,
        sims=sims,
        device=device,
        prune_radius=prune_radius,
    )
    player = 1
    while True:
        mcts = mcts_x if player == 1 else mcts_o
        pi = mcts.run(env, player=player, add_noise=eval_noise)
        step_temp = eval_temp if env.move_count < eval_temp_steps else 0.0
        action = mcts.select_action(pi, temp=step_temp)
        env.step(action, player)
        mcts_x.advance(action, env)
        mcts_o.advance(action, env)
        result = env.check_winner()
        if result is not None:
            return result
        player *= -1


def _eval_game_worker(args: tuple) -> int:
    (
        current_state,
        best_state,
        current_is_x,
        sims,
        device_str,
        prune_radius,
        eval_temp,
        eval_noise,
        eval_temp_steps,
    ) = args
    device = torch.device(device_str)
    model_x = AZNet().to(device)
    model_o = AZNet().to(device)
    if current_is_x:
        model_x.load_state_dict(current_state)
        model_o.load_state_dict(best_state)
    else:
        model_x.load_state_dict(best_state)
        model_o.load_state_dict(current_state)
    model_x.eval()
    model_o.eval()
    return _play_eval_game(
        model_x,
        model_o,
        sims,
        device,
        prune_radius,
        eval_temp,
        eval_noise,
        eval_temp_steps,
    )


def eval_models(
    current: AZNet,
    best: AZNet,
    games: int,
    sims: int,
    device: torch.device,
    prune_radius: int,
    eval_temp: float,
    eval_noise: bool,
    eval_temp_steps: int,
    devices: list[str] | None,
    workers: int,
) -> tuple[float, int, int, int]:
    current_was_training = current.training
    best_was_training = best.training
    current.eval()
    best.eval()
    if not devices:
        devices = [str(device)]
    state_current = {k: v.detach().cpu() for k, v in current.state_dict().items()}
    state_best = {k: v.detach().cpu() for k, v in best.state_dict().items()}
    tasks = []
    for i in range(games):
        current_is_x = i % 2 == 0
        device_str = devices[i % len(devices)]
        tasks.append(
            (
                state_current,
                state_best,
                current_is_x,
                sims,
                device_str,
                prune_radius,
                eval_temp,
                eval_noise,
                eval_temp_steps,
            )
        )
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, workers)) as pool:
        results = pool.map(_eval_game_worker, tasks)
    
    score = 0.0
    wins = 0
    draws = 0
    losses = 0
    for i, result in enumerate(results):
        if result == 0:
            score += 0.5
            draws += 1
            continue
        if i % 2 == 0:
            if result == 1:
                score += 1.0
                wins += 1
            else:
                losses += 1
        else:
            if result == -1:
                score += 1.0
                wins += 1
            else:
                losses += 1
    
    if current_was_training:
        current.train()
    if best_was_training:
        best.train()
    return score / max(1, games), wins, draws, losses


def update_elo(rating: float, score: float, expected: float, k: float) -> float:
    return rating + k * (score - expected)


def main() -> None:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="AlphaZero-like training for Gomoku.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--train-forever", action="store_true")
    parser.add_argument("--games-per-iter", type=int, default=50)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--value-weight", type=float, default=1.5)
    parser.add_argument("--model-path", default=os.path.join("artifacts", "az_model.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--latest-model-path", default=None)
    parser.add_argument("--save-prefix", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--best-path", default=os.path.join("artifacts", "az_model_best.pt"))
    parser.add_argument("--log-path", default=os.path.join("artifacts", "az_train.log"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--selfplay-device", default="cpu")
    parser.add_argument("--selfplay-devices", default=None)
    parser.add_argument("--async-selfplay", action="store_true")
    parser.add_argument("--async-queue", type=int, default=16)
    parser.add_argument("--prune-radius", type=int, default=2)
    parser.add_argument("--temp-steps", type=int, default=8)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--replay-sample", type=int, default=BATCH_SIZE * 32)
    parser.add_argument("--eval-games", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-threshold", type=float, default=0.55)
    parser.add_argument("--eval-sims", type=int, default=2000)
    parser.add_argument("--eval-temp", type=float, default=0.0)
    parser.add_argument("--eval-noise", action="store_true")
    parser.add_argument("--eval-temp-steps", type=int, default=2)
    parser.add_argument("--eval-devices", default=None)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--elo-k", type=float, default=16.0)
    parser.add_argument("--elo-init", type=float, default=1000.0)
    aug_group = parser.add_mutually_exclusive_group()
    aug_group.add_argument("--full-augment", action="store_true")
    aug_group.add_argument("--random-augment", action="store_true")
    args = parser.parse_args()
    
    # 逻辑处理：eval sims
    eval_sims = args.eval_sims if args.eval_sims > 0 else args.sims
    eval_devices = None
    if args.eval_devices:
        eval_devices = [d.strip() for d in args.eval_devices.split(",") if d.strip()]

    # 逻辑处理：selfplay devices
    if args.selfplay_devices:
        selfplay_devices = [d.strip() for d in args.selfplay_devices.split(",") if d.strip()]
    else:
        selfplay_devices = [args.selfplay_device]

    device = torch.device(args.device)
    model = AZNet().to(device)
    if args.resume and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Resumed from {args.model_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 修复：移除 verbose=True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=0.0,
    )

    os.makedirs("artifacts", exist_ok=True)
    log_path = args.log_path
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    if args.save_prefix:
        save_prefix = args.save_prefix
    else:
        base = os.path.splitext(args.model_path)[0]
        save_prefix = f"{base}_r" if args.resume else base

    best_model = None
    eval_winrate_ema = None
    current_elo = args.elo_init
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.latest_model_path:
        latest_path = args.latest_model_path
    else:
        base = os.path.splitext(args.model_path)[0]
        latest_path = f"{base}_latest_{run_stamp}.pt"
    os.makedirs(os.path.dirname(latest_path) or ".", exist_ok=True)
    
    if args.train_forever:
        iter_range = itertools.count(1)
    else:
        iter_range = range(1, args.iterations + 1)
    if tqdm:
        iter_range = tqdm(iter_range, desc="Iterations")
        
    replay = ReplayBuffer(args.replay_size)
    if args.save_best and args.eval_games > 0 and os.path.exists(args.best_path):
        best_model = AZNet().to(device)
        best_model.load_state_dict(torch.load(args.best_path, map_location=device))

    ctx = mp.get_context("spawn")
    async_workers = []
    async_queue = None
    async_shared = None
    async_stop = None
    
    if args.async_selfplay:
        async_queue = ctx.Queue(maxsize=max(1, args.async_queue))
        async_stop = ctx.Event()
        manager = mp.Manager()
        async_shared = manager.dict()
        async_shared["version"] = 0
        async_shared["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        worker_count = max(1, args.workers)
        
        print(f"Starting {worker_count} workers...")
        
        for w in range(worker_count):
            # 轮询分配设备
            worker_device_str = selfplay_devices[w % len(selfplay_devices)]
            p = ctx.Process(
                target=_self_play_worker,
                args=(
                    async_shared,
                    async_queue,
                    async_stop,
                    args.sims,
                    worker_device_str, # 传入指定的 GPU/CPU
                    args.prune_radius,
                    args.temp_steps,
                    args.temp,
                    1000 + w,
                ),
            )
            p.start()
            async_workers.append(p)
            
    try:
        for it in iter_range:
            buffer: List[Sample] = []
            collect_start = time.monotonic()
            
            if args.async_selfplay:
                games_collected = 0
                while games_collected < args.games_per_iter:
                    try:
                        batch = async_queue.get(timeout=2.0)
                    except queue.Empty:
                        continue
                    if not batch:
                        continue
                    buffer.extend(batch)
                    games_collected += 1
            elif args.workers > 1:
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                worker_count = min(args.workers, args.games_per_iter)
                games_left = args.games_per_iter
                tasks = []
                for w in range(worker_count):
                    g = games_left // (worker_count - w)
                    games_left -= g
                    if g <= 0: continue
                    # 同步模式的设备分配
                    d_str = selfplay_devices[w % len(selfplay_devices)]
                    tasks.append(
                        (
                            state,
                            args.sims,
                            d_str,
                            args.prune_radius,
                            args.temp_steps,
                            args.temp,
                            g,
                            1000 * it + w,
                        )
                    )
                with ctx.Pool(processes=worker_count) as pool:
                    for batch in pool.map(_self_play_batch, tasks):
                        buffer.extend(batch)
            else:
                if str(args.device).startswith("cuda"):
                    torch.set_num_threads(1)
                for _ in range(args.games_per_iter):
                    buffer.extend(
                        self_play(
                            model,
                            sims=args.sims,
                            device=device,
                            add_noise=True,
                            prune_radius=args.prune_radius,
                            temp_steps=args.temp_steps,
                            temp=args.temp,
                        )
                    )
            collect_end = time.monotonic()

            replay.add(buffer)
            train_samples = (
                replay.sample(args.replay_sample) if (args.replay_sample > 0 and len(replay) > 0) else list(replay.data)
            )
            if len(train_samples) > args.replay_sample and args.replay_sample > 0:
                 train_samples = random.sample(train_samples, args.replay_sample)
            
            avg_loss = 0.0
            train_start = time.monotonic()
            
            if len(train_samples) > args.batch_size:
                for _ in range(args.epochs):
                    full_augment = True if args.full_augment else (False if args.random_augment else True)
                    avg_loss = train_epoch(
                        model,
                        train_samples,
                        args.batch_size,
                        device,
                        optimizer,
                        args.value_weight,
                        full_augment=full_augment,
                    )
            train_end = time.monotonic()

            if args.async_selfplay and async_shared is not None:
                async_shared["state"] = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
                async_shared["version"] = int(async_shared.get("version", 0)) + 1

            saved_path = None
            eval_winrate = None
            eval_wins = 0
            eval_draws = 0
            eval_losses = 0
            eval_wdl_str = "n/a"
            
            if args.save_every > 0 and it % args.save_every == 0:
                save_path = f"{save_prefix}_iter{it:04d}.pt"
                torch.save(model.state_dict(), save_path)
                saved_path = save_path
                
            torch.save(model.state_dict(), latest_path)
            torch.save(model.state_dict(), args.model_path)
            
            if args.save_best and args.eval_games > 0 and it % args.eval_every == 0:
                if best_model is None:
                    torch.save(model.state_dict(), args.best_path)
                    best_model = AZNet().to(device)
                    best_model.load_state_dict(model.state_dict())
                    saved_path = args.best_path
                    eval_winrate = 1.0
                    eval_wins = args.eval_games
                    eval_wdl_str = f"{eval_wins}/0/0"
                else:
                    eval_winrate, eval_wins, eval_draws, eval_losses = eval_models(
                        model,
                        best_model,
                        args.eval_games,
                        eval_sims,
                        device,
                        args.prune_radius,
                        args.eval_temp,
                        args.eval_noise,
                        args.eval_temp_steps,
                        eval_devices,
                        args.eval_workers,
                    )
                    eval_wdl_str = f"{eval_wins}/{eval_draws}/{eval_losses}"
                    if eval_winrate_ema is None:
                        eval_winrate_ema = eval_winrate
                    else:
                        eval_winrate_ema = 0.8 * eval_winrate_ema + 0.2 * eval_winrate
                    
                    expected = 0.5
                    current_elo = update_elo(
                        current_elo,
                        score=eval_winrate,
                        expected=expected,
                        k=args.elo_k,
                    )
                    scheduler.step(current_elo)
                    if eval_winrate >= args.eval_threshold:
                        torch.save(model.state_dict(), args.best_path)
                        best_model.load_state_dict(model.state_dict())
                        saved_path = args.best_path

            total_samples = len(replay)
            train_count = len(train_samples)
            avg_game_time = (collect_end - collect_start) / max(1, args.games_per_iter)
            train_time = train_end - train_start
            stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
            log_line = (
                f"time={stamp} iter={it} games={args.games_per_iter} samples={len(buffer)} "
                f"replay_total={total_samples} train_samples={train_count} "
                f"loss={avg_loss:.4f} train_s={train_time:.1f} "
                f"avg_game_s={avg_game_time:.2f} saved={saved_path or '-'}"
            )
            if eval_winrate is not None:
                log_line += f" eval_winrate={eval_winrate:.2f}"
                if eval_winrate_ema is not None:
                    log_line += f" eval_ema={eval_winrate_ema:.2f}"
                log_line += f" eval_wdl={eval_wdl_str}"
                log_line += f" elo={current_elo:.1f}"
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
            
            if tqdm:
                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "replay": total_samples,
                    "train_s": f"{train_time:.1f}",
                    "game_s": f"{avg_game_time:.2f}",
                }
                if eval_winrate is not None:
                    postfix["win"] = f"{eval_winrate:.2f}"
                    postfix["elo"] = f"{current_elo:.0f}"
                iter_range.set_postfix(postfix)
            else:
                print(log_line)

    except KeyboardInterrupt:
        pass
    finally:
        if args.async_selfplay and async_stop is not None:
            async_stop.set()
            for p in async_workers:
                p.terminate()


if __name__ == "__main__":
    main()
