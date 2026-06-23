"""Generación de partidas de auto-juego para entrenamiento."""

import dataclasses
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
import torch

from .config import Config
from .game import Connect4
from .mcts import MCTS
from .network import Connect4Net

# (estado, política_mcts, resultado)
Experience = Tuple[np.ndarray, np.ndarray, float]


def select_move_from_policy(
    policy: np.ndarray,
    legal_moves: List[int],
    temperature: float = 1.0,
    deterministic: bool = False,
) -> int:
    """
    Selecciona una columna según la política MCTS.
    Con temperatura > 0: muestreo proporcional a visitas^temperature.
    Determinista: columna con mayor probabilidad.
    """
    if deterministic or temperature == 0:
        return max(legal_moves, key=lambda c: policy[c])

    probs = np.array([policy[c] for c in legal_moves], dtype=np.float64)
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        probs = np.ones(len(legal_moves)) / len(legal_moves)
    else:
        probs /= probs_sum

    return np.random.choice(legal_moves, p=probs)


def self_play(
    network: Connect4Net,
    config: Config,
    device: torch.device,
    num_games: int = 1,
) -> Tuple[List[Experience], int, int]:
    """
    Genera partidas completas de auto-juego.
    Por cada movimiento guarda (estado, política_mcts, resultado_final).
    El resultado se asigna al final desde la perspectiva del jugador que movió.

    Retorna: (experiences, draw_count, total_moves)
    """
    all_experiences: List[Experience] = []
    mcts = MCTS(network, config, device, add_dirichlet_noise=True)

    draw_count = 0
    total_moves = 0

    for _ in range(num_games):
        game = Connect4(config.rows, config.cols, config.win_length)
        game_history: List[Tuple[np.ndarray, np.ndarray, int]] = []

        while not game.is_game_over():
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            state = game.get_state()
            policy, _ = mcts.search(game)

            # Temperatura en los primeros movimientos para exploración
            if game.move_count < config.temperature_moves:
                col = select_move_from_policy(
                    policy, legal_moves, temperature=config.temperature, deterministic=False
                )
            else:
                col = select_move_from_policy(
                    policy, legal_moves, temperature=0, deterministic=True
                )

            player_who_moved = game.get_current_player()
            game_history.append((state, policy.copy(), player_who_moved))
            game.apply_move(col)

        # Asignar resultados finales a cada posición guardada
        winner = game.get_winner()
        if winner is None:
            draw_count += 1

        total_moves += len(game_history)

        for state, policy, player in game_history:
            if winner is None:
                outcome = 0.0
            elif winner == player:
                outcome = 1.0
            else:
                outcome = -1.0
            all_experiences.append((state, policy, outcome))

    return all_experiences, draw_count, total_moves


def _play_games_worker(state_dict, config_kwargs, num_games):
    """Worker de multiprocessing: crea red en CPU y juega partidas."""
    import torch as _torch
    from connect4.config import Config
    from connect4.network import Connect4Net

    _torch.set_num_threads(1)  # evitar oversubscription: cada worker usa 1 thread
    config = Config(**config_kwargs)
    device = _torch.device('cpu')
    network = Connect4Net(config.rows, config.cols).to(device)
    network.load_state_dict(state_dict)
    network.eval()
    return self_play(network, config, device, num_games)


def self_play_parallel(
    network: Connect4Net,
    config: Config,
    device: torch.device,
    num_games: int = 1,
    num_workers: int = 4,
) -> Tuple[List[Experience], int, int]:
    """Auto-juego paralelo con workers en CPU. Fallback a secuencial si num_workers <= 1."""
    if num_workers <= 1:
        return self_play(network, config, device, num_games)

    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    config_kwargs = dataclasses.asdict(config)

    n = num_workers
    games_split = [num_games // n + (1 if i < num_games % n else 0) for i in range(n)]
    args = [(state_dict, config_kwargs, g) for g in games_split]

    ctx = mp.get_context('spawn')
    with ctx.Pool(num_workers) as pool:
        results = pool.starmap(_play_games_worker, args)

    all_exp: List[Experience] = []
    total_draws, total_moves = 0, 0
    for exp, draws, moves in results:
        all_exp.extend(exp)
        total_draws += draws
        total_moves += moves
    return all_exp, total_draws, total_moves
