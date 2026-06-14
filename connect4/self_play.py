"""Generación de partidas de auto-juego para entrenamiento."""

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
) -> List[Experience]:
    """
    Genera partidas completas de auto-juego.
    Por cada movimiento guarda (estado, política_mcts, resultado_final).
    El resultado se asigna al final desde la perspectiva del jugador que movió.
    """
    all_experiences: List[Experience] = []
    mcts = MCTS(network, config, device, add_dirichlet_noise=True)

    for _ in range(num_games):
        game = Connect4(config.rows, config.cols, config.win_length)
        game_history: List[Tuple[np.ndarray, np.ndarray, int]] = []

        while not game.is_game_over():
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            state = game.get_state()
            policy = mcts.search(game)

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
        for state, policy, player in game_history:
            if winner is None:
                outcome = 0.0
            elif winner == player:
                outcome = 1.0
            else:
                outcome = -1.0
            all_experiences.append((state, policy, outcome))

    return all_experiences
