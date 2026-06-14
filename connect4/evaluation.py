"""Evaluación por torneos y gestión de checkpoints."""

import os
from typing import Optional, Tuple

import torch
import torch.optim as optim

from .config import Config
from .game import Connect4
from .mcts import MCTS
from .network import Connect4Net
from .self_play import select_move_from_policy


def play_game(
    network1: Connect4Net,
    network2: Connect4Net,
    config: Config,
    device: torch.device,
    network1_starts: bool = True,
    mcts_sims: Optional[int] = None,
) -> int:
    """
    Partida entre dos redes. Devuelve 1 si gana network1, 2 si gana network2, 0 empate.
    MCTS sin ruido Dirichlet y selección determinista.
    """
    if mcts_sims is None:
        mcts_sims = config.eval_mcts_simulations

    game = Connect4(config.rows, config.cols, config.win_length)
    mcts1 = MCTS(network1, config, device, add_dirichlet_noise=False)
    mcts2 = MCTS(network2, config, device, add_dirichlet_noise=False)

    while not game.is_game_over():
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        is_network1_turn = (
            (game.get_current_player() == Connect4.PLAYER1 and network1_starts)
            or (game.get_current_player() == Connect4.PLAYER2 and not network1_starts)
        )

        if is_network1_turn:
            policy = mcts1.search(game, num_simulations=mcts_sims)
        else:
            policy = mcts2.search(game, num_simulations=mcts_sims)

        col = select_move_from_policy(policy, legal_moves, deterministic=True)
        game.apply_move(col)

    winner = game.get_winner()
    if winner is None:
        return 0

    network1_player = Connect4.PLAYER1 if network1_starts else Connect4.PLAYER2
    if winner == network1_player:
        return 1
    return 2


def evaluate_models(
    current_net: Connect4Net,
    best_net: Connect4Net,
    config: Config,
    device: torch.device,
) -> float:
    """
    Torneo de eval_games partidas (mitad empezando cada red).
    Devuelve la tasa de victorias del modelo actual (excluyendo empates del denominador).
    """
    wins_current = 0
    total_decided = 0
    games_per_side = config.eval_games // 2

    for i in range(config.eval_games):
        network1_starts = i < games_per_side
        result = play_game(
            current_net, best_net, config, device,
            network1_starts=network1_starts,
        )
        if result == 1:
            wins_current += 1
            total_decided += 1
        elif result == 2:
            total_decided += 1

    if total_decided == 0:
        return 0.5
    return wins_current / total_decided


def save_checkpoint(
    network: Connect4Net,
    optimizer: optim.Optimizer,
    path: str,
    iteration: int,
    total_games: int,
) -> None:
    """Guarda el modelo, optimizador y metadatos de entrenamiento."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
            "total_games": total_games,
        },
        path,
    )


def load_checkpoint(
    network: Connect4Net,
    optimizer: optim.Optimizer,
    path: str,
    device: torch.device,
) -> Tuple[int, int]:
    """Carga un checkpoint. Devuelve (iteración, total_games)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("iteration", 0), checkpoint.get("total_games", 0)
