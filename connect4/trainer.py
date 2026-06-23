"""Bucle principal de entrenamiento AlphaZero."""

import copy
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

from .config import Config
from .evaluation import evaluate_models, load_checkpoint, save_checkpoint
from .logging_utils import setup_logging
from .network import Connect4Net
from .self_play import self_play
from .training import ReplayBuffer, train_network


def train_alphazero(config: Config) -> None:
    """
    Ciclo infinito de entrenamiento AlphaZero:
        auto-juego → entrenamiento → evaluación → actualización del mejor modelo
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    logger = setup_logging(config.log_file)

    logger.info("=" * 60)
    logger.info("AlphaZero Connect 4 — Inicio de entrenamiento")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"MCTS simulaciones: {config.mcts_simulations}")
    logger.info(f"Búfer máximo: {config.buffer_max_size}")
    logger.info("=" * 60)

    network = Connect4Net(config.rows, config.cols).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_max_size)

    best_network = Connect4Net(config.rows, config.cols).to(device)
    best_network.load_state_dict(copy.deepcopy(network.state_dict()))
    best_optimizer = optim.Adam(best_network.parameters(), lr=config.learning_rate)

    iteration = 0
    total_games_played = 0

    if os.path.exists(config.best_model_path):
        try:
            iter_loaded, games_loaded = load_checkpoint(
                best_network, best_optimizer, config.best_model_path, device
            )
            logger.info(
                f"Checkpoint cargado: iteración {iter_loaded}, "
                f"{games_loaded} partidas previas"
            )
            network.load_state_dict(copy.deepcopy(best_network.state_dict()))
        except Exception as e:
            logger.warning(f"No se pudo cargar checkpoint: {e}. Entrenando desde cero.")

    try:
        while True:
            iteration += 1
            iter_start = time.time()

            # --- Fase 1: Auto-juego ---
            logger.info(f"Iteración {iteration} | Generando {config.self_play_games_per_iter} partidas...")
            t0 = time.time()
            experiences, draw_count, total_moves_sp = self_play(
                network, config, device, num_games=config.self_play_games_per_iter
            )
            buffer.add(experiences)
            total_games_played += config.self_play_games_per_iter
            self_play_time = time.time() - t0
            draw_rate = draw_count / config.self_play_games_per_iter
            avg_game_length = total_moves_sp / config.self_play_games_per_iter
            logger.info(
                f"  Auto-juego: {len(experiences)} estados | "
                f"Búfer: {len(buffer)}/{config.buffer_max_size} | "
                f"Empates: {draw_rate:.1%} | Longitud media: {avg_game_length:.1f} | "
                f"Tiempo: {self_play_time:.1f}s"
            )

            # --- Fase 2: Entrenamiento ---
            t0 = time.time()
            avg_loss, avg_v_loss, avg_p_loss, value_acc, policy_entropy = train_network(
                network, optimizer, buffer, config, device
            )
            train_time = time.time() - t0
            logger.info(
                f"  Entrenamiento: loss={avg_loss:.4f} "
                f"(value={avg_v_loss:.4f}, policy={avg_p_loss:.4f}) | "
                f"Val.Acc={value_acc:.1%} | Entropía={policy_entropy:.3f} | "
                f"Tiempo: {train_time:.1f}s"
            )

            # --- Fase 3: Evaluación periódica ---
            if total_games_played % config.eval_frequency < config.self_play_games_per_iter:
                logger.info(
                    f"  Torneo de evaluación ({config.eval_games} partidas)..."
                )
                t0 = time.time()
                win_rate, eval_draw_rate, _ = evaluate_models(network, best_network, config, device)
                eval_time = time.time() - t0
                logger.info(
                    f"  Tasa de victorias del modelo actual: {win_rate:.1%} | "
                    f"Empates eval: {eval_draw_rate:.1%} | "
                    f"Tiempo: {eval_time:.1f}s"
                )

                if win_rate > config.win_threshold:
                    logger.info(
                        f"  ¡Nuevo mejor modelo! ({win_rate:.1%} > {config.win_threshold:.0%})"
                    )
                    best_network.load_state_dict(copy.deepcopy(network.state_dict()))
                    save_checkpoint(
                        best_network,
                        optimizer,
                        config.best_model_path,
                        iteration,
                        total_games_played,
                    )
                    logger.info(f"  Checkpoint guardado en {config.best_model_path}")
                else:
                    logger.info(
                        f"  Modelo actual no supera el umbral ({config.win_threshold:.0%}). "
                        f"Continuando entrenamiento sin revertir."
                    )

            iter_time = time.time() - iter_start
            logger.info(
                f"Iteración {iteration} completada en {iter_time:.1f}s | "
                f"Total partidas: {total_games_played}"
            )
            logger.info("-" * 60)

    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido por el usuario.")
        save_checkpoint(
            network, optimizer, config.best_model_path, iteration, total_games_played
        )
        logger.info(f"Checkpoint final guardado en {config.best_model_path}")
