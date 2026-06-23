import copy
import json
import logging
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.optim as optim

from connect4 import Config
from connect4.evaluation import evaluate_models, evaluate_vs_random, load_checkpoint, save_checkpoint
from connect4.logging_utils import setup_logging
from connect4.network import Connect4Net
from connect4.self_play import self_play, self_play_parallel
from connect4.training import ReplayBuffer, train_network

# Códigos ANSI para colores en terminal
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Colores
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

    # Combinaciones
    SUCCESS = GREEN + BOLD
    WARNING = YELLOW + BOLD
    ERROR = RED + BOLD
    INFO = CYAN
    DEBUG = GRAY

class ColoredFormatter(logging.Formatter):
    """Formateador personalizado con colores para consola."""

    COLORS_MAP = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.ERROR,
    }

    def format(self, record):
        # Aplicar color según nivel
        color = self.COLORS_MAP.get(record.levelno, Colors.WHITE)

        # Colorear el nivel de log
        levelname_colored = f"{color}{record.levelname}{Colors.RESET}"

        # Crear formato personalizado
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        message = record.getMessage()

        # Colorear emojis y símbolos
        message = message.replace("✓", f"{Colors.GREEN}✓{Colors.RESET}")
        message = message.replace("✗", f"{Colors.RED}✗{Colors.RESET}")
        message = message.replace("🎉", f"{Colors.GREEN}🎉{Colors.RESET}")
        message = message.replace("⚠️", f"{Colors.YELLOW}⚠️{Colors.RESET}")
        message = message.replace("🚨", f"{Colors.ERROR}🚨{Colors.RESET}")
        message = message.replace("⏸️", f"{Colors.YELLOW}⏸️{Colors.RESET}")
        message = message.replace("ℹ️", f"{Colors.INFO}ℹ️{Colors.RESET}")
        message = message.replace("⚙️", f"{Colors.MAGENTA}⚙️{Colors.RESET}")

        return f"{timestamp} | {levelname_colored} | {message}"

def get_max_gpu_temperature():
    """Obtiene la temperatura máxima entre las GPUs usando nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            encoding="utf-8"
        )
        temps = [int(t.strip()) for t in output.strip().split('\n') if t.strip().isdigit()]
        return max(temps) if temps else -1
    except Exception:
        return -1  # Retorna -1 si no hay GPU o no está disponible nvidia-smi

def setup_cumulative_logging(log_file):
    """Configura logging que ACUMULA en lugar de sobrescribir con colores en terminal."""
    logger = logging.getLogger("AutoTrain")
    logger.setLevel(logging.DEBUG)

    # Remover handlers anteriores para evitar duplicados
    logger.handlers.clear()

    # Handler para archivo SIN colores (APPEND mode)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Handler para consola CON colores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def auto_train():
    config = Config()
    config.log_file = "auto_train.log"

    logger = setup_cumulative_logging(config.log_file)
    logger.info("=" * 70)
    logger.info(f"{Colors.SUCCESS}{'='*70}{Colors.RESET}")
    logger.info(f"{Colors.SUCCESS}Iniciando Auto-Entrenamiento Inteligente AlphaZero{Colors.RESET}")
    logger.info(f"{Colors.SUCCESS}{'='*70}{Colors.RESET}")
    device = torch.device(config.device)
    logger.info(f"Dispositivo: {device}")
    logger.info(f"MCTS simulaciones: {config.mcts_simulations} | Épocas: {config.train_epochs} | LR: {config.learning_rate}")
    logger.info(f"dirichlet_alpha: {config.dirichlet_alpha} | c_puct: {config.c_puct} | Umbral: {config.win_threshold}")
    logger.info("=" * 70)

    network = Connect4Net(config.rows, config.cols).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_max_size)

    best_network = Connect4Net(config.rows, config.cols).to(device)
    best_network.load_state_dict(copy.deepcopy(network.state_dict()))
    best_optimizer = optim.Adam(best_network.parameters(), lr=config.learning_rate)

    iteration = 0
    total_games_played = 0
    stagnation_counter = 0
    stagnation_cycle = 0
    random_eval_counter = 0
    base_mcts = config.mcts_simulations
    base_self_play_games = config.self_play_games_per_iter
    initial_lr = config.learning_rate
    metrics_file = "training_metrics.json"

    # Crear directorio de checkpoints si no existe
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Cargar checkpoint si existe
    if os.path.exists(config.best_model_path):
        try:
            iter_loaded, games_loaded = load_checkpoint(
                best_network, best_optimizer, config.best_model_path, device
            )
            iteration = iter_loaded
            total_games_played = games_loaded
            network.load_state_dict(copy.deepcopy(best_network.state_dict()))
            optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
            logger.info(f"{Colors.SUCCESS}✓ Checkpoint cargado exitosamente:{Colors.RESET}")
            logger.info(f"  - Iteración: {iteration}")
            logger.info(f"  - Partidas jugadas: {total_games_played}")
            logger.info(f"  - Ruta: {config.best_model_path}")
        except Exception as e:
            logger.warning(f"{Colors.WARNING}✗ No se pudo cargar checkpoint: {e}. Iniciando desde cero.{Colors.RESET}")

    # Cargar métricas anteriores si existen (para acumular)
    metrics = []
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                logger.info(f"{Colors.SUCCESS}✓ Métricas previas cargadas: {len(metrics)} registros{Colors.RESET}")
        except json.JSONDecodeError:
            logger.warning(f"{Colors.WARNING}✗ Archivo de métricas corrupto. Iniciando nuevo registro.{Colors.RESET}")
            metrics = []

    logger.info("=" * 70)

    try:
        while True:
            # 1. Comprobación de seguridad de GPU
            iter_start = time.time()
            gpu_temp = get_max_gpu_temperature()
            if gpu_temp >= 85:
                logger.error(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
                logger.error(f"{Colors.ERROR}¡ALERTA CRÍTICA! Temperatura de GPU en {gpu_temp}°C (Límite: 85°C).{Colors.RESET}")
                logger.error(f"{Colors.ERROR}Deteniendo entrenamiento inmediatamente para proteger hardware.{Colors.RESET}")
                logger.error(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
                save_checkpoint(
                    best_network, best_optimizer, config.best_model_path, iteration, total_games_played
                )
                logger.info(f"{Colors.SUCCESS}✓ Checkpoint de emergencia guardado en {config.best_model_path}{Colors.RESET}")
                break
            elif gpu_temp != -1:
                temp_color = Colors.GREEN if gpu_temp < 75 else Colors.YELLOW if gpu_temp < 80 else Colors.RED
                logger.debug(f"Temperatura GPU: {temp_color}{gpu_temp}°C{Colors.RESET} ✓")

            iteration += 1
            logger.info(f"{Colors.CYAN}--- Iteración {iteration} | MCTS: {config.mcts_simulations} | LR: {config.learning_rate:.5f} | Umbral: {config.win_threshold:.2f} ---{Colors.RESET}")

            # Auto-juego
            experiences, draw_count, total_moves_sp = self_play_parallel(
                network, config, device,
                num_games=config.self_play_games_per_iter,
                num_workers=config.num_self_play_workers,
            )
            buffer.add(experiences)
            total_games_played += config.self_play_games_per_iter
            draw_rate = draw_count / config.self_play_games_per_iter
            avg_game_length = total_moves_sp / config.self_play_games_per_iter
            logger.info(
                f"{Colors.GREEN}  ✓ Auto-juego:{Colors.RESET} {len(experiences)} estados | "
                f"Búfer: {len(buffer)}/{config.buffer_max_size} | "
                f"Empates: {draw_rate:.1%} | Longitud media: {avg_game_length:.1f}"
            )

            # Entrenamiento
            avg_loss, avg_v_loss, avg_p_loss, value_acc, policy_entropy = train_network(
                network, optimizer, buffer, config, device
            )
            logger.info(
                f"{Colors.GREEN}  ✓ Entrenamiento:{Colors.RESET} "
                f"loss={avg_loss:.4f} (value={avg_v_loss:.4f}, policy={avg_p_loss:.4f}) | "
                f"Val.Acc={value_acc:.1%} | Entropía={policy_entropy:.3f}"
            )

            # Evaluación periódica
            if total_games_played % config.eval_frequency < config.self_play_games_per_iter:
                logger.info(f"{Colors.MAGENTA}  ⚙️  Torneo de evaluación ({config.eval_games} partidas)...{Colors.RESET}")
                win_rate, eval_draw_rate, _ = evaluate_models(network, best_network, config, device)

                # Evaluación vs jugador aleatorio (cada 5 evaluaciones)
                random_eval_counter += 1
                win_rate_vs_random = None
                if random_eval_counter % 3 == 0:
                    wr_random, _, _ = evaluate_vs_random(network, config, device, num_games=20)
                    win_rate_vs_random = round(wr_random, 4)
                    logger.info(f"{Colors.CYAN}  ✓ Win rate vs aleatorio: {wr_random:.1%}{Colors.RESET}")

                wr_color = Colors.GREEN if win_rate > config.win_threshold else Colors.YELLOW
                logger.info(
                    f"{Colors.GREEN}  ✓ Tasa de victorias vs mejor:{Colors.RESET} {wr_color}{win_rate:.1%}{Colors.RESET} | "
                    f"Empates eval: {eval_draw_rate:.1%}"
                )

                # Registrar métricas (ACUMULAR en JSON)
                iter_metrics = {
                    "iteration": iteration,
                    "total_games": total_games_played,
                    "loss": round(avg_loss, 4),
                    "policy_loss": round(avg_p_loss, 4),
                    "value_loss": round(avg_v_loss, 4),
                    "value_accuracy": round(value_acc, 4),
                    "policy_entropy": round(policy_entropy, 4),
                    "win_rate": round(win_rate, 4),
                    "win_rate_vs_random": win_rate_vs_random,
                    "draw_rate": round(draw_rate, 4),
                    "avg_game_length": round(avg_game_length, 2),
                    "mcts_simulations": config.mcts_simulations,
                    "learning_rate": config.learning_rate,
                    "win_threshold": config.win_threshold,
                    "gpu_temp": gpu_temp,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                metrics.append(iter_metrics)

                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.debug(f"Métricas actualizadas ({len(metrics)} registros total)")

                # Lógica de actualización del mejor modelo
                if win_rate > config.win_threshold:
                    logger.info(f"{Colors.SUCCESS}  🎉 ¡Nuevo mejor modelo encontrado! ({win_rate:.1%} > {config.win_threshold:.2%}){Colors.RESET}")
                    best_network.load_state_dict(copy.deepcopy(network.state_dict()))
                    best_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                    save_checkpoint(best_network, best_optimizer, config.best_model_path, iteration, total_games_played)
                    logger.info(f"{Colors.SUCCESS}  ✓ Checkpoint guardado: {config.best_model_path}{Colors.RESET}")
                    stagnation_counter = 0
                    stagnation_cycle = 0
                    config.mcts_simulations = base_mcts
                    config.self_play_games_per_iter = base_self_play_games
                    logger.info(f"  ↩ Parámetros restaurados: MCTS={base_mcts}, juegos/iter={base_self_play_games}")
                else:
                    # No revertir el modelo — continuar entrenando desde donde está
                    logger.info(f"{Colors.INFO}  ℹ️  Modelo no superó umbral ({config.win_threshold:.2%}). Continuando entrenamiento.{Colors.RESET}")
                    stagnation_counter += 1
                    logger.info(f"{Colors.YELLOW}  ⚠️  Contador de estancamiento: {stagnation_counter}/5{Colors.RESET}")

                    # ESTRATEGIAS DE DESESTANCAMIENTO (después de 5 fallos consecutivos)
                    if stagnation_counter >= 5:
                        stagnation_counter = 0
                        stagnation_cycle = (stagnation_cycle % 4) + 1

                        logger.warning(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
                        logger.warning(f"{Colors.ERROR}🚨 ESTANCAMIENTO — Estrategia {stagnation_cycle}/4{Colors.RESET}")
                        logger.warning(f"{Colors.ERROR}{'='*70}{Colors.RESET}")

                        if stagnation_cycle == 1:
                            # MCTS boost (con cota) + reset optimizer
                            new_mcts = min(int(config.mcts_simulations * 1.25), config.max_mcts_simulations)
                            config.mcts_simulations = new_mcts
                            optimizer.state.clear()
                            logger.warning(f"{Colors.YELLOW}  [1/4] MCTS → {new_mcts} | Optimizer reseteado{Colors.RESET}")

                        elif stagnation_cycle == 2:
                            # Perturbación de pesos + reset optimizer
                            sigma = 1e-3
                            with torch.no_grad():
                                for p in network.parameters():
                                    p.add_(torch.randn_like(p) * sigma)
                            optimizer.state.clear()
                            logger.warning(f"{Colors.YELLOW}  [2/4] Perturbación de pesos (σ={sigma}) | Optimizer reseteado{Colors.RESET}")

                        elif stagnation_cycle == 3:
                            # Más partidas de auto-juego + reset optimizer
                            new_games = min(config.self_play_games_per_iter + 50, 200)
                            config.self_play_games_per_iter = new_games
                            optimizer.state.clear()
                            logger.warning(f"{Colors.YELLOW}  [3/4] Auto-juego → {new_games} partidas | Optimizer reseteado{Colors.RESET}")

                        else:
                            # Reset completo a valores base
                            config.mcts_simulations = base_mcts
                            config.self_play_games_per_iter = base_self_play_games
                            for pg in optimizer.param_groups:
                                pg['lr'] = initial_lr
                            optimizer.state.clear()
                            stagnation_cycle = 0
                            logger.warning(
                                f"{Colors.YELLOW}  [4/4] Reset completo: "
                                f"MCTS→{base_mcts}, juegos→{base_self_play_games}, LR→{initial_lr} | Optimizer reseteado{Colors.RESET}"
                            )

                        logger.warning(f"{Colors.ERROR}{'='*70}{Colors.RESET}")

            elapsed = time.time() - iter_start
            logger.info(f"  ⏱ Tiempo iteración: {elapsed/60:.1f} min")
            logger.info(f"{Colors.GRAY}-{'-'*68}{Colors.RESET}")

    except KeyboardInterrupt:
        logger.info(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
        logger.info(f"{Colors.YELLOW}⏸️  Entrenamiento detenido manualmente por el usuario{Colors.RESET}")
        logger.info(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
    finally:
        logger.info(f"{Colors.YELLOW}Guardando checkpoint final...{Colors.RESET}")
        save_checkpoint(best_network, best_optimizer, config.best_model_path, iteration, total_games_played)
        logger.info(f"{Colors.SUCCESS}✓ Checkpoint final guardado: {config.best_model_path}{Colors.RESET}")

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"{Colors.SUCCESS}✓ Métricas finales guardadas: {len(metrics)} registros en {metrics_file}{Colors.RESET}")

        logger.info(f"{Colors.SUCCESS}{'='*70}{Colors.RESET}")
        logger.info(f"{Colors.SUCCESS}✓ Auto-entrenamiento finalizado correctamente{Colors.RESET}")
        logger.info(f"{Colors.SUCCESS}{'='*70}{Colors.RESET}")

if __name__ == "__main__":
    auto_train()
