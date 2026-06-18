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
from connect4.evaluation import evaluate_models, load_checkpoint, save_checkpoint
from connect4.logging_utils import setup_logging
from connect4.network import Connect4Net
from connect4.self_play import self_play
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
    # Parámetros iniciales recomendados
    config.log_file = "auto_train.log"
    
    logger = setup_cumulative_logging(config.log_file)
    logger.info("=" * 70)
    logger.info(f"{Colors.SUCCESS}{'='*70}{Colors.RESET}")
    logger.info(f"{Colors.SUCCESS}Iniciando Auto-Entrenamiento Inteligente AlphaZero{Colors.RESET}")
    logger.info(f"{Colors.SUCCESS}{'='*70}{Colors.RESET}")
    device = torch.device(config.device)
    logger.info(f"Dispositivo: {device}")
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
    metrics_file = "training_metrics.json"

    # Crear directorio de checkpoints si no existe
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Cargar checkpoint si existe
    checkpoint_loaded = False
    if os.path.exists(config.best_model_path):
        try:
            iter_loaded, games_loaded = load_checkpoint(
                best_network, best_optimizer, config.best_model_path, device
            )
            iteration = iter_loaded
            total_games_played = games_loaded
            network.load_state_dict(copy.deepcopy(best_network.state_dict()))
            optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
            checkpoint_loaded = True
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
            experiences = self_play(network, config, device, num_games=config.self_play_games_per_iter)
            buffer.add(experiences)
            total_games_played += config.self_play_games_per_iter
            logger.info(f"{Colors.GREEN}  ✓ Auto-juego:{Colors.RESET} {len(experiences)} estados | Búfer: {len(buffer)}/{config.buffer_max_size}")

            # Entrenamiento
            avg_loss, avg_v_loss, avg_p_loss = train_network(network, optimizer, buffer, config, device)
            logger.info(f"{Colors.GREEN}  ✓ Entrenamiento:{Colors.RESET} loss={avg_loss:.4f} (value={avg_v_loss:.4f}, policy={avg_p_loss:.4f})")

            # Evaluación periódica
            if total_games_played % config.eval_frequency < config.self_play_games_per_iter:
                logger.info(f"{Colors.MAGENTA}  ⚙️  Torneo de evaluación ({config.eval_games} partidas)...{Colors.RESET}")
                win_rate = evaluate_models(network, best_network, config, device)
                
                # Color del win rate basado en el umbral
                wr_color = Colors.GREEN if win_rate > config.win_threshold else Colors.YELLOW
                logger.info(f"{Colors.GREEN}  ✓ Tasa de victorias:{Colors.RESET} {wr_color}{win_rate:.1%}{Colors.RESET}")

                # Registrar métricas (ACUMULAR en JSON)
                iter_metrics = {
                    "iteration": iteration,
                    "total_games": total_games_played,
                    "loss": round(avg_loss, 4),
                    "win_rate": round(win_rate, 4),
                    "mcts_simulations": config.mcts_simulations,
                    "learning_rate": config.learning_rate,
                    "win_threshold": config.win_threshold,
                    "gpu_temp": gpu_temp,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                metrics.append(iter_metrics)
                
                # Guardar métricas acumuladas en JSON
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.debug(f"Métricas actualizadas ({len(metrics)} registros total)")

                # Lógica de actualización y estancamiento
                if win_rate > config.win_threshold:
                    logger.info(f"{Colors.SUCCESS}  🎉 ¡Nuevo mejor modelo encontrado! ({win_rate:.1%} > {config.win_threshold:.2%}){Colors.RESET}")
                    best_network.load_state_dict(copy.deepcopy(network.state_dict()))
                    best_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                    save_checkpoint(best_network, best_optimizer, config.best_model_path, iteration, total_games_played)
                    logger.info(f"{Colors.SUCCESS}  ✓ Checkpoint guardado: {config.best_model_path}{Colors.RESET}")
                    stagnation_counter = 0  # Reiniciar contador de estancamiento
                else:
                    logger.info(f"{Colors.INFO}  ℹ️  Modelo no superó umbral ({config.win_threshold:.2%}). Manteniendo mejor modelo anterior.{Colors.RESET}")
                    network.load_state_dict(copy.deepcopy(best_network.state_dict()))
                    stagnation_counter += 1
                    logger.info(f"{Colors.YELLOW}  ⚠️  Contador de estancamiento: {stagnation_counter}/3{Colors.RESET}")

                    # ESTRATEGIAS DE DESESTANCAMIENTO
                    if stagnation_counter >= 3:
                        logger.warning(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
                        logger.warning(f"{Colors.ERROR}🚨 ESTANCAMIENTO DETECTADO - Aplicando ajustes dinámicos{Colors.RESET}")
                        logger.warning(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
                        
                        # Guardar configuración anterior
                        old_lr = config.learning_rate
                        old_mcts = config.mcts_simulations
                        old_threshold = config.win_threshold
                        
                        # 1. Reducir Learning Rate para afinar pesos (decay)
                        config.learning_rate = max(1e-5, config.learning_rate * 0.75)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = config.learning_rate
                        
                        # 2. Aumentar poder de búsqueda
                        config.mcts_simulations = int(config.mcts_simulations * 1.25)
                        
                        # 3. Relajar ligeramente el umbral (hasta un mínimo de 52%)
                        if config.win_threshold > 0.52:
                            config.win_threshold = max(0.52, config.win_threshold - 0.01)

                        logger.warning(f"{Colors.YELLOW}Cambios aplicados:{Colors.RESET}")
                        logger.warning(f"{Colors.YELLOW}  LR: {old_lr:.5f} → {Colors.GREEN}{config.learning_rate:.5f}{Colors.RESET}")
                        logger.warning(f"{Colors.YELLOW}  MCTS: {old_mcts} → {Colors.GREEN}{config.mcts_simulations}{Colors.RESET}")
                        logger.warning(f"{Colors.YELLOW}  Umbral: {old_threshold:.2f} → {Colors.GREEN}{config.win_threshold:.2f}{Colors.RESET}")
                        logger.warning(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
                        
                        # Resetear contador para darle tiempo con los nuevos valores
                        stagnation_counter = 0

                logger.info(f"{Colors.GRAY}-{'-'*68}{Colors.RESET}")

    except KeyboardInterrupt:
        logger.info(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
        logger.info(f"{Colors.YELLOW}⏸️  Entrenamiento detenido manualmente por el usuario{Colors.RESET}")
        logger.info(f"{Colors.ERROR}{'='*70}{Colors.RESET}")
    finally:
        # Guardar estado actual de forma segura - SIEMPRE
        logger.info(f"{Colors.YELLOW}Guardando checkpoint final...{Colors.RESET}")
        save_checkpoint(best_network, best_optimizer, config.best_model_path, iteration, total_games_played)
        logger.info(f"{Colors.SUCCESS}✓ Checkpoint final guardado: {config.best_model_path}{Colors.RESET}")
        
        # Guardar métricas finales
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"{Colors.SUCCESS}✓ Métricas finales guardadas: {len(metrics)} registros en {metrics_file}{Colors.RESET}")
        
        logger.info(f"{Colors.SUCCESS}{'='*70}{Colors.RESET}")
        logger.info(f"{Colors.SUCCESS}✓ Auto-entrenamiento finalizado correctamente{Colors.RESET}")
        logger.info(f"{Colors.SUCCESS}{'='*70}{Colors.RESET}")

if __name__ == "__main__":
    auto_train()
