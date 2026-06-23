"""Parámetros configurables del entrenamiento AlphaZero."""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Config:
    """Parámetros configurables del entrenamiento AlphaZero."""

    # Tablero
    rows: int = 6
    cols: int = 7
    win_length: int = 4

    # MCTS
    mcts_simulations: int = 100
    max_mcts_simulations: int = 200   # cota máxima para el ajuste dinámico
    c_puct: float = 1.5
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.5      # ~10/num_legal_moves para Connect4 (7 cols)

    # Auto-juego
    temperature_moves: int = 20
    temperature: float = 1.0
    self_play_games_per_iter: int = 100
    num_self_play_workers: int = 4       # 0 ó 1 = secuencial (fallback)

    # Búfer y entrenamiento
    buffer_max_size: int = 100_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_epochs: int = 3             # 10 épocas causaba sobreajuste severo

    # Evaluación
    eval_frequency: int = 200          # cada N partidas de auto-juego acumuladas
    eval_games: int = 40               # 20 como P1 + 20 como P2
    eval_mcts_simulations: int = 100
    win_threshold: float = 0.52

    # General
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pt"
    log_file: Optional[str] = "training.log"
    seed: int = 42
