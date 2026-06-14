"""
Compatibilidad hacia atrás: redirige al nuevo punto de entrada.

Uso:
    python alphazero_connect4.py
    python main.py
"""

from connect4 import Config, train_alphazero

if __name__ == "__main__":
    config = Config(
        mcts_simulations=50,
        self_play_games_per_iter=100,
        buffer_max_size=100_000,
        batch_size=256,
        train_epochs=10,
        eval_frequency=200,
        eval_mcts_simulations=100,
        temperature_moves=15,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    print("=" * 60)
    print("  AlphaZero para Conecta 4")
    print("  Presiona Ctrl+C para detener el entrenamiento")
    print("=" * 60)

    train_alphazero(config)
