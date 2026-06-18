#!/usr/bin/env python3
"""
Script principal para ejecutar el juego contra la IA.

Uso:
    python play.py          # Dificultad normal (100 simulaciones)
    python play.py --easy   # Facil (50 simulaciones)
    python play.py --hard   # Dificil (200 simulaciones)
"""

import sys
import argparse
import tkinter as tk
from game_ui import Connect4GUI


# Colores ANSI
class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RESET = '\033[0m'


def main():
    parser = argparse.ArgumentParser(
        description="Juega Connect 4 contra la IA AlphaZero entrenada"
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "normal", "hard", "expert"],
        default="normal",
        help="Dificultad del juego (default: normal)"
    )
    
    args = parser.parse_args()
    
    # Mapear dificultad a simulaciones MCTS
    difficulty_map = {
        "easy": 25,
        "normal": 100,
        "hard": 200,
        "expert": 400,
    }
    
    mcts_sims = difficulty_map[args.difficulty]
    
    print(f"{Color.CYAN}{'='*60}")
    print(f"  Connect 4 vs IA AlphaZero")
    print(f"{'='*60}")
    print(f"Dificultad: {args.difficulty.upper()} ({mcts_sims} simulaciones MCTS)")
    print(f"{'='*60}{Color.RESET}")
    
    # Iniciar GUI con la dificultad correcta
    root = tk.Tk()
    try:
        app = Connect4GUI(root, ai_difficulty=mcts_sims)
        root.mainloop()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPasos para resolver:")
        print("1. Ejecuta en la carpeta raiz: python main.py o python auto_train.py")
        print("2. Espera a que se genere el archivo: checkpoints/best_model.pt")
        print("3. Luego ejecuta este script nuevamente")
        root.destroy()
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
        root.destroy()


if __name__ == "__main__":
    main()
