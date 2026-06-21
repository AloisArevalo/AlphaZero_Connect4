#!/usr/bin/env python3
"""
Modo terminal simple para jugar Connect 4 contra la IA.
Útil para debugging y juego sin GUI.

Uso:
    python game_ui/terminal_play.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_ui.game_engine import GameEngine
from game_ui.utils import state_to_visual, get_column_input, count_stats


# Colores ANSI
class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'


def main():
    print(f"{Color.CYAN}{'='*60}")
    print(f"  Connect 4 vs IA AlphaZero - Modo Terminal")
    print(f"{'='*60}{Color.RESET}")
    print()
    
    # Cargar engine
    try:
        engine = GameEngine(mcts_simulations=50)
    except FileNotFoundError as e:
        print(f"{Color.RED}Error: {e}{Color.RESET}")
        print("\nPasos para resolver:")
        print("1. Ejecuta en la carpeta raiz: python main.py o python auto_train.py")
        print("2. Espera a que se genere el archivo: checkpoints/best_model.pt")
        print("3. Luego ejecuta este script nuevamente")
        sys.exit(1)
    
    # Iniciar juego
    state = engine.reset_game()
    current_player = 1
    human_player = 1
    ai_player = 2
    game_over = False
    
    print("Leyenda: . = Vacio, R = Tu, B = IA")
    print()
    
    while not game_over:
        # Mostrar tablero
        board = engine.get_board_visual()
        print(state_to_visual(board))
        print()
        
        stats = count_stats(board)
        print(f"Fichas: Tu={stats['human']} | IA={stats['ai']} | Vacios={stats['empty']}")
        print()
        
        if current_player == human_player:
            # Turno del humano
            print(f"{Color.GREEN}TU TURNO{Color.RESET}")
            legal_moves = engine.get_legal_moves()
            print(f"Columnas disponibles: {legal_moves}")
            
            col = get_column_input(engine.game.board[0] == 0)
            
            new_state, valid, is_terminal, outcome = engine.step(col)
            if not valid:
                print(f"{Color.RED}Movimiento invalido!{Color.RESET}")
                continue
            
            state = new_state
            
            if is_terminal:
                board = engine.get_board_visual()
                print(state_to_visual(board))
                if outcome == 1:
                    print(f"\n{Color.GREEN}GANASTE!{Color.RESET}")
                else:
                    print(f"\n{Color.RED}LA IA GANO{Color.RESET}")
                game_over = True
            else:
                current_player = ai_player
        
        else:
            # Turno de la IA
            print(f"{Color.CYAN}TURNO DE LA IA{Color.RESET}")
            print("La IA esta pensando...")
            
            ai_move = engine.get_ai_move()
            print(f"La IA elige columna: {ai_move}")
            
            new_state, valid, is_terminal, outcome = engine.step(ai_move)
            if not valid:
                print(f"{Color.RED}Error: La IA intento un movimiento invalido!{Color.RESET}")
                break
            
            state = new_state
            
            if is_terminal:
                board = engine.get_board_visual()
                print(state_to_visual(board))
                if outcome == -1:
                    print(f"\n{Color.RED}LA IA GANO{Color.RESET}")
                else:
                    print(f"\n{Color.YELLOW}EMPATE{Color.RESET}")
                game_over = True
            else:
                current_player = human_player
        
        print()
    
    print(f"{Color.CYAN}{'='*60}")
    print("  Gracias por jugar!")
    print(f"{'='*60}{Color.RESET}")


if __name__ == "__main__":
    main()
