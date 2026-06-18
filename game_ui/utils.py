"""Utilidades para la interfaz de juego."""

import numpy as np


def state_to_visual(state):
    """
    Convierte un estado del juego a una representación visual.
    
    Args:
        state: Array de forma (6, 7) con valores 0, 1, 2
        
    Returns:
        str: Representación ASCII del tablero
    """
    symbols = {
        0: ".",
        1: "R",  # Red (jugador)
        2: "B",  # Blue (IA)
    }
    
    output = []
    output.append("  0 1 2 3 4 5 6")
    output.append("+-----------+")
    
    for row in range(len(state)):
        line = "|"
        for col in range(len(state[row])):
            line += f" {symbols[int(state[row, col])]}"
        line += " |"
        output.append(line)
    
    output.append("+-----------+")
    
    return "\n".join(output)


def get_column_input(valid_moves=None):
    """
    Obtiene entrada del usuario de forma interactiva en terminal.
    
    Args:
        valid_moves: Array bool de movimientos válidos
        
    Returns:
        int: Columna seleccionada (0-6)
    """
    while True:
        try:
            col = int(input("Selecciona columna (0-6): "))
            if not (0 <= col < 7):
                print("Columna fuera de rango. Intenta de nuevo.")
                continue
            
            if valid_moves is not None and not valid_moves[col]:
                print("Esa columna esta llena! Intenta otra.")
                continue
            
            return col
        except ValueError:
            print("Entrada invalida. Introduce un numero del 0 al 6.")


def count_stats(state):
    """
    Calcula estadísticas del tablero.
    
    Args:
        state: Array de forma (6, 7)
        
    Returns:
        dict: Estadísticas del juego
    """
    return {
        "human": int((state == 1).sum()),
        "ai": int((state == 2).sum()),
        "empty": int((state == 0).sum()),
        "total_moves": int((state != 0).sum()),
    }
