"""Motor de juego que envuelve el modelo entrenado para jugar interactivamente."""

import os
import sys
import torch
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connect4 import Config, Connect4, Connect4Net
from connect4.mcts import MCTS
from connect4.evaluation import load_checkpoint


# Colores ANSI
class Color:
    GREEN = '\033[92m'
    RESET = '\033[0m'


class GameEngine:
    """Engine que carga el modelo entrenado y juega Connect 4."""
    
    def __init__(self, model_path="checkpoints/best_model.pt", mcts_simulations=100):
        """
        Inicializa el engine con el modelo entrenado.
        
        Args:
            model_path: Ruta al checkpoint del mejor modelo
            mcts_simulations: Número de simulaciones MCTS para cada movimiento
        """
        self.config = Config(mcts_simulations=mcts_simulations)
        self.device = torch.device(self.config.device)
        
        # Crear red neuronal
        self.network = Connect4Net(self.config.rows, self.config.cols).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Cargar modelo entrenado
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontro el modelo en {model_path}. "
                "Primero entrena el modelo ejecutando: python main.py o python auto_train.py"
            )
        
        try:
            load_checkpoint(self.network, optimizer, model_path, self.device)
            self.network.eval()  # Modo evaluacion (sin dropout, etc.)
            print(f"{Color.GREEN}Modelo cargado exitosamente desde: {model_path}{Color.RESET}")
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo: {e}")
        
        # Juego y MCTS
        self.game = Connect4()
        self.mcts = MCTS(self.network, self.config, self.device)
    
    def get_ai_move(self, state):
        """
        Obtiene el siguiente movimiento de la IA usando MCTS.
        
        Args:
            state: Estado actual del juego (numpy array shape (3, 6, 7))
            
        Returns:
            int: Columna donde la IA quiere jugar (0-6)
        """
        # Ejecutar MCTS desde la perspectiva del jugador actual
        action_probs = self.mcts.search(state, num_simulations=self.config.mcts_simulations)
        
        # Seleccionar movimiento con mayor probabilidad (determinista en juego)
        best_action = int(action_probs.argmax())
        return best_action
    
    def reset_game(self):
        """Reinicia el juego y retorna el estado inicial."""
        self.game.reset()
        return self.game.get_state()
    
    def step(self, col):
        """
        Realiza un movimiento en el juego.
        
        Args:
            col: Columna donde se quiere jugar (0-6)
            
        Returns:
            tuple: (nuevo_estado, es_válido, es_fin_juego, resultado)
                - nuevo_estado: array actualizado o None si es inválido
                - es_válido: bool indicando si el movimiento es válido
                - es_fin_juego: bool si el juego terminó
                - resultado: 1 (actual gana), -1 (oponente gana), 0 (empate), None (no terminó)
        """
        # Validar movimiento
        legal_moves = self.game.get_legal_moves()
        if col not in legal_moves:
            return None, False, False, None
        
        # Hacer el movimiento
        if not self.game.apply_move(col):
            return None, False, False, None
        
        # Obtener nuevo estado
        new_state = self.game.get_state()
        
        # Verificar fin del juego
        if self.game.is_game_over():
            winner = self.game.get_winner()
            if winner is None:
                # Empate
                outcome = 0
            elif winner == self.game.PLAYER1:
                # Jugador 1 (primero en mover) ganó
                # Pero necesitamos saber desde la perspectiva de quién movió
                # Si el ganador es el Player1 y el current_player es Player2, significa que Player1 acaba de ganar
                outcome = 1  # El que acaba de mover ganó
            else:
                # Player2 ganó
                outcome = -1  # El oponente ganó
            
            return new_state, True, True, outcome
        
        return new_state, True, False, None
    
    def get_legal_moves(self):
        """Retorna lista de columnas con movimientos legales."""
        return self.game.get_legal_moves()
    
    def get_board_visual(self):
        """Retorna el tablero en forma 2D simple (6, 7) en lugar de (3, 6, 7)."""
        return self.game.board.copy()

