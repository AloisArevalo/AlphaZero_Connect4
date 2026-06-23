"""Motor de juego que envuelve el modelo entrenado para jugar interactivamente."""

import os
import sys
import torch
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connect4 import Config, Connect4, Connect4Net
from connect4.mcts import MCTS, MCTSNode
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
        self.mcts = MCTS(self.network, self.config, self.device, add_dirichlet_noise=False)
        # Nodo raíz del árbol MCTS reutilizado entre turnos
        self._mcts_root: Optional[MCTSNode] = None
    
    def get_ai_move(self) -> int:
        """
        Obtiene el siguiente movimiento de la IA usando MCTS con reutilización de árbol.

        El árbol construido en turnos anteriores se reutiliza: el subárbol correspondiente
        al estado actual ya tiene visitas acumuladas, por lo que las nuevas simulaciones
        refinan decisiones ya exploradas en lugar de partir de cero.

        Returns:
            int: Columna donde la IA quiere jugar (0-6)
        """
        policy, root = self.mcts.search(
            self.game,
            num_simulations=self.config.mcts_simulations,
            root=self._mcts_root,
        )
        # Guardar el nodo raíz actualizado; step() lo avanzará al hijo correcto
        self._mcts_root = root
        return int(policy.argmax())
    
    def reset_game(self):
        """Reinicia el juego y retorna el estado inicial."""
        self.game.reset()
        self._mcts_root = None
        return self.game.get_state()
    
    def step(self, col):
        """
        Realiza un movimiento en el juego y avanza el árbol MCTS al subárbol correspondiente.

        Args:
            col: Columna donde se quiere jugar (0-6)

        Returns:
            tuple: (nuevo_estado, es_válido, es_fin_juego, resultado)
                - nuevo_estado: array actualizado o None si es inválido
                - es_válido: bool indicando si el movimiento es válido
                - es_fin_juego: bool si el juego terminó
                - resultado: 1 (actual gana), -1 (oponente gana), 0 (empate), None (no terminó)
        """
        # Validar movimiento antes de tocar el árbol
        legal_moves = self.game.get_legal_moves()
        if col not in legal_moves:
            return None, False, False, None

        # Avanzar la raíz MCTS al hijo correspondiente a este movimiento.
        # Si ese hijo no fue explorado, se descarta el árbol (None = partir de cero la próxima vez).
        if self._mcts_root is not None:
            self._mcts_root = self._mcts_root.children.get(col)
        
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

