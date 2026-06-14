"""Entorno del juego de Conecta 4."""

from typing import List, Optional

import numpy as np


class Connect4:
    """
    Entorno de Conecta 4 con tablero 6x7.
    El estado se representa como tensor (3, rows, cols):
        canal 0: fichas del jugador actual
        canal 1: fichas del oponente
        canal 2: espacios vacíos
    """

    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2

    def __init__(self, rows: int = 6, cols: int = 7, win_length: int = 4):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.board: np.ndarray = np.zeros((rows, cols), dtype=np.int8)
        self.current_player: int = self.PLAYER1
        self.move_count: int = 0
        self.winner: Optional[int] = None
        self.game_over: bool = False

    def reset(self) -> None:
        """Reinicia el tablero y el estado del juego."""
        self.board.fill(self.EMPTY)
        self.current_player = self.PLAYER1
        self.move_count = 0
        self.winner = None
        self.game_over = False

    def get_state(self) -> np.ndarray:
        """
        Devuelve el estado desde la perspectiva del jugador actual.
        Forma: (3, rows, cols), valores 0 o 1.
        """
        state = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.board[r, c]
                if cell == self.EMPTY:
                    state[2, r, c] = 1.0
                elif cell == self.current_player:
                    state[0, r, c] = 1.0
                else:
                    state[1, r, c] = 1.0
        return state

    def get_legal_moves(self) -> List[int]:
        """Devuelve las columnas donde se puede colocar una ficha."""
        return [c for c in range(self.cols) if self.board[0, c] == self.EMPTY]

    def apply_move(self, col: int) -> bool:
        """
        Aplica un movimiento en la columna indicada.
        Devuelve True si el movimiento fue válido.
        """
        if self.game_over or col not in self.get_legal_moves():
            return False

        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == self.EMPTY:
                self.board[row, col] = self.current_player
                self.move_count += 1
                if self._check_winner(row, col):
                    self.winner = self.current_player
                    self.game_over = True
                elif self.move_count == self.rows * self.cols:
                    self.game_over = True
                else:
                    self.current_player = (
                        self.PLAYER2 if self.current_player == self.PLAYER1 else self.PLAYER1
                    )
                return True
        return False

    def _check_winner(self, row: int, col: int) -> bool:
        """Comprueba si el último movimiento generó una línea ganadora."""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # Dirección positiva
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # Dirección negativa
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= self.win_length:
                return True
        return False

    def is_game_over(self) -> bool:
        """Indica si la partida ha terminado (victoria o empate)."""
        return self.game_over

    def get_winner(self) -> Optional[int]:
        """Devuelve el jugador ganador (1 o 2) o None si hay empate."""
        return self.winner

    def get_current_player(self) -> int:
        """Devuelve el jugador que debe mover (1 o 2)."""
        return self.current_player

    def clone(self) -> "Connect4":
        """Crea una copia independiente del juego."""
        new_game = Connect4(self.rows, self.cols, self.win_length)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        new_game.winner = self.winner
        new_game.game_over = self.game_over
        return new_game
