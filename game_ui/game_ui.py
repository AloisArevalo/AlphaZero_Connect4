"""Interfaz gráfica para jugar Connect 4 contra la IA entrenada."""

import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import threading
import time
from game_engine import GameEngine


class Connect4GUI:
    """Interfaz gráfica tkinter para jugar Connect 4."""
    
    ROWS = 6
    COLS = 7
    CELL_SIZE = 60
    COLORS = {
        0: "#E8E8E8",      # Vacío (gris claro)
        1: "#FF6B6B",      # Jugador humano (rojo)
        2: "#4ECDC4",      # IA (turquesa)
    }
    
    def __init__(self, root, model_path="checkpoints/best_model.pt", ai_difficulty=100):
        """
        Inicializa la interfaz gráfica.
        
        Args:
            root: Ventana raíz de tkinter
            model_path: Ruta al modelo entrenado
            ai_difficulty: Simulaciones MCTS (25, 50, 100, 200)
        """
        self.root = root
        self.root.title("Connect 4 vs IA AlphaZero")
        self.root.resizable(False, False)
        
        # Cargar engine
        try:
            self.engine = GameEngine(model_path, mcts_simulations=ai_difficulty)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
            raise
        
        self.state = self.engine.reset_game()
        self.human_player = 1  # Humano juega con fichas rojas
        self.ai_player = 2     # IA juega con fichas turquesas
        self.current_player = 1
        self.game_over = False
        self.ai_thinking = False
        
        # Crear interfaz
        self._create_widgets()
        self._update_display()
    
    def _create_widgets(self):
        """Crea los widgets de la interfaz."""
        
        # Frame superior con título e información
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, columnspan=self.COLS, sticky="ew")
        
        title_label = ttk.Label(
            top_frame, 
            text="Connect 4 vs IA AlphaZero",
            font=("Arial", 16, "bold")
        )
        title_label.pack()
        
        self.status_label = ttk.Label(
            top_frame,
            text="",
            font=("Arial", 12),
            foreground="#FF6B6B"
        )
        self.status_label.pack()
        
        # Canvas para el tablero
        canvas_width = self.COLS * self.CELL_SIZE
        canvas_height = self.ROWS * self.CELL_SIZE
        self.canvas = tk.Canvas(
            self.root,
            width=canvas_width,
            height=canvas_height,
            bg="#1E90FF",
            highlightthickness=2,
            highlightbackground="#000080"
        )
        self.canvas.grid(row=1, column=0, columnspan=self.COLS, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        # Frame inferior con botones y leyenda
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.grid(row=2, column=0, columnspan=self.COLS, sticky="ew")
        
        # Leyenda
        legend_frame = ttk.Frame(bottom_frame)
        legend_frame.pack(side=tk.LEFT, padx=20)
        
        human_legend = tk.Canvas(legend_frame, width=30, height=30, bg="white", highlightthickness=0)
        human_legend.pack(side=tk.LEFT, padx=5)
        human_legend.create_oval(3, 3, 27, 27, fill=self.COLORS[1], outline="black", width=2)
        ttk.Label(legend_frame, text="Tú", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        ai_legend = tk.Canvas(legend_frame, width=30, height=30, bg="white", highlightthickness=0)
        ai_legend.pack(side=tk.LEFT, padx=5)
        ai_legend.create_oval(3, 3, 27, 27, fill=self.COLORS[2], outline="black", width=2)
        ttk.Label(legend_frame, text="IA", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        # Botones
        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(side=tk.RIGHT, padx=20)
        
        self.reset_button = ttk.Button(
            button_frame,
            text="Nuevo Juego",
            command=self._reset_game
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.hint_button = ttk.Button(
            button_frame,
            text="Sugerencia",
            command=self._show_hint
        )
        self.hint_button.pack(side=tk.LEFT, padx=5)
        
        self.quit_button = ttk.Button(
            button_frame,
            text="Salir",
            command=self.root.quit
        )
        self.quit_button.pack(side=tk.LEFT, padx=5)
        
        # Frame de información del juego
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.grid(row=3, column=0, columnspan=self.COLS, sticky="ew")
        
        self.info_label = ttk.Label(
            info_frame,
            text="",
            font=("Arial", 9),
            foreground="#666666"
        )
        self.info_label.pack()
    
    def _draw_board(self):
        """Dibuja el tablero en el canvas."""
        self.canvas.delete("all")
        
        # Obtener tablero simple 2D
        board = self.engine.get_board_visual()
        
        # Dibujar grid
        for row in range(self.ROWS):
            for col in range(self.COLS):
                x1 = col * self.CELL_SIZE
                y1 = row * self.CELL_SIZE
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE
                
                # Fondo de celda
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill="#1E90FF",
                    outline="#000080",
                    width=2
                )
                
                # Agujero del tablero
                center_x = x1 + self.CELL_SIZE // 2
                center_y = y1 + self.CELL_SIZE // 2
                radius = self.CELL_SIZE // 2 - 5
                
                # Ficha
                cell_value = board[row, col]
                color = self.COLORS.get(int(cell_value), self.COLORS[0])
                
                self.canvas.create_oval(
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius,
                    fill=color,
                    outline="black",
                    width=2
                )
    
    def _on_canvas_click(self, event):
        """Maneja clicks en el canvas."""
        if self.game_over or self.ai_thinking or self.current_player != self.human_player:
            return
        
        # Calcular columna
        col = event.x // self.CELL_SIZE
        
        if not (0 <= col < self.COLS):
            messagebox.showwarning("Movimiento inválido", "¡Columna fuera del rango!")
            return
        
        # Intentar hacer el movimiento
        new_state, valid, is_terminal, outcome = self.engine.step(col)
        
        if not valid:
            messagebox.showwarning("Movimiento inválido", "¡Esa columna está llena!")
            return
        
        self.state = new_state
        
        # Verificar si el humano ganó
        if is_terminal:
            self._end_game(outcome)
            return
        
        # Turno de la IA
        self._make_ai_move()
    
    def _make_ai_move(self):
        """La IA hace su movimiento (en un thread para no congelar la UI)."""
        self.ai_thinking = True
        self.current_player = self.ai_player
        self._update_display()
        self.root.update_idletasks()
        
        def ai_thread():
            try:
                time.sleep(0.5)  # Pequeño delay para que se vea más natural
                
                ai_move = self.engine.get_ai_move()
                
                # Hacer el movimiento
                new_state, valid, is_terminal, outcome = self.engine.step(ai_move)
                
                if valid:
                    self.state = new_state
                    
                    # Verificar si la IA ganó
                    if is_terminal:
                        self.root.after(0, lambda: self._end_game(outcome))
                    else:
                        self.current_player = self.human_player
                        self.ai_thinking = False
                        self.root.after(0, self._update_display)
                else:
                    messagebox.showerror("Error", "¡La IA intentó un movimiento inválido!")
                    self.ai_thinking = False
                    
            except Exception as e:
                messagebox.showerror("Error", f"Error en movimiento de IA:\n{e}")
                self.ai_thinking = False
        
        thread = threading.Thread(target=ai_thread, daemon=True)
        thread.start()
    
    def _end_game(self, outcome):
        """Termina el juego."""
        self.game_over = True
        self.ai_thinking = False
        
        if outcome == 1:
            result = "GANASTE!\nFelicidades, derrotaste a la IA!"
            color = "#00AA00"
        elif outcome == -1:
            result = "LA IA GANO\nMejor suerte la proxima vez!"
            color = "#AA0000"
        else:  # outcome == 0
            result = "EMPATE\nBuen juego!"
            color = "#AA7700"
        
        self._update_display()
        messagebox.showinfo("Fin del Juego", result)
    
    def _show_hint(self):
        """Muestra una sugerencia del siguiente movimiento."""
        if self.game_over or self.current_player != self.human_player:
            messagebox.showinfo("Sugerencia", "No es tu turno!")
            return
        
        messagebox.showinfo(
            "Sugerencia",
            "Piensa estrategicamente:\n"
            "1. Intenta conectar 4\n"
            "2. Bloquea a la IA\n"
            "3. Controla el centro del tablero"
        )
    
    def _update_display(self):
        """Actualiza la pantalla."""
        self._draw_board()
        
        # Obtener estadísticas
        board = self.engine.get_board_visual()
        human_count = (board == 1).sum()
        ai_count = (board == 2).sum()
        total_moves = human_count + ai_count
        
        if self.game_over:
            self.status_label.config(text="JUEGO TERMINADO", foreground="#FF0000")
            self.reset_button.config(state="normal")
            self.hint_button.config(state="disabled")
        elif self.ai_thinking:
            self.status_label.config(text="IA esta pensando...", foreground="#4ECDC4")
            self.hint_button.config(state="disabled")
        elif self.current_player == self.human_player:
            self.status_label.config(text="Tu turno - Haz clic en una columna", foreground="#FF6B6B")
            self.hint_button.config(state="normal")
        else:
            self.status_label.config(text="Turno de la IA", foreground="#4ECDC4")
            self.hint_button.config(state="disabled")
        
        # Información del tablero
        self.info_label.config(
            text=f"Fichas: Tú={human_count} | IA={ai_count} | Total={total_moves}/42"
        )
    
    def _reset_game(self):
        """Reinicia el juego."""
        self.state = self.engine.reset_game()
        self.current_player = self.human_player
        self.game_over = False
        self.ai_thinking = False
        self._update_display()


def main():
    """Punto de entrada de la aplicación."""
    root = tk.Tk()
    
    try:
        app = Connect4GUI(root, ai_difficulty=100)
        root.mainloop()
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
        root.destroy()
    except Exception as e:
        messagebox.showerror("Error", f"Error inesperado:\n{e}")
        root.destroy()


if __name__ == "__main__":
    main()
