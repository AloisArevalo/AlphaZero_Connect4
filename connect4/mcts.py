"""Monte Carlo Tree Search guiado por la red neuronal."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .game import Connect4
from .network import Connect4Net
from .tensor_utils import np_to_torch, torch_to_np


class MCTSNode:
    """Nodo del árbol de búsqueda MCTS."""

    def __init__(self, prior: float = 0.0):
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, "MCTSNode"] = {}
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """
    Búsqueda Monte Carlo guiada por la red neuronal con selección PUCT.
    Fórmula PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
    """

    def __init__(
        self,
        network: Connect4Net,
        config: Config,
        device: torch.device,
        add_dirichlet_noise: bool = True,
    ):
        self.network = network
        self.config = config
        self.device = device
        self.add_dirichlet_noise = add_dirichlet_noise
        self.network.eval()

    def search(self, game: Connect4, num_simulations: Optional[int] = None) -> np.ndarray:
        """
        Ejecuta MCTS desde el estado actual del juego.
        Devuelve la política de búsqueda: distribución proporcional a visitas en la raíz.
        """
        if num_simulations is None:
            num_simulations = self.config.mcts_simulations

        root = MCTSNode()
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return np.zeros(self.config.cols, dtype=np.float32)

        # Expandir raíz
        self._expand(root, game, add_noise=self.add_dirichlet_noise)

        for _ in range(num_simulations):
            sim_game = game.clone()
            node = root
            search_path = [node]

            # Fase 1: Selección — descender por PUCT hasta un nodo hoja
            while node.is_expanded and node.children:
                col, node = self._select_child(node)
                sim_game.apply_move(col)
                search_path.append(node)

            # Fase 2: Expansión y evaluación
            value = 0.0
            if not sim_game.is_game_over():
                value = self._expand(node, sim_game, add_noise=False)
            else:
                value = self._terminal_value(sim_game)

            # Fase 3: Retropropagación (negar valor al subir porque cambia el jugador)
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        return self._get_policy(root, legal_moves)

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Selecciona el hijo con mayor puntuación PUCT."""
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = math.sqrt(total_visits)

        best_score = -float("inf")
        best_col = -1
        best_child = None

        for col, child in node.children.items():
            q = child.q_value
            u = (
                self.config.c_puct
                * child.prior
                * sqrt_total
                / (1 + child.visit_count)
            )
            score = q + u
            if score > best_score:
                best_score = score
                best_col = col
                best_child = child

        return best_col, best_child

    def _expand(self, node: MCTSNode, game: Connect4, add_noise: bool = False) -> float:
        """
        Expande un nodo hoja evaluando con la red neuronal.
        Devuelve el valor estimado desde la perspectiva del jugador actual.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            node.is_expanded = True
            return 0.0

        state = game.get_state()
        state_tensor = np_to_torch(state, self.device).unsqueeze(0)

        legal_mask = torch.zeros(1, self.config.cols, device=self.device)
        for col in legal_moves:
            legal_mask[0, col] = 1.0

        with torch.no_grad():
            policy_logits, value = self.network(state_tensor, legal_mask)
            policy_probs = torch_to_np(F.softmax(policy_logits, dim=1).squeeze(0))
            value = value.item()

        if add_noise and self.add_dirichlet_noise:
            noise = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(legal_moves)
            )
            for i, col in enumerate(legal_moves):
                policy_probs[col] = (
                    (1 - self.config.dirichlet_epsilon) * policy_probs[col]
                    + self.config.dirichlet_epsilon * noise[i]
                )

        for col in legal_moves:
            node.children[col] = MCTSNode(prior=policy_probs[col])

        node.is_expanded = True
        return value

    def _terminal_value(self, game: Connect4) -> float:
        """Valor de un estado terminal: +1 victoria, -1 derrota, 0 empate."""
        winner = game.get_winner()
        if winner is None:
            return 0.0
        # Tras apply_move, current_player ya cambió. Reconstruimos el último jugador:
        last_player = (
            Connect4.PLAYER2 if game.current_player == Connect4.PLAYER1 else Connect4.PLAYER1
        )
        if winner == last_player:
            return 1.0
        return -1.0

    def _get_policy(self, root: MCTSNode, legal_moves: List[int]) -> np.ndarray:
        """Convierte las visitas de los hijos de la raíz en una distribución de probabilidad."""
        policy = np.zeros(self.config.cols, dtype=np.float32)
        total_visits = sum(root.children[col].visit_count for col in legal_moves)

        if total_visits == 0:
            for col in legal_moves:
                policy[col] = 1.0 / len(legal_moves)
            return policy

        for col in legal_moves:
            policy[col] = root.children[col].visit_count / total_visits

        return policy
