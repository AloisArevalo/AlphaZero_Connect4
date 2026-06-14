"""Búfer de repetición y entrenamiento de la red neuronal."""

import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import Config
from .network import Connect4Net
from .self_play import Experience
from .tensor_utils import np_to_torch


class ReplayBuffer:
    """Almacena las últimas N experiencias (estado, política, valor) para entrenamiento."""

    def __init__(self, max_size: int):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, experiences: List[Experience]) -> None:
        self.buffer.extend(experiences)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


def compute_loss(
    network: Connect4Net,
    batch: List[Experience],
    device: torch.device,
    weight_decay: float,
) -> Tuple[torch.Tensor, float, float]:
    """
    Calcula la pérdida combinada:
        total = value_loss (MSE) + policy_loss (cross-entropy) + L2 regularization
    """
    states = np_to_torch(np.stack([exp[0] for exp in batch]), device)
    target_policies = np_to_torch(np.stack([exp[1] for exp in batch]), device)
    target_values = torch.tensor(
        [exp[2] for exp in batch], dtype=torch.float32, device=device
    ).unsqueeze(1)

    # Máscara legal: cualquier columna con probabilidad MCTS > 0 es legal
    legal_mask = (target_policies > 0).float()

    policy_logits, pred_values = network(states, legal_mask)

    # Pérdida de valor (MSE)
    value_loss = F.mse_loss(pred_values, target_values)

    # Pérdida de política (entropía cruzada con distribución MCTS)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

    # Regularización L2 manual sobre los pesos
    l2_reg = torch.tensor(0.0, device=device)
    for param in network.parameters():
        l2_reg = l2_reg + param.pow(2).sum()
    l2_loss = weight_decay * l2_reg

    total_loss = value_loss + policy_loss + l2_loss
    return total_loss, value_loss.item(), policy_loss.item()


def train_network(
    network: Connect4Net,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    config: Config,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Entrena la red durante varias épocas sobre muestras aleatorias del búfer.
    Devuelve las pérdidas medias de la última época.
    """
    if len(buffer) < config.batch_size:
        return 0.0, 0.0, 0.0

    network.train()
    avg_total, avg_value, avg_policy = 0.0, 0.0, 0.0
    num_batches = 0

    for _ in range(config.train_epochs):
        batches_per_epoch = max(1, len(buffer) // config.batch_size)
        for _ in range(batches_per_epoch):
            batch = buffer.sample(config.batch_size)
            optimizer.zero_grad()
            loss, v_loss, p_loss = compute_loss(
                network, batch, device, config.weight_decay
            )
            loss.backward()
            optimizer.step()
            avg_total += loss.item()
            avg_value += v_loss
            avg_policy += p_loss
            num_batches += 1

    if num_batches > 0:
        avg_total /= num_batches
        avg_value /= num_batches
        avg_policy /= num_batches

    return avg_total, avg_value, avg_policy
