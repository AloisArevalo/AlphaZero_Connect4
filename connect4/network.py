"""Red neuronal convolucional residual con cabezales de política y valor."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Bloque convolucional residual: conv -> BN -> ReLU -> conv -> BN + skip -> ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class Connect4Net(nn.Module):
    """
    Red convolucional residual con dos cabezales:
        - Política: probabilidades sobre las 7 columnas (softmax enmascarado)
        - Valor: escalar en [-1, 1] (tanh)
    """

    def __init__(self, rows: int = 6, cols: int = 7, num_channels: int = 128, num_res_blocks: int = 5):
        super().__init__()
        self.rows = rows
        self.cols = cols

        # Cuerpo convolucional
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # Cabeza de política
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * rows * cols, cols)

        # Cabeza de valor
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(4 * rows * cols, 128)
        self.value_fc2 = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialización de pesos compatible con entrenamiento desde cero."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor (batch, 3, rows, cols)
            legal_mask: tensor (batch, cols) con 1 para movimientos legales, 0 ilegales

        Returns:
            policy_logits: (batch, cols)
            value: (batch, 1) en rango [-1, 1]
        """
        # Cuerpo
        out = self.input_conv(x)
        for block in self.res_blocks:
            out = block(out)

        # Cabeza de política
        policy = self.policy_conv(out)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # Enmascarar movimientos ilegales
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)

        # Cabeza de valor
        value = self.value_conv(out)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value
