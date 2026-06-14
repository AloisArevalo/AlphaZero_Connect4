"""Utilidades de conversión entre tensores de PyTorch y arrays de NumPy."""

from typing import Optional

import numpy as np
import torch


def np_to_torch(arr: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convierte un ndarray a tensor de PyTorch.
    Usa fallback vía listas si el puente torch↔numpy no está disponible
    (p. ej. incompatibilidad NumPy 2.x con PyTorch compilado contra NumPy 1.x).
    """
    try:
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
    except RuntimeError:
        tensor = torch.tensor(arr.tolist(), dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def torch_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convierte un tensor de PyTorch a ndarray con el mismo fallback seguro."""
    try:
        return tensor.detach().cpu().numpy()
    except RuntimeError:
        return np.array(tensor.detach().cpu().tolist(), dtype=np.float32)
