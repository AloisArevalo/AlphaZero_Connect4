"""Configuración de logging para el entrenamiento."""

import logging
from typing import Optional


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configura logging en consola y opcionalmente en archivo."""
    logger = logging.getLogger("AlphaZeroConnect4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
