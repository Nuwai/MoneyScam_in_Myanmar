import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Optional


def setup_logging(log_dir: str = "logs", level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure application logging with console + rotating file handler."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = RotatingFileHandler(os.path.join(log_dir, "app.log"), maxBytes=5_000_000, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
