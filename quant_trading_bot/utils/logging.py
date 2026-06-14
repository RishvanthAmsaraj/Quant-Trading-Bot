"""Project-wide logging configuration.

A single ``get_logger`` function gives every module a consistent,
colourised console logger plus an optional rotating file handler.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialised = False


def _init_root(log_file: Optional[Path] = None, level: int = logging.INFO) -> None:
    global _initialised
    if _initialised:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # Optional rotating file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(file_handler)

    _initialised = True


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Return a configured logger with the given name.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__``).
    log_file:
        Optional path to a rotating log file.  Created if missing.
    """
    _init_root(log_file=log_file)
    return logging.getLogger(name)
