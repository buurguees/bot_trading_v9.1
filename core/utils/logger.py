from __future__ import annotations
from loguru import logger
import sys, pathlib

def init_logger(json: bool = True, level: str = "INFO", path: str | None = None):
    logger.remove()
    logger.add(sys.stdout, level=level, serialize=json)
    if path:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(path, level=level, serialize=json, rotation="10 MB", retention="30 days")
    return logger
