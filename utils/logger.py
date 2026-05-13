# -*- coding: utf-8 -*-
"""
日志工具
"""
import logging
import sys
from pathlib import Path
import config


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    获取一个配置好的 logger。
    同时输出到控制台和日志文件。
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    # 格式
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 文件
    if log_file is None:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = str(config.LOG_DIR / f"{name}.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
