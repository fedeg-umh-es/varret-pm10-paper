"""Minimal logging helpers for P33 scripts."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with basic configuration."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
