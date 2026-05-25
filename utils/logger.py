# utils/logger.py
# Thin wrapper around stdlib logging (defers to utils.logging_config levels).

import logging
import traceback
from typing import Dict, Any


class Logger:
    """Structured logger wrapper — respects application-wide log levels."""

    def __init__(self, name: str = __name__):
        self.logger = logging.getLogger(name)
        self.logger.propagate = True

    def info(self, message: str, **extra):
        if extra:
            message = f"{message} {extra}"
        self.logger.info(message)

    def debug(self, message: str, **extra):
        if extra:
            message = f"{message} {extra}"
        self.logger.debug(message)

    def warning(self, message: str, **extra):
        if extra:
            message = f"{message} {extra}"
        self.logger.warning(message)

    def error(self, message: str, exc_info=None, **extra):
        if extra:
            message = f"{message} {extra}"
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, **extra):
        if extra:
            message = f"{message} {extra}"
        self.logger.critical(message)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        self.info(f"Event: {event_type}", **data)

    def log_prompt(self, prompt: str, max_chars: int = 200):
        truncated = prompt[:max_chars] + "..." if len(prompt) > max_chars else prompt
        self.debug(f"Prompt: {truncated}", prompt_length=len(prompt))

    def log_response(self, response: str, max_chars: int = 200):
        truncated = response[:max_chars] + "..." if len(response) > max_chars else response
        self.debug(f"Response: {truncated}", response_length=len(response))
