__version__ = "1.0.0"
__all__ = ("__version__", "LLM", "Message")

import logging

from aihuber.aihuber import LLM, Message

logger = logging.getLogger("aihuber")
logger.addHandler(logging.NullHandler())
