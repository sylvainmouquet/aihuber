import logging
import pytest
import structlog
from aihuber.logger import (
    setup_structlog,
    get_logger,
)  # Remplacez par le nom de votre fichier


@pytest.fixture(autouse=True)
def clean_logging():
    """Réinitialise structlog et les handlers de logging avant chaque test."""
    structlog.reset_defaults()
    # Supprimer les handlers existants pour forcer basicConfig
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    yield


def test_logger_normal():
    setup_structlog(log_level="DEBUG", json_logs=False)
    logger = get_logger("aihuber")

    logger.debug("Demo")
    logger.info("Demo")
    logger.warn("Demo")
    logger.error("Demo")
    logger.critical("Demo")


def test_logger_json():
    setup_structlog(log_level="DEBUG", json_logs=True)
    logger = get_logger("aihuber")

    logger.debug("Demo")
    logger.info("Demo")
    logger.warn("Demo")
    logger.error("Demo")
    logger.critical("Demo")


""" 

def test_logger_exception(capsys):
    setup_structlog(json_logs=True)
    logger = get_logger("error_test")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("calcul_failed")

    captured = json.loads(capsys.readouterr().out.strip())
    assert "exception" in captured
    assert "ZeroDivisionError" in captured["exception"]
"""
