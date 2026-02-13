import logging
import sys
from logging.handlers import RotatingFileHandler
from .config import LOGS_DIR


def setup_logger(name=__name__, log_file="app.log", level=logging.INFO):
    """
    Setup a logger with console and file handlers.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicates
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        LOGS_DIR / log_file, maxBytes=10485760, backupCount=5  # 10MB
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Create a default logger
logger = setup_logger("login_anomaly_detection")
