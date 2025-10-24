import logging
import os
from datetime import datetime

from flwr.common.logger import FLOWER_LOGGER
from flwr.common.logger import log as flwr_log


LOG_DIR = "logs"


def setup_log_file(identifier: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not any(isinstance(h, logging.StreamHandler) for h in FLOWER_LOGGER.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        FLOWER_LOGGER.addHandler(console_handler)

    FLOWER_LOGGER.setLevel(logging.INFO)
    log_dir_path = os.path.join(os.getcwd(), LOG_DIR)

    try:
        os.makedirs(log_dir_path, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create log directory '{log_dir_path}': {e}") from e

    safe_identifier = "".join(
        c if c.isalnum() or c in "-_." else "_" for c in identifier
    )

    filename = os.path.join(log_dir_path, f"{timestamp}_{safe_identifier}.log")

    try:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        FLOWER_LOGGER.addHandler(file_handler)
        log(f"Log file created: {filename}")
    except OSError as e:
        raise OSError(f"Failed to create log file '{filename}': {e}") from e


def log(msg: object) -> None:
    flwr_log(logging.INFO, msg)
