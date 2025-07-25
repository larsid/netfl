import logging
import os
from datetime import datetime

from flwr.common.logger import FLOWER_LOGGER
from flwr.common.logger import log as flwr_log


def setup_log_file(identifier: str) -> None:
    dir = "logs"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, f"{timestamp}_{identifier}.log")

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    FLOWER_LOGGER.addHandler(file_handler)


def log(msg: object):
    flwr_log(logging.INFO, msg)
