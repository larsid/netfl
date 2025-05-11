import logging
import os
from datetime import datetime


def setup_logs(filename_prefix: str = "", level: int = logging.INFO) -> None:
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if filename_prefix:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.log")
        handlers.append(logging.FileHandler(log_filename))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
    )
