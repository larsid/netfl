import logging
import os
from datetime import datetime

from flwr.common.logger import FLOWER_LOGGER
from flwr.common.logger import log as flwr_log


LOG_DIR = "logs"


def setup_log_file(identifier: str) -> None:
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	
	try:
		os.makedirs(LOG_DIR, exist_ok=True)
	except OSError as e:
		raise OSError(f"Failed to create log directory '{LOG_DIR}': {e}") from e

	safe_identifier = "".join(c if c.isalnum() or c in "-_." else "_" for c in identifier)
	filename = os.path.join(LOG_DIR, f"{timestamp}_{safe_identifier}.log")

	try:
		file_handler = logging.FileHandler(filename)
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		FLOWER_LOGGER.addHandler(file_handler)
	except OSError as e:
		raise OSError(f"Failed to create log file '{filename}': {e}") from e


def log(msg: object) -> None:
	flwr_log(logging.INFO, msg)
