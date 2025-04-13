import logging
from dataclasses import asdict

from flwr.server import ServerConfig, start_server

from netfl.core.task import Task


class Server:
    def __init__(
        self,
        task: Task
    ) -> None:
        self._task = task

    def start(self, server_port: int) -> None:
        logging.info("Dataset info: %s", asdict(self._task._dataset_info))
        logging.info("Train config: %s", asdict(self._task._train_config))
        start_server(
            config= ServerConfig(num_rounds=self._task._train_config.num_rounds),
            server_address=f"0.0.0.0:{server_port}",
            strategy=self._task.aggregation_strategy(),
        )
        logging.info("Server has stopped")
