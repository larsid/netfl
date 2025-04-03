import logging

from flwr.server import ServerConfig, start_server

from fogfl.core.task import Task


class Server:
    def __init__(
        self,
        task: Task
    ) -> None:
        self._aggregation_strategy = task.aggregation_strategy()
        self._num_rounds = task.train_config.num_rounds

    def start(self, server_port: int) -> None:
        start_server(
            config= ServerConfig(num_rounds=self._num_rounds),
            server_address=f"0.0.0.0:{server_port}",
            strategy=self._aggregation_strategy,
        )
        logging.info("Server has stopped")
