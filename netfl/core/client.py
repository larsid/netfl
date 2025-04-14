import logging
from dataclasses import asdict

from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar

from netfl.core.task import Task


class Client(NumPyClient):
    def __init__(
        self,
        client_id: int,
        task: Task,
    ) -> None:
        self._client_id = client_id
        self._task = task
        self._model = task.model()
        self._dataset = task.dataset(task._dataset_partition(client_id))

    @property
    def client_id(self):
        return self._client_id

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self._model.set_weights(parameters)        
        self._model.fit(
            self._dataset.x_train,
            self._dataset.y_train,
            batch_size=self._task._train_config.batch_size,
            epochs=self._task._train_config.epochs,
            verbose="2",
        )
        train_dataset_size = len(self._dataset.x_train)
        return (
            self._model.get_weights(),
            train_dataset_size,
            {},
        )

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        self._model.set_weights(parameters)
        loss, accuracy = self._model.evaluate(
            self._dataset.x_test, 
            self._dataset.y_test, 
            verbose="2",
        )
        train_dataset_size = len(self._dataset.x_train)
        test_dataset_size = len(self._dataset.x_test)
        return (
            loss,
            test_dataset_size,
            {
                "client_id": self._client_id, 
                "loss": loss, 
                "accuracy": accuracy, 
                "train_dataset_size": train_dataset_size,
                "test_dataset_size": test_dataset_size,
            }
        )

    def start(self, server_address: str, server_port: int) -> None:
        logging.info(f"Starting client {self._client_id}")
        logging.info("Dataset info: %s", asdict(self._task._dataset_info))
        logging.info("Train config: %s", asdict(self._task._train_config))
        start_client(
            client=self.to_client(),
            server_address=f"{server_address}:{server_port}",
        )
        logging.info("Client has stopped")
