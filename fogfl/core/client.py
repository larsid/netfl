import logging

from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar

from fogfl.core.task import Task


class Client(NumPyClient):
    def __init__(
        self,
        client_id: int,
        task: Task,
    ) -> None:
        self._client_id = client_id
        self._model = task.model()
        self._dataset = task.client_dataset(client_id)
        self._batch_size = task._train_config.batch_size
        self._epochs = task._train_config.epochs
        self._verbose = task._train_config.verbose

    @property
    def client_id(self):
        return self._client_id

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self._model.set_weights(parameters)        
        self._model.fit(
            self._dataset.x_train,
            self._dataset.y_train,
            batch_size=self._batch_size,
            epochs=self._epochs,
            verbose=self._verbose,
        )
        return (
            self._model.get_weights(),
            len(self._dataset.x_train),
            {},
        )

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        self._model.set_weights(parameters)
        loss, accuracy = self._model.evaluate(
            self._dataset.x_test, 
            self._dataset.y_test, 
            verbose=self._verbose,
        )
        return (
            loss,
            len(self._dataset.x_test),
            {"client_id": self._client_id, "loss": loss, "accuracy": accuracy}
        )

    def start(self, server_address: str, server_port: int) -> None:
        logging.info(f"Starting client {self._client_id}")
        start_client(
            client=self.to_client(),
            server_address=f"{server_address}:{server_port}",
        )
