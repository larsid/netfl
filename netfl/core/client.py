from datetime import datetime

from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar

from netfl.core.task import Task
from netfl.utils.log import log


class Client(NumPyClient):
	def __init__(
		self,
		client_id: int,
		task: Task,
	) -> None:
		self._client_id = client_id
		self._dataset = task.train_dataset(client_id)
		self._model = task.model()
		self._train_configs = task.train_configs()

		task.print_configs()
		
	@property
	def client_id(self) -> int:
		return self._client_id

	def fit(self, parameters: NDArrays, configs: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
		self._model.set_weights(parameters)

		self._model.fit(
			self._dataset.x,
			self._dataset.y,
			batch_size=self._train_configs.batch_size,
			epochs=self._train_configs.epochs,
			verbose="2",
		)

		metrics = self.fit_metrics(configs)
		
		return (
			self._model.get_weights(),
			len(self._dataset.x),
			metrics,
		)
	
	def fit_metrics(self, configs: dict[str, Scalar]) -> dict[str, Scalar]:
		return {
			"client_id": self._client_id,
			"round": configs["round"],
			"dataset_length": len(self._dataset.x),
			"cpu_avg": 0,
			"memory_avg": 0,
			"timestamp": datetime.now().isoformat(),
		}

	def start(self, server_address: str, server_port: int) -> None:
		log(f"Starting client {self._client_id}")
		start_client(client=self.to_client(), server_address=f"{server_address}:{server_port}")
		log("Client has stopped")
