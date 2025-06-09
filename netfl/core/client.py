import time
from datetime import datetime

from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar

from netfl.core.task import Task
from netfl.utils.log import log
from netfl.utils.metrics import measure_time


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
		self._receive_time = 0
		self._send_time = 0

		task.print_configs()
		
	@property
	def client_id(self) -> int:
		return self._client_id

	def fit(self, parameters: NDArrays, configs: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
		self._receive_time = time.perf_counter()

		self._model.set_weights(parameters)

		_, train_time = measure_time(
			lambda: self._model.fit(
				self._dataset.x,
				self._dataset.y,
				batch_size=self._train_configs.batch_size,
				epochs=self._train_configs.epochs,
				verbose="2",
			)	
		)

		weights = self._model.get_weights()
		dataset_length = len(self._dataset.x)
		metrics = self.fit_metrics(configs["round"], dataset_length, train_time)
		
		self._send_time = time.perf_counter()

		return (
			weights,
			dataset_length,
			metrics,
		)
	
	def fit_metrics(self, round: Scalar, dataset_length: int, train_time: float) -> dict[str, Scalar]:
		metrics = {
			"client_id": self._client_id,
			"round": round,
			"dataset_length": dataset_length,
			"train_time": train_time,
			"cpu_avg": 0,
			"memory_avg": 0,
			"timestamp": datetime.now().isoformat(),
		}

		if self._send_time != 0:
			metrics["update_time"] = self._receive_time - self._send_time

		return metrics

	def start(self, server_address: str, server_port: int) -> None:
		log(f"Starting client {self._client_id}")
		start_client(client=self.to_client(), server_address=f"{server_address}:{server_port}")
		log("Client has stopped")
