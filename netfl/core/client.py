import time
import json
from datetime import datetime

from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar

from netfl.core.task import Task
from netfl.utils.log import log


class Client(NumPyClient):
	def __init__(
		self,
		client_id: int,
		client_name: str,
		task: Task,
	) -> None:
		self._client_id = client_id
		self._client_name = client_name
		self._dataset, self._dataset_length = task.batch_dataset(
			task.train_dataset(client_id)
		)
		self._model = task.model()
		self._train_configs = task.train_configs()
		self._receive_time = 0
		self._send_time = 0

		task.print_configs()

	@property
	def client_id(self) -> int:
		return self._client_id
	
	def train_metrics(
		self,
		round: Scalar,
		dataset_length: int,
		train_time: float,
	) -> dict[str, Scalar]:
		metrics = {
			"client_id": self._client_id,
			"client_name": self._client_name,
			"round": round,
			"dataset_length": dataset_length,
			"train_time": train_time,
			"timestamp": datetime.now().isoformat(),
		}

		exchange_time = self._receive_time - self._send_time if self._send_time else None
		if exchange_time is not None:
			metrics["exchange_time"] = exchange_time

		return metrics

	def fit(self, parameters: NDArrays, configs: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
		self._receive_time = time.perf_counter()

		self._model.set_weights(parameters)
		start_train_time = time.perf_counter()

		self._model.fit(
			self._dataset,
			epochs=self._train_configs.epochs,
			verbose="2",
		)

		end_train_time = time.perf_counter()
		weights = self._model.get_weights()

		train_time = end_train_time - start_train_time

		metrics = self.train_metrics(
			configs["round"],
			self._dataset_length,
			train_time,
		)

		self._send_time = time.perf_counter()

		self.print_metrics(metrics)

		return (
			weights,
			self._dataset_length,
			metrics,
		)

	def print_metrics(self, metrics: dict[str, Scalar]) -> None:
		log(f"[ROUND {metrics['round']}]")
		log(f"[METRICS]\n{json.dumps(metrics, indent=2, default=str)}")

	def start(self, server_address: str, server_port: int) -> None:
		log(f"Starting client {self._client_id}")
		start_client(client=self.to_client(), server_address=f"{server_address}:{server_port}")
		log("Client has stopped")
