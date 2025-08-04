import time
import traceback
from datetime import datetime

from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar

from netfl.core.task import Task
from netfl.utils.log import log
from netfl.utils.metrics import ResourceSampler, measure_time


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
		self._resource_sampler = ResourceSampler()

		task.print_configs()

	@property
	def client_id(self) -> int:
		return self._client_id

	def fit(self, parameters: NDArrays, configs: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
		try:
			self._receive_time = time.perf_counter()
			self._model.set_weights(parameters)
			self._resource_sampler.start()

			_, train_time = measure_time(
				lambda: self._model.fit(
					self._dataset.x,
					self._dataset.y,
					batch_size=self._train_configs.batch_size,
					epochs=self._train_configs.epochs,
					verbose="2",
				)	
			)

			cpu_avg_percent, memory_avg_mb = self._resource_sampler.stop()
			weights = self._model.get_weights()
			dataset_length = len(self._dataset.x)
			
			metrics = self.fit_metrics(
				configs["round"],
				dataset_length,
				train_time,
				cpu_avg_percent,
				memory_avg_mb
			)

			self._send_time = time.perf_counter()
			return weights,	dataset_length,	metrics
		except Exception as e:
			log(f"Client {self._client_id} error in fit(): {e}")
			traceback.print_exc()
			raise
	
	def fit_metrics(
		self,
		round: Scalar,
		dataset_length: int,
		train_time: float,
		cpu_avg_percent: float,
		memory_avg_mb: float,
	) -> dict[str, Scalar]:
		metrics = {
			"client_id": self._client_id,
			"round": round,
			"dataset_length": dataset_length,
			"train_time": train_time,
			"cpu_avg_percent": cpu_avg_percent,
			"memory_avg_mb": memory_avg_mb,
			"timestamp": datetime.now().isoformat(),
		}

		exchange_time = self._receive_time - self._send_time if self._send_time else None
		if exchange_time is not None:
			metrics["exchange_time"] = exchange_time

		return metrics

	def start(self, server_address: str, server_port: int) -> None:
		try:
			log(f"Starting client {self._client_id}")
			start_client(client=self.to_client(), server_address=f"{server_address}:{server_port}")
		except Exception as e:
			log(f"Client {self._client_id} error in start(): {e}")
			traceback.print_exc()
			raise
		finally:
			log(f"Client {self._client_id} has stopped")

