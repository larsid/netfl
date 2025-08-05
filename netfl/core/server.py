import gc
import json
from datetime import datetime

from keras import ops, backend
from flwr.server import ServerConfig, start_server
from flwr.common import ndarrays_to_parameters, NDArrays, Metrics, Scalar

from netfl.core.task import Task
from netfl.utils.log import log


class Server:
	def __init__(
		self,
		task: Task
	) -> None:
		dataset = task.test_dataset()

		self._task = task
		self._dataset_x = ops.convert_to_tensor(dataset.x)
		self._dataset_y = ops.convert_to_tensor(dataset.y)
		self._model = task.model()
		self._strategy = task.aggregation_strategy()
		self._train_configs = task.train_configs()
		self._train_metrics = []
		self._evaluate_metrics = []
		
		task.print_configs()
		task.delete_downloaded_dataset()

	def _clear_model(self) -> None:
		backend.clear_session()
		del self._model
		gc.collect()
		self._model = self._task.model()

	def fit_configs(self, round: int) -> dict[str, Scalar]:
		return { 
			"round": round,
		}

	def train_metrics(self, metrics: list[tuple[int, Metrics]]) -> Metrics:
		train_metrics = [m for _, m in metrics]
		train_metrics = sorted(train_metrics, key=lambda m: m["client_id"])
		self._train_metrics.extend(train_metrics)
		return {}

	def evaluate(self, round: int, parameters: NDArrays, configs: dict[str, Scalar]) -> tuple[float, dict[str, Scalar]]:
		self._clear_model()
		self._model.set_weights(parameters)

		loss, accuracy = self._model.evaluate(
			self._dataset_x,
			self._dataset_y,
			verbose="2",
		)

		self._evaluate_metrics.append({
			"round": round,
			"loss": loss,
			"accuracy": accuracy,
			"dataset_length": int(self._dataset_x.shape[0]),
			"timestamp": datetime.now().isoformat(),
		})
		
		return (
			loss,
			{},
		)
	
	def print_metrics(self):
		metrics = {
			"train": self._train_metrics,
			"evaluate": self._evaluate_metrics,
		}
		log(f"[METRICS]\n{json.dumps(metrics, indent=2)}")

	def start(self, server_port: int) -> None:
		start_server(
			config= ServerConfig(num_rounds=self._train_configs.num_rounds),
			server_address=f"0.0.0.0:{server_port}",
			strategy=self._strategy(
				on_fit_config_fn=self.fit_configs,
				fit_metrics_aggregation_fn=self.train_metrics,
				fraction_evaluate=0,
				initial_parameters=ndarrays_to_parameters(self._model.get_weights()),
				min_fit_clients=self._train_configs.num_clients,
				min_evaluate_clients=self._train_configs.num_clients,
				min_available_clients=self._train_configs.num_clients,
				evaluate_fn=self.evaluate
			),
		)
		self.print_metrics()
		log("Server has stopped")
