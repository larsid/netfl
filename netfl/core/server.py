from datetime import datetime

from flwr.server import ServerConfig, start_server
from flwr.common import ndarrays_to_parameters, NDArrays, Metrics, Scalar

from netfl.core.task import Task
from netfl.utils.log import log


class Server:
	def __init__(
		self,
		task: Task
	) -> None:
		self._dataset = task.test_dataset()
		self._model = task.model()
		self._strategy = task.aggregation_strategy()
		self._train_configs = task.train_configs()

	def aggregation(self, metrics: list[tuple[int, Metrics]]) -> Metrics:
		log(metrics)
		return {}

	def evaluate(self, round: int, parameters: NDArrays, configs: dict[str, Scalar]) -> tuple[float, dict[str, Scalar]]:
		self._model.set_weights(parameters)

		loss, accuracy = self._model.evaluate(
			self._dataset.x, 
			self._dataset.y, 
			verbose="2",
		)
		
		return (
			loss,
			{
				"round": round,
				"loss": loss, 
				"accuracy": accuracy, 
				"test_dataset_length": len(self._dataset.x),
				"timestamp": datetime.now().isoformat(),
			}
		)

	def start(self, server_port: int) -> None:
		start_server(
			config= ServerConfig(num_rounds=self._train_configs.num_rounds),
			server_address=f"0.0.0.0:{server_port}",
			strategy=self._strategy(
				fit_metrics_aggregation_fn=self.aggregation,
				fraction_evaluate=0,
				fraction_fit=self._train_configs.fraction_fit,
				initial_parameters=ndarrays_to_parameters(self._model.get_weights()),
				min_available_clients=self._train_configs.min_available,
				evaluate_fn=self.evaluate
			),
		)
		log("Server has stopped")
