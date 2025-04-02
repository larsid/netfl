from keras import layers, models
from flwr.server.strategy import Strategy, FedAvg
from flwr.common import Metrics

from fogfl.core.task import Dataset, Task, TrainConfig, DatasetConfig


class MNIST(Task):
    def __init__(self) -> None:
        train_config = TrainConfig(
            batch_size=32,
            epochs=1,
            fraction_evaluate=1.0,
            fraction_fit=1.0,
            learning_rate=0.001,
            min_available=2,
            max_available=2,
            num_rounds=3,
            seed=42,
            shuffle=True,
            test_size=0.2,
            verbose="2",
        )
        dataset_config = DatasetConfig(
            dataset_name="ylecun/mnist",
            item_name="image",
            label_name="label",
        )
        super().__init__(train_config, dataset_config)

    def client_dataset(self, client_id: int) -> Dataset:
        dataset = self._dataset_partition(client_id)
        normalized_dataset = Dataset(
            x_train=(dataset.x_train / 255.0),
            x_test=(dataset.x_test / 255.0),
            y_train=dataset.y_train,
            y_test=dataset.y_test,
        )
        return normalized_dataset

    def model(self) -> models.Model:
        model = models.Sequential([
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def aggregation_strategy(self) -> Strategy:
        return self._aggregation_strategy_factory(FedAvg)
    
    def aggregation_evaluate_metrics(self, metrics: list[tuple[int, Metrics]]) -> Metrics:
        print(metrics)
        return {}


class MainTask(MNIST):
    pass
