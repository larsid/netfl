from keras import layers, models
from flwr.server.strategy import FedAvg

from netfl.core.task import Task, Dataset, DatasetInfo, DatasetPartitioner, TrainConfigs
from netfl.utils.partitioner import IidPartitioner


class MNIST(Task):
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            huggingface_path="ylecun/mnist",
            item_name="image",
            label_name="label",
        )
    
    def dataset_partitioner(self) -> DatasetPartitioner:
        return IidPartitioner()

    def normalized_dataset(self, raw_dataset: Dataset) -> Dataset:
        return Dataset(
            x=(raw_dataset.x / 255.0),
            y=raw_dataset.y,
        )

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

    def aggregation_strategy(self) -> type[FedAvg]:
        return FedAvg
    
    def train_configs(self) -> TrainConfigs:
	    return TrainConfigs(
            batch_size=32,
            epochs=1,
            fraction_fit=1.0,
            learning_rate=0.001,
            min_available=4,
            max_available=4,
            num_rounds=10,
            seed=42,
            shuffle=True,
            test_size=0.2,
        )


class MainTask(MNIST):
    pass
