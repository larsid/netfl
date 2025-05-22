from keras import layers, models
from flwr.server.strategy import FedAvg

from netfl.core.task import Task, Dataset, DatasetInfo, DatasetPartitioner, TrainConfigs
from netfl.core.partitioner import IidPartitioner


class Cifar10(Task):
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            huggingface_path="uoft-cs/cifar10",
            item_name="img",
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
        model = models.Sequential(
            [
                layers.Input(shape=(32, 32, 3)),

                layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Flatten(),

                layers.Dense(512, activation="relu"),
                layers.Dropout(0.5),

                layers.Dense(10, activation="softmax"),
            ]
        )
        
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
            learning_rate=0.001,
            min_clients=4,
            max_clients=4,
            num_rounds=10,
            seed_data=42,
            shuffle_data=True,
        )


class MainTask(Cifar10):
    pass
