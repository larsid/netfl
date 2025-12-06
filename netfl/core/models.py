from keras import layers, models, initializers, optimizers, losses


DEFAULT_INITIALIZER = initializers.VarianceScaling(
    scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
)


def cnn3(
    input_shape: tuple[int, int, int],
    output_classes: int,
    optimizer: optimizers.Optimizer,
) -> models.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.Conv2D(32, (5, 5), activation="relu", kernel_initializer=DEFAULT_INITIALIZER),  # type: ignore[arg-type]
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (5, 5), activation="relu", kernel_initializer=DEFAULT_INITIALIZER),  # type: ignore[arg-type]
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu", kernel_initializer=DEFAULT_INITIALIZER),  # type: ignore[arg-type]
            layers.Dense(output_classes, kernel_initializer=DEFAULT_INITIALIZER),  # type: ignore[arg-type]
        ]
    )

    model.compile(
        optimizer=optimizer,  # type: ignore[arg-type]
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
