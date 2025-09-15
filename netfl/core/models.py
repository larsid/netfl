from keras import layers, models, optimizers


def cnn3(
		input_shape: tuple[int, int, int],
		output_classes: int,
		optimizer: optimizers.Optimizer,
	) -> models.Model:
	model = models.Sequential([
		layers.Input(shape=input_shape),

		layers.Conv2D(32, kernel_size=(5, 5), activation="relu", padding="valid"),
		layers.MaxPooling2D(pool_size=(2, 2)),

		layers.Conv2D(64, kernel_size=(5, 5), activation="relu", padding="valid"),
		layers.MaxPooling2D(pool_size=(2, 2)),

		layers.Flatten(),

		layers.Dense(512, activation="relu"),
		layers.Dense(output_classes, activation="softmax"),
	])

	model.compile(
		optimizer=optimizer, # type: ignore[arg-type]
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)

	return model
