class ServerAlreadyExistsError(RuntimeError):
	def __init__(self) -> None:
		super().__init__("The experiment already has a server.")


class ServerNotCreatedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("The server must be created before creating devices.")


class MaxDevicesReachedError(RuntimeError):
    def __init__(self, max_devices: int) -> None:
        super().__init__(f"The maximum number of devices ({max_devices}) has been reached.")
