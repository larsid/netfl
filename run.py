from os import getenv

from netfl.utils.log import setup_log_file
from netfl.utils.net import wait_server_reachable
from netfl.utils.initializer import (
	EXPERIMENT_ENV_VAR,
	AppType,
	get_args,
	serve_task_file,
	start_server,
	validate_client_args,
	download_task_file,
	start_client,
)


def main():
	args = get_args()

	if args.type == AppType.SERVER:
		setup_log_file(getenv(EXPERIMENT_ENV_VAR, ""))
		serve_task_file()
		from task import MainTask
		start_server(args, MainTask())
	elif args.type == AppType.CLIENT:
		validate_client_args(args)
		wait_server_reachable(args.server_address, args.server_port)
		download_task_file(args.server_address)
		from task import MainTask
		start_client(args, MainTask())
	else:
		raise ValueError(f"Unsupported application type: {args.type}.") 


if __name__ == "__main__":
	main()
