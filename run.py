from os import getenv, getcwd

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
    validate_task_dir,
)


def load_task():
    from task import FLTask

    return FLTask()


def main():
    args = get_args()
    current_dir = getcwd()

    if args.type == AppType.SERVER:
        validate_task_dir(current_dir)
        task = load_task()
        setup_log_file(getenv(EXPERIMENT_ENV_VAR, ""))
        serve_task_file()
        start_server(args, task)
    elif args.type == AppType.CLIENT:
        validate_client_args(args)
        wait_server_reachable(args.server_address, args.server_port)
        download_task_file(args.server_address)
        task = load_task()
        start_client(args, task)
    else:
        raise ValueError(f"Unsupported application type: {args.type}.")


if __name__ == "__main__":
    main()
