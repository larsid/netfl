from threading import Event

from netfl.utils.initializer import (
    get_args,
    AppType,
    start_serve_task,
    start_server,
    validate_client_args,
    download_task_file,
    start_client,
)
from netfl.utils.net import wait_host_reachable
from netfl.utils.log import setup_logs


def main():
    args = get_args()

    if args.type == AppType.SERVER:
        setup_logs("server_logs")
        start_serve_task()
        from task import MainTask
        start_server(args, MainTask())
    else:
        validate_client_args(args)
        setup_logs(f"client_{args.client_id}_logs")
        wait_host_reachable(args.server_address, args.server_port)
        download_task_file(args.server_address)
        from task import MainTask
        start_client(args, MainTask())

    Event().wait() 


if __name__ == "__main__":
    main()
