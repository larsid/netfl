from fogfl.utils.initializer import (
    get_args,
    AppType,
    start_serve_task,
    start_server,
    validate_client_args,
    MAIN_TASK_FILENAME,
    start_client,
)
from fogfl.utils.net import (
    wait_until_host_reachable,
    download_file,
)
from fogfl.utils.log import setup_logfile

def main():
    args = get_args()

    if args.type == AppType.SERVER:
        start_serve_task()
        from task import MainTask
        setup_logfile("server_logs")
        start_server(args, MainTask())
    else:
        validate_client_args(args)
        wait_until_host_reachable(args.server_address, args.server_port)
        download_file(MAIN_TASK_FILENAME, address=args.server_address)
        from task import MainTask
        start_client(args, MainTask())

if __name__ == "__main__":
    main()
