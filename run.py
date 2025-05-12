from threading import Event

from netfl import (
    AppType,
    get_args,
    start_serve_task,
    start_server,
    validate_client_args,
    download_task_file,
    start_client,
    wait_host_reachable,
    setup_logs
)


def main():
    args = get_args()

    if args.type == AppType.SERVER:
        setup_logs("server")
        start_serve_task()
        from task import MainTask
        start_server(args, MainTask())
    elif args.type == AppType.CLIENT:
        validate_client_args(args)
        setup_logs()
        wait_host_reachable(args.server_address, args.server_port)
        download_task_file(args.server_address)
        from task import MainTask
        start_client(args, MainTask())
    else:
        raise ValueError(f"Unsupported application type: {args.type}")

    Event().wait() 


if __name__ == "__main__":
    main()
