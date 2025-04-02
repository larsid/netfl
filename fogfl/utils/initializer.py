import argparse
import socket
import threading
from enum import Enum
from dataclasses import dataclass

from fogfl.core.task import Task
from fogfl.core.server import Server
from fogfl.core.client import Client
from fogfl.utils.net import serve_file, download_file, wait_until_host_reachable


MAIN_TASK_FILENAME = "task.py"


class AppType(Enum):
    CLIENT = "client"
    SERVER = "server"


@dataclass
class Args:
    type: AppType
    server_port: int
    server_address: str | None
    client_id: int | None


def valid_app_type(value: str) -> AppType:
    try:
        return AppType(value.lower())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid type '{value}'. Choose from: {[e.value for e in AppType]}")


def valid_port(port) -> int:
    try:
        value = int(port)
        if value < 1 or value > 65535:
            raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("Port must be an integer")


def valid_ip(ip) -> str:
    try:
        socket.inet_aton(ip)
        return ip
    except socket.error:
        raise argparse.ArgumentTypeError("Invalid IP address format")


def valid_client_id(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Client ID must be a positive integer")
    return ivalue


def valid_bool(value: str) -> bool:
    value_lower = value.lower()
    if value_lower in {"true", "false"}:
        return value_lower == "true"
    raise argparse.ArgumentTypeError("Boolean value expected: 'true' or 'false'")


def get_args():
    parser = argparse.ArgumentParser(description="Configure application settings")
    parser.add_argument("--type", type=valid_app_type, required=True, help="Type of application: client or server")
    parser.add_argument("--server_port", type=valid_port, required=True, help="Server port number (1-65535)")
    parser.add_argument("--server_address", type=valid_ip, help="Server IP address (required for client type)")
    parser.add_argument("--client_id", type=valid_client_id, help="Client ID (required for client type)")
    return parser.parse_args()


def validate_client_args(args) -> None:
    missing_args = []
    if args.server_address is None:
        missing_args.append("--server_address")
    if args.client_id is None:
        missing_args.append("--client_id")

    if missing_args:
        raise argparse.ArgumentError(None, f"Missing required arguments for client type: {', '.join(missing_args)}")


def start_serve_task():
    http_thread = threading.Thread(
        target=serve_file,
        args=(MAIN_TASK_FILENAME,),
        daemon=True
    )
    http_thread.start()


def start_server(args, task: Task) -> None:
    server = Server(task)
    server.start(server_port=args.server_port)


def start_client(args, task: Task) -> None:
    client = Client(args.client_id, task)
    client.start(server_address=args.server_address, server_port=args.server_port)
