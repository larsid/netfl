import argparse
import socket
import threading
import os
import importlib
from enum import Enum
from dataclasses import dataclass

from netfl.core.task import Task
from netfl.core.server import Server
from netfl.core.client import Client
from netfl.utils.net import serve_file, download_file


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
        raise argparse.ArgumentTypeError(f"Invalid type '{value}'. Choose from: {[e.value for e in AppType]}.")


def valid_port(port) -> int:
    try:
        value = int(port)
        if value < 1 or value > 65535:
            raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("Port must be an integer.")


def valid_ip(ip) -> str:
    try:
        socket.inet_aton(ip)
        return ip
    except socket.error:
        raise argparse.ArgumentTypeError("Invalid IP address format.")


def valid_client_id(value) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Client ID must be a positive integer.")
    return ivalue


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
        raise argparse.ArgumentError(None, f"Missing required arguments for client type: {', '.join(missing_args)}.")


def start_serve_task() -> None:
    http_thread = threading.Thread(
        target=serve_file,
        args=(MAIN_TASK_FILENAME,),
        daemon=True
    )
    http_thread.start()


def start_server(args, task: Task) -> None:
    server = Server(task)
    server.start(server_port=args.server_port)


def download_task_file(server_address: str) -> None:
    download_file(MAIN_TASK_FILENAME, address=server_address)


def validate_task_dir(task_dir: str) -> None:
		if not os.path.isdir(task_dir):
			raise FileNotFoundError(f"Task directory '{task_dir}' does not exist.")

		task_file = os.path.join(task_dir, "task.py")
		if not os.path.isfile(task_file):
			raise FileNotFoundError(f"'task.py' not found in the task directory '{task_dir}'.")


def get_task_dir(task: Task) -> str:
		task_cls = task.__class__
		module_name = task_cls.__module__
		module = importlib.import_module(module_name)

		if hasattr(module, "__file__") and isinstance(module.__file__, str):
			task_dir = os.path.dirname(os.path.abspath(module.__file__))
			validate_task_dir(task_dir)
			return task_dir

		raise FileNotFoundError("Could not determine the task directory.")


def start_client(args, task: Task) -> None:
    client = Client(args.client_id, task)
    client.start(server_address=args.server_address, server_port=args.server_port)
