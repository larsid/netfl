import socket
from enum import Enum


class NetConfigs(Enum):
    SERVER_ADDRESS = "0.0.0.0"
    CONNECTION_TIMEOUT = 1


def is_host_reachable(address: str, port: int, timeout) -> bool:
    try:
        sock = socket.create_connection((address, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, socket.error):
        return False
