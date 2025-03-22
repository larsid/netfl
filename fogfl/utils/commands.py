import base64
import gzip
from enum import Enum

from fogbed import Container

from fogfl.utils.initializer import Args


class DefaultFiles(Enum):
    APP_PATH = "/app/fogfl"
    MAIN_TASK = "task.py"
    TASK_RUNNER = "run.py"


def transfer_task(task_path: str, container: Container) -> None:
    print(f"Transferring task to {container.name}")
    try:
        with open(task_path, "rb") as file:
            data = file.read()
        encoded_data = base64.b64encode(gzip.compress(data)).decode()
        
        result = container.cmd(
            f"mkdir -p {DefaultFiles.APP_PATH.value} && "
            f"echo '{encoded_data}' | base64 -d | "
            f"gunzip > {DefaultFiles.MAIN_TASK.value}"
        )
        print(result)
        print(f"Task transferred successfully to {container.name}")
    except FileNotFoundError:
        print(f"Error: {task_path} not found")
    except Exception as e:
        print(f"An error occurred during file transfer: {e}")


def run_task(args: Args, background: bool, container: Container) -> None:
    print(f"Starting task on {container.name} (background={background})")
    result = container.cmd(
        f"python3 {DefaultFiles.TASK_RUNNER.value} --type={args.type} "
        f"--server_port={args.server_port} "
        f"--server_address='{args.server_address}' "
        f"--client_id={args.client_id} "
        f"--lazy_client={args.lazy_client} "
        f"{'&' if background else ''}"
    )
    print(result)
    print(f"Task started successfully on {container.name}")
