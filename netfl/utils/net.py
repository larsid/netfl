import os
import socket
from typing import Callable, Any
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib import request
from time import sleep
from random import randint

from netfl.utils.log import log


def serve_file(filename: str, port: int = 9393) -> None:
	class FileServer(SimpleHTTPRequestHandler):
		def do_GET(self):
			if self.path == f"/{filename}":
				self.path = filename
				return super().do_GET()
			self.send_error(404, "File not found")

		def log_message(self, format, *args):
			pass
	
	if not os.path.exists(filename):
		raise FileNotFoundError(f"File '{filename}' does not exist.")
	
	server_address = ("", port)
	httpd = HTTPServer(server_address, FileServer)
	log(f"Serving file {filename} on port {port}")
	httpd.serve_forever()


def download_file(filename: str, address: str, port: int = 9393) -> None:
	url = f"http://{address}:{port}/{filename}"
	file_path = os.path.join(os.getcwd(), filename)
	
	try:
		log(f"Downloading file {filename}")
		request.urlretrieve(url, file_path)
		log(f"File downloaded successfully")
	except Exception as e:
		log(f"Error downloading file: {e}")


def is_server_reachable(address: str, port: int, timeout: int = 5) -> bool:
	try:
		sock = socket.create_connection((address, port), timeout=timeout)
		sock.close()
		return True
	except (socket.timeout, socket.error):
		return False


def wait_server_reachable(address: str, port: int, timeout: int = 5) -> None:
	log(f"Waiting for the server to become reachable on {address}:{port}")
	while not is_server_reachable(address, port,):
		log(f"Server is unreachable, retrying in {timeout} seconds")
		sleep(timeout)

def execute(function: Callable[[], Any], timeout: int = 60, retries: int = 30) -> Any:
	for attempt in range(1, retries + 2):
		try:
			return function()
		except Exception as e:
			log(f"Execution attempt {attempt}/{retries} failed: {e}")
			if attempt <= retries:
				sleep(randint(1, max(1, timeout)))
			else:
				raise
