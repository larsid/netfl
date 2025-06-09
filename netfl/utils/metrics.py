import time
import psutil
import threading
import statistics
from typing import Callable, Any


class ResourceSampler:
	def __init__(self, interval: float = 0.1) -> None:
		self.interval = interval
		self.cpu_samples: list[float] = []
		self.memory_samples: list[float] = []
		self._sampling = False
		self._thread: threading.Thread | None = None

	def _sample(self) -> None:
		process = psutil.Process()
		while self._sampling:
			try:
				cpu = process.cpu_percent()
				mem = process.memory_info().rss
				self.cpu_samples.append(cpu)
				self.memory_samples.append(mem)
			except Exception:
				pass
			time.sleep(self.interval)

	def start(self) -> None:
		if self._sampling:
			raise RuntimeError("Sampling is already in progress")
		self.cpu_samples.clear()
		self.memory_samples.clear()
		self._sampling = True
		self._thread = threading.Thread(target=self._sample, daemon=True)
		self._thread.start()

	def stop(self) -> tuple[float, float]:
		self._sampling = False
		if self._thread:
			self._thread.join()
		self._thread = None
		cpu_avg_percent = statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0
		memory_avg_mb = statistics.mean(self.memory_samples) / (1024**2) if self.memory_samples else 0.0
		return cpu_avg_percent, memory_avg_mb


def measure_time(func: Callable[[], Any]) -> tuple[Any, float]:
	start = time.perf_counter()
	result = func()
	end = time.perf_counter()
	elapsed = end - start
	return result, elapsed
