import time
import threading
import os


class ResourceSampler:
	def __init__(self, interval: float = 2.0) -> None:
		self._interval = interval
		self._sampling = False
		self._thread: threading.Thread | None = None
		self._lock = threading.Lock()
		self._memory_sum = 0.0
		self._memory_count = 0
		self._start_time = 0.0
		self._start_cpu = 0

	def start(self) -> None:
		with self._lock:
			if self._sampling:
				raise RuntimeError("Resource sampling already in progress.")

			self._sampling = True
			self._memory_sum = 0.0
			self._memory_count = 0
			self._start_time = time.perf_counter()
			self._start_cpu = self._read_cpu_usage() or 0

			self._thread = threading.Thread(target=self._run, daemon=True)
			self._thread.start()

	def stop(self) -> tuple[float, float]:
		with self._lock:
			self._sampling = False

		if self._thread:
			self._thread.join()
			self._thread = None

		end_cpu = self._read_cpu_usage() or self._start_cpu
		elapsed = time.perf_counter() - self._start_time
		cpus = self._get_cpu_limit()

		cpu_delta_sec = (end_cpu - self._start_cpu) / 1_000_000
		cpu_avg_percent = (cpu_delta_sec / (elapsed * cpus)) * 100 if elapsed > 0 else 0.0
		memory_avg_mb = (self._memory_sum / self._memory_count) / (1024 ** 2) if self._memory_count > 0 else 0.0

		return round(cpu_avg_percent, 6), round(memory_avg_mb, 6)

	def _run(self) -> None:
		while True:
			with self._lock:
				if not self._sampling:
					break

			mem = self._read_memory_usage()
			if mem is not None:
				self._memory_sum += mem
				self._memory_count += 1

			time.sleep(self._interval)

	def _read_cpu_usage(self) -> int | None:
		try:
			with open("/sys/fs/cgroup/cpu.stat", "r") as f:
				for line in f:
					if line.startswith("usage_usec"):
						return int(line.split()[1])
		except Exception:
			return None

	def _read_memory_usage(self) -> int | None:
		try:
			with open("/sys/fs/cgroup/memory.current", "rb") as f:
				total = int(f.read().strip())

			file_cache = 0
			with open("/sys/fs/cgroup/memory.stat", "rb") as f:
				for line in f:
					if line.startswith(b"file "):
						file_cache = int(line[5:].strip())
						break

			return total - file_cache
		except Exception:
			return None

	def _get_cpu_limit(self) -> float:
		try:
			with open("/sys/fs/cgroup/cpu.max", "r") as f:
				quota_str, period_str = f.read().strip().split()
				if quota_str in {"max", "0"} or period_str == "0":
					return self._cpu_count_fallback()
				return int(quota_str) / int(period_str)
		except Exception:
			return self._cpu_count_fallback()

	@staticmethod
	def _cpu_count_fallback() -> int:
		try:
			return os.cpu_count() or 1
		except Exception:
			return 1
