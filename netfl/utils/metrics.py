import time
import threading
import psutil
from pathlib import Path

from netfl.utils.log import log


class ResourceSampler:
    def __init__(self, interval: float = 0.1) -> None:
        self._interval = interval
        self._sampling = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._cpu_sum = 0.0
        self._cpu_count = 0
        self._memory_sum = 0.0
        self._memory_count = 0

        self._process = psutil.Process()
        self._cpu_limit = self._get_cpu_limit()
        self._use_cgroup_memory = self._detect_cgroup_memory()

        self._cgroup_memory_file = None
        self._cgroup_memory_failed = False
        if self._use_cgroup_memory:
            self._open_cgroup_memory_file()

    def start(self) -> None:
        with self._lock:
            if self._sampling:
                raise RuntimeError("Resource sampling already in progress.")

            self._sampling = True
            self._cpu_sum = 0.0
            self._cpu_count = 0
            self._memory_sum = 0.0
            self._memory_count = 0
            self._cgroup_memory_failed = False

            if self._use_cgroup_memory and not self._cgroup_memory_file:
                self._open_cgroup_memory_file()

        self._prime_cpu_tracking()

        with self._lock:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> tuple[float, float]:
        with self._lock:
            self._sampling = False

        if self._thread:
            self._thread.join()
            self._thread = None

        cpu_avg_percent = (
            self._cpu_sum / self._cpu_count if self._cpu_count > 0 else 0.0
        )

        memory_avg_mb = (
            (self._memory_sum / self._memory_count) / (1024**2)
            if self._memory_count > 0
            else 0.0
        )

        return round(cpu_avg_percent, 2), round(memory_avg_mb, 2)

    def __del__(self) -> None:
        if hasattr(self, "_cgroup_memory_file") and self._cgroup_memory_file:
            try:
                self._cgroup_memory_file.close()
            except:
                pass

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._sampling:
                    break
                use_cgroup = self._use_cgroup_memory and not self._cgroup_memory_failed

            try:
                cpu_percent_raw = 0.0
                memory_bytes = 0

                if use_cgroup:
                    memory_bytes = self._read_cgroup_memory_fast()
                    with self._lock:
                        if self._cgroup_memory_failed:
                            use_cgroup = False

                children = []
                if not use_cgroup or memory_bytes == 0:
                    try:
                        children = self._process.children(recursive=True)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                    if memory_bytes == 0:
                        memory_bytes = self._collect_rss_memory(children)

                try:
                    cpu_percent_raw += self._process.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                if not children and use_cgroup:
                    try:
                        children = self._process.children(recursive=True)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                for child in children:
                    try:
                        cpu_val = child.cpu_percent(interval=None)
                        cpu_percent_raw += cpu_val
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                cpu_percent_normalized = (
                    (cpu_percent_raw / self._cpu_limit) if self._cpu_limit > 0 else 0.0
                )

                cpu_percent_normalized = max(0.0, min(100.0, cpu_percent_normalized))

                with self._lock:
                    self._cpu_sum += cpu_percent_normalized
                    self._cpu_count += 1
                    self._memory_sum += memory_bytes
                    self._memory_count += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                log(f"Resource Sampler error: {e}")

            time.sleep(self._interval)

    def _collect_rss_memory(self, children: list) -> int:
        memory_bytes = 0

        try:
            memory_bytes += self._process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        for child in children:
            try:
                memory_bytes += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return memory_bytes

    def _prime_cpu_tracking(self) -> None:
        try:
            self._process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        try:
            for child in self._process.children(recursive=True):
                try:
                    child.cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _get_cpu_limit(self) -> float:
        try:
            if Path("/sys/fs/cgroup/cpu.max").exists():
                with open("/sys/fs/cgroup/cpu.max", "r") as f:
                    quota_str, period_str = f.read().strip().split()

                    if quota_str == "max":
                        return float(psutil.cpu_count() or 1)

                    quota = int(quota_str)
                    period = int(period_str)

                    if period == 0:
                        return float(psutil.cpu_count() or 1)

                    return float(quota) / float(period)

            elif (
                Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").exists()
                and Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").exists()
            ):
                with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
                    quota = int(f.read().strip())

                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
                    period = int(f.read().strip())

                if quota == -1:
                    return float(psutil.cpu_count() or 1)

                if period == 0:
                    return float(psutil.cpu_count() or 1)

                return float(quota) / float(period)

            else:
                return float(psutil.cpu_count() or 1)

        except (FileNotFoundError, ValueError, IOError):
            return float(psutil.cpu_count() or 1)

    def _detect_cgroup_memory(self) -> bool:
        cgroup_v2_memory = Path("/sys/fs/cgroup/memory.current")
        return cgroup_v2_memory.exists()

    def _open_cgroup_memory_file(self) -> None:
        try:
            cgroup_v2_path = Path("/sys/fs/cgroup/memory.current")
            if cgroup_v2_path.exists():
                self._cgroup_memory_file = open(cgroup_v2_path, "r")
                return

            cgroup_v1_path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            if cgroup_v1_path.exists():
                self._cgroup_memory_file = open(cgroup_v1_path, "r")
                return
        except (FileNotFoundError, IOError, PermissionError):
            pass

    def _read_cgroup_memory_fast(self) -> int:
        if self._cgroup_memory_file:
            try:
                self._cgroup_memory_file.seek(0)
                value = self._cgroup_memory_file.read().strip()
                return int(value)
            except (ValueError, IOError, OSError):
                with self._lock:
                    self._cgroup_memory_failed = True
                try:
                    self._cgroup_memory_file.close()
                except:
                    pass
                self._cgroup_memory_file = None

        return 0
