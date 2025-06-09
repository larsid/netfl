import time
from typing import Callable, Any


def measure_time(func: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    elapsed = end - start
    return result, elapsed
