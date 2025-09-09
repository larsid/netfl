from dataclasses import dataclass
from typing import Any


COMPUTE_UNIT_PRECISION = 3
COMPUTE_UNIT_ERROR = 1 / 10 ** (COMPUTE_UNIT_PRECISION + 1)


@dataclass
class LinkResources:
	bw: int | None = None
	delay: str | None = None
	loss: int | None = None

	@property
	def params(self) -> dict[str, Any]:
		return {k: v for k, v in vars(self).items() if v is not None}


def calculate_compute_units(clock_host_ghz: float, clock_device_ghz: float) -> float:
	if clock_host_ghz <= 0 or clock_device_ghz <= 0:
		raise ValueError("Clocks must be greater than zero.")
	if clock_device_ghz > clock_host_ghz:
		raise ValueError(f"Device clock cannot exceed host clock.")

	return round(clock_device_ghz / clock_host_ghz, COMPUTE_UNIT_PRECISION)


def cu_with_margin(cu: float) -> float:
	return cu + COMPUTE_UNIT_ERROR
