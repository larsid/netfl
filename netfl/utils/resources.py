from dataclasses import dataclass
from typing import Any


@dataclass
class LinkResources:
	bw: int | None = None
	delay: str | None = None
	loss: int | None = None

	def to_params(self) -> dict[str, Any]:
		return {k: v for k, v in vars(self).items() if v is not None}
