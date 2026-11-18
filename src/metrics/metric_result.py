from dataclasses import dataclass
from typing import Any


@dataclass
class MetricResult:
    name: str
    value: Any
    unit: str = ""
    higher_is_better: bool = True
    format: str = "{:.3f}"

    def formatted(self) -> str:
        if isinstance(self.value, (int, float)):
            return self.format.format(self.value)
        return str(self.value)

