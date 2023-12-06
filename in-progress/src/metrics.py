from dataclasses import dataclass, field
import numpy as np

@dataclass
class Metric:
    values: list[np.float32] = field(default_factory=list)
    running_total: np.float32 = np.float32(0.0)
    num_updates: np.float32 = np.float32(0.0)
    average: np.float32 = np.float32(0.0)

    def __str__(self):
        return f"Metric(average={self.average:0.4f})"

    def update(self, value: np.float32, batch_size: int) -> None:
        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates
