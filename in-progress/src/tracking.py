from enum import Enum, auto
from typing import Protocol

import numpy as np
import numpy.typing as npt


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class ExperimentTracker(Protocol):
    def add_batch_metric(self, name: str, value: np.float32, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: np.float32, step: int):
        """Implements logging a epoch-level metric."""

    def add_epoch_confusion_matrix(
        self, y_true: list[npt.NDArray], y_pred: list[npt.NDArray], step: int
    ):
        """Implements logging a confusion matrix at epoch-level."""

    def set_stage(self, stage: Stage):
        """Sets the stage of the experiment."""

    def flush(self):
        """Flushes the experiment."""
