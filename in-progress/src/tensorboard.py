from pathlib import Path
from typing import Tuple

import matplotlib.figure
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.tracking import Stage
from torch.utils.tensorboard.writer import SummaryWriter


class TensorboardExperiment:
    stage: Stage

    def __init__(self, log_dir: str, create=True) -> None:
        self._validate_log_dir(log_dir, create=create)
        self._writer = SummaryWriter(log_dir=log_dir)
        plt.ioff()

    def set_stage(self, stage: Stage) -> None:
        """Set the stage of the experiment."""
        self.stage = stage

    def flush(self):
        """Flushes the experiment."""
        self._writer.flush()

    @staticmethod
    def _validate_log_dir(log_dir, create=True) -> None:
        log_dir = Path(log_dir).resolve()
        if log_dir.exists():
            return
        elif not log_dir.exists() and create:
            log_dir.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def add_batch_metric(self, name: str, value: np.float32, step: int) -> None:
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: np.float32, step: int) -> None:
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_confusion_matrix(
        self, y_true: list[npt.NDArray], y_pred: list[npt.NDArray], step: int
    ) -> None:
        y_true_concat, y_pred_concat = self.collapse_batches(y_true, y_pred)
        fig = self.create_confusion_matrix(y_true_concat, y_pred_concat, step)
        tag = f"{self.stage.name}/epoch/confusion_matrix"
        self._writer.add_figure(tag, fig, step)

    @staticmethod
    def collapse_batches(
        y_true: list[npt.NDArray], y_pred: list[npt.NDArray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.concatenate(y_true), np.concatenate(y_pred)

    def create_confusion_matrix(
        self, y_true: npt.NDArray, y_pred: npt.NDArray, step: int
    ) -> matplotlib.figure.Figure:
        cm = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap="Blues")
        fig: matplotlib.figure.Figure = cm.figure_
        ax: plt.Axes = cm.ax_
        ax.set_title(f"{self.stage.name} Epoch: {step}")
        return fig
