from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import accuracy_score
from src.metrics import Metric
from src.tracking import ExperimentTracker, Stage
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class Runner:
    def __init__(
        self,
        loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.accuracy_metric: Metric = Metric()
        if self.optimizer is not None:
            self.compute_loss = torch.nn.CrossEntropyLoss(reduction="mean")

        self.iteration: int = 0

        self.y_true_batches: list[npt.NDArray] = []
        self.y_pred_batches: list[npt.NDArray] = []

    @property
    def avg_accuracy(self) -> np.float32:
        """Average accuracy."""
        return self.accuracy_metric.average

    def run_loop(self, desc: str, tracker: ExperimentTracker) -> None:
        """Runs the training loop for one epoch.

        Args:
            desc (str): description of the epoch
            tracker (ExperimentTracker): experiment tracker
        """
        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            batch_accuracy = self._run_single_iteration(x, y)
            tracker.add_batch_metric("accuracy", batch_accuracy, self.iteration)

    def _run_single_iteration(self, x: torch.Tensor, y: torch.Tensor) -> np.float32:
        """run a single iteration of the training loop.

        Args:
            x (torch.Tensor): batch data
            y (torch.Tensor): batch labels

        Returns:
            float: average accuracy of the current batch
        """
        self.iteration += 1
        batch_size = x.shape[0]

        # forward step
        prediction = self.model(x)
        # backpropagation
        if self.optimizer:
            self.optimizer.zero_grad()
            loss = self.compute_loss(prediction, y)
            loss.backward()
            self.optimizer.step()

        # Compute Batch Metrics
        y_np = y.detach().numpy()
        y_prediction_np = np.argmax(prediction.detach().numpy(), axis=1)
        batch_accuracy = np.float32(accuracy_score(y_np, y_prediction_np))

        self.accuracy_metric.update(batch_accuracy, batch_size)
        self.y_pred_batches += [y_prediction_np]
        self.y_true_batches += [y_np]

        return batch_accuracy

    def reset(self) -> None:
        """Resets the state of the runner."""
        self.accuracy_metric = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []


def run_epoch(
    test_runner: Runner,
    train_runner: Runner,
    experiment: ExperimentTracker,
    epoch: int,
    epochs_total: int,
):
    experiment.set_stage(Stage.TRAIN)
    train_runner.run_loop("Train batches", experiment)

    experiment.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch)

    experiment.set_stage(Stage.VAL)
    test_runner.run_loop("Validation batches", experiment)

    experiment.add_epoch_metric("accuracy", test_runner.avg_accuracy, epoch)
    experiment.add_epoch_confusion_matrix(
        test_runner.y_true_batches, test_runner.y_pred_batches, epoch
    )

    # Compute Average Epoch Metrics
    summary = ", ".join(
        [
            f"[Epoch: {epoch + 1}/{epochs_total}]",
            f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
            f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
        ]
    )
    print("\n" + summary + "\n")

    train_runner.reset()
    test_runner.reset()
