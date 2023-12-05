from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from src.metrics import Metric
from src.tracking import ExperimentTracker, Stage
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        # loss: Optional[torch.nn.Module] = None
        ) -> None:

        self.run_count = 0
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.accuracy_metric = Metric()
        self.compute_loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []


    @property
    def avg_accuracy(self) -> float:
        return self.accuracy_metric.average


    def run(self, desc: str, experiment: ExperimentTracker):
        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            batch_accuracy = self._run_single(x, y)
            experiment.add_batch_metric('accuracy', batch_accuracy, self.run_count)


    def _run_single(self, x: Any, y: Any):
        self.run_count += 1
        batch_size = x.shape[0]
        prediction = self.model(x)
        loss = self.compute_loss(prediction, y)

        # Compute Batch Training Metrics
        y_np = y.detach().numpy()
        y_prediction_np = np.argmax(prediction.detach().numpy(), axis=1)
        batch_accuracy = accuracy_score(y_np, y_prediction_np)
        self.accuracy_metric.update(batch_accuracy, batch_size)
        self.y_pred_batches += [y_prediction_np]
        self.y_true_batches += [y_np]

        # Reverse-mode AutoDiff (backpropagation)
        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return batch_accuracy

    def reset(self):
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
    train_runner.run('Train batches', experiment)

    experiment.add_epoch_metric('accuracy', train_runner.avg_accuracy, epoch)

    experiment.set_stage(Stage.VAL)
    test_runner.run('Validation batches', experiment)

    experiment.add_epoch_metric('accuracy', test_runner.avg_accuracy, epoch)
    experiment.add_epoch_confusion_matrix(
        test_runner.y_true_batches,
        test_runner.y_pred_batches,
        epoch
        )

    # Compute Average Epoch Metrics
    summary = ', '.join([
        f"[Epoch: {epoch + 1}/{epochs_total}]",
        f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
        f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
    ])
    print('\n' + summary + '\n')

    train_runner.reset()
    test_runner.reset()
