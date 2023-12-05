from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from src.load_data import load_image_data, load_labels
from torch.utils.data import DataLoader, Dataset


class MNIST(Dataset):
    x: torch.Tensor
    y: torch.Tensor

    TRAIN_MAX = 255.0
    MU = 0.1306604762738429
    STD = 0.3081078038564622

    def __init__(self, data: npt.NDArray, targets: npt.NDArray):
        if len(data) != len(targets):
            raise ValueError('data and targets must be the same length. '
                             f'{len(data)} != {len(targets)}')

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._get_x(idx)
        y = self._get_y(idx)
        return x, y

    def _get_x(self, idx: int):
        x = self.data[idx].astype(np.float32)
        x /= self.TRAIN_MAX
        x = (x - self.MU) / self.STD
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        return x

    def _get_y(self, idx: int):
        y = self.targets[idx]
        return torch.tensor(y, dtype=torch.long)


def create_dataloader(
    data_path: Path,
    labels_path: Path,
    batch_size: int,
    shuffle: bool = True
    ) -> DataLoader[Any]:

    data = load_image_data(data_path)
    labels = load_labels(labels_path)

    return DataLoader(
        dataset=MNIST(data, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
