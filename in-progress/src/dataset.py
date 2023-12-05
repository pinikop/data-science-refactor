from pathlib import Path
from typing import Any

import numpy as np
import torch
from src.load_data import load_image_data, load_labels
from torch.utils.data import DataLoader, Dataset


class MNIST(Dataset):
    idx: int  # requested data index
    x: torch.Tensor
    y: torch.Tensor

    TRAIN_MAX = 255.0
    TRAIN_NORMALIZED_MEAN = 0.1306604762738429
    TRAIN_NORMALIZED_STDEV = 0.3081078038564622

    def __init__(self, data: np.ndarray, targets: np.ndarray):
        if len(data) != len(targets):
            raise ValueError('data and targets must be the same length. '
                             f'{len(data)} != {len(targets)}')

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.get_x(idx)
        y = self.get_y(idx)
        return x, y

    def get_x(self, idx: int):
        self.idx = idx
        self.preprocess_x()
        return self.x

    def preprocess_x(self):
        self.x = self.data[self.idx].copy().astype(np.float64)
        self.x /= self.TRAIN_MAX
        self.x -= self.TRAIN_NORMALIZED_MEAN
        self.x /= self.TRAIN_NORMALIZED_STDEV
        self.x = self.x.astype(np.float32)
        self.x = torch.from_numpy(self.x)
        self.x = self.x.unsqueeze(0)

    def get_y(self, idx: int):
        self.idx = idx
        self.preprocess_y()
        return self.y

    def preprocess_y(self):
        self.y = self.targets[self.idx]
        self.y = torch.tensor(self.y, dtype=torch.long)


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
