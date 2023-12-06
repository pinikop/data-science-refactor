import gzip
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset


class MNIST(Dataset):
    x: torch.Tensor
    y: torch.Tensor

    NORMALIZE_FACTOR = 255.0
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
        x /= self.NORMALIZE_FACTOR
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

    data = load_mnist_file(data_path)
    labels = load_mnist_file(labels_path)

    return DataLoader(
        dataset=MNIST(data, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def load_mnist_file(file_name: Path) -> npt.NDArray:
    # Load the specified file
    if 'images' in file_name.stem:
        offset = 16
        shape = (-1, 28, 28)
    else:
        offset = 8
        shape = (-1,)

    with gzip.open(file_name, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset)

    # Reshape the data to a 28x28 NumPy array if it is an image file, or a 1-dimensional NumPy array if it is a label file
    data = data.reshape(shape)

    return data
