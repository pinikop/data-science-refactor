import gzip
from pathlib import Path
from typing import Tuple

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
            raise ValueError(
                "data and targets must be the same length. "
                f"{len(data)} != {len(targets)}"
            )

        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._get_x(idx)
        y = self._get_y(idx)
        return x, y

    def _get_x(self, idx: int) -> torch.Tensor:
        x = self.data[idx].astype(np.float32)
        # scale the data to [0, 1] range
        x /= self.NORMALIZE_FACTOR
        # standardize the data to have zero mean and unit variance
        x = (x - self.MU) / self.STD
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        return x

    def _get_y(self, idx: int) -> torch.Tensor:
        y = self.targets[idx]
        return torch.tensor(y, dtype=torch.long)


def create_dataloader(
    data_path: Path,
    labels_path: Path,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Creates a DataLoader for the MNIST dataset.

    Args:
        data_path (Path): Path to the MNIST data file.
        labels_path (Path): Path to the MNIST labels file.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool, optional): Whether to shuffle the data during training. Defaults to True.

    Returns:
        DataLoader[Tuple[torch.Tensor, torch.Tensor]]: A DataLoader object for training a neural network on the MNIST dataset.
    """

    data = load_data(data_path)
    labels = load_data(labels_path)

    loader = DataLoader(
        dataset=MNIST(data, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

    return loader


def load_data(file_path: Path) -> npt.NDArray:
    """
    Load a MNIST data file and return a NumPy array.

    Args:
        file_path (str): Path to the MNIST data file.

    Returns:
        np.ndarray: A NumPy array containing the data from the file.
    """

    # set the values of offset and shape for images or labels
    print(file_path.as_posix())
    if "images" in file_path.stem:
        offset = 16
        shape = (-1, 28, 28)
    else:
        offset = 8
        shape = (-1,)

    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset)

    # Reshape the data to a 28x28 NumPy array if it is an image file, or a 1-dimensional NumPy array if it is a label file
    data = data.reshape(shape)

    return data
