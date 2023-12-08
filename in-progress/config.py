from dataclasses import dataclass


@dataclass
class Paths:
    log: str
    data: str


@dataclass
class Files:
    test_data: str
    test_labels: str
    train_data: str
    train_labels: str


@dataclass
class Params:
    epochs: int
    batch_size: int
    lr: float


@dataclass
class MNISTConfig:
    paths: Paths
    files: Files
    params: Params
