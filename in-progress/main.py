from pathlib import Path

import torch

from src.dataset import create_dataloader
from src.models import LinearNet
from src.runner import Runner, run_epoch
from src.tensorboard import TensorboardExperiment
from src.utils import generate_tensorboard_experiment_directory

# Hyperparameters
EPOCHS = 20
LR = 5e-5
BATCH_SIZE = 128
ROOT_PATH = Path(__file__).parent
LOG_PATH =  ROOT_PATH / 'runs'

# Data configuration
DATA_DIR = ROOT_PATH / 'data'
TEST_DATA = DATA_DIR / "t10k-images-idx3-ubyte.gz"
TEST_LABELS = DATA_DIR / "t10k-labels-idx1-ubyte.gz"
TRAIN_DATA = DATA_DIR / "train-images-idx3-ubyte.gz"
TRAIN_LABELS = DATA_DIR / "train-labels-idx1-ubyte.gz"


def main():
    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Data
    train_loader = create_dataloader(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE)
    test_loader = create_dataloader(TEST_DATA, TEST_LABELS, batch_size=BATCH_SIZE, shuffle=False)

    # Create the Runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Experiment Trackers
    log_dir = generate_tensorboard_experiment_directory(root=LOG_PATH)
    experiment = TensorboardExperiment(log_dir=log_dir)

    for epoch in range(EPOCHS):
        run_epoch(test_runner, train_runner, experiment, epoch, EPOCHS)

    experiment.flush()


if __name__ == '__main__':
    main()