import torch
from src.dataset import get_test_dataloader, get_train_dataloader
from src.models import LinearNet
from src.runner import Runner, run_epoch
from src.tensorboard import TensorboardExperiment
from src.tracking import Stage
from src.utils import generate_tensorboard_experiment_directory


# Hyperparameters
EPOCHS = 20
LR = 5e-5
BATCH_SIZE = 128

def main():
    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Data
    train_loader = get_train_dataloader(batch_size=BATCH_SIZE)
    test_loader = get_test_dataloader(batch_size=BATCH_SIZE)

    # Create the Runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Experiment Trackers
    log_dir = generate_tensorboard_experiment_directory(root='./runs')
    experiment = TensorboardExperiment(log_dir=log_dir)

    for epoch in range(EPOCHS):
        run_epoch(test_runner, train_runner, experiment, epoch, EPOCHS)

    experiment.flush()


if __name__ == '__main__':
    main()