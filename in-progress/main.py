import torch
from src.dataset import get_test_dataloader, get_train_dataloader
from src.models import LinearNet
from src.runner import Runner
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
            f"[Epoch: {epoch + 1}/{EPOCHS}]",
            f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
            f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
        ])
        print('\n' + summary + '\n')

        train_runner.reset()
        test_runner.reset()

    experiment.flush()


if __name__ == '__main__':
    main()