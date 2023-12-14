from torch import Tensor, nn


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28 * 28, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=10),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
