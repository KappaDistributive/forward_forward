import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

PYTORCH_DATA_DIR = os.getenv("PYTORCH_DATA_DIR")


def MNIST_loaders(train_batch_size: int, test_batch_size: int) -> (DataLoader, DataLoader):
    train_loader = DataLoader(
        MNIST(
            PYTORCH_DATA_DIR,
            train=True,
            download=True,
            transform=ToTensor(),
        ),
        shuffle=True,
        batch_size=train_batch_size,
    )
    test_loader = DataLoader(
        MNIST(
            PYTORCH_DATA_DIR,
            train=False,
            download=True,
            transform=ToTensor(),
        ),
        shuffle=True,
        batch_size=test_batch_size,
    )

    return train_loader, test_loader


def overlay_label(images: Tensor, labels: Tensor) -> Tensor:
    x = images.clone()
    x[:, :, 0, :10] = 0.0
    x[range(x.shape[0]), :, 0, labels] = 1.0

    return x


class Layer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        lr: float,
        threshold: float,
        device: Optional[torch.device] = None,
        dtype=None,
    ) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        self.activation = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-5)
        return self.activation(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_positive: Tensor, x_negative: Tensor) -> (Tensor, Tensor, Tensor):
        bus_positive = self.forward(x_positive)
        bus_negative = self.forward(x_negative)

        goodness_positive = bus_positive.pow(2).mean(1)
        goodness_negative = bus_negative.pow(2).mean(1)

        # [Logistic Margin Loss](http://juliaml.github.io/LossFunctions.jl/stable/losses/margin/#LogitMarginLoss-1)
        loss = torch.log(
            1 + torch.exp(-torch.cat([goodness_positive - self.threshold, -goodness_negative + self.threshold]))
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return bus_positive.detach(), bus_negative.detach(), loss


class Net(nn.Module):
    def __init__(
        self, dimensions: list[int], lr: float, threshold: float, device: Optional[torch.device] = None, dtype=None
    ):
        super().__init__()
        self.layers: list[Layer] = []
        for index in range(len(dimensions) - 1):
            self.layers.append(
                Layer(
                    dimensions[index],
                    dimensions[index + 1],
                    bias=True,
                    lr=lr,
                    threshold=threshold,
                    device=device,
                    dtype=dtype,
                )
            )

    def train(self, x: Tensor, y: Tensor):
        x_positive = overlay_label(x, y)

        y_negative = torch.remainder(y + torch.randint(1, 10, y.shape), 10)
        x_negative = overlay_label(x, y_negative)
        hidden_positive, hidden_negative = torch.flatten(x_positive, 1), torch.flatten(x_negative, 1)
        for index, layer in enumerate(self.layers):
            hidden_positive, hidden_negative, _ = layer.train(hidden_positive, hidden_negative)

    def predict(self, x: Tensor) -> Tensor:
        goodness_per_label = []
        for label in range(10):
            h = torch.flatten(overlay_label(x, label), 1)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(h.pow(2).mean(1))
            goodness_per_label.append(sum(goodness).unsqueeze(1))
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)


if __name__ == "__main__":
    train_loader, test_loader = MNIST_loaders(1000, 1000)
    net = Net([28 * 28, 500, 500, 500, 500, 500, 500], lr=0.03, threshold=5.0)
    for epoch in range(500):
        print(f"Epoch #{epoch+1}")
        for x, y in train_loader:
            net.train(x, y)

        y_true = []
        y_prediction = []
        for x, y in test_loader:
            y_true.append(y)
            y_prediction.append(net.predict(x))
        y_true = torch.cat(y_true)
        y_prediction = torch.cat(y_prediction)
        print(f"Accuracy: {100. * (torch.sum(y_true == y_prediction) / y_true.shape[0]).item():.2f}%")
