"""
[The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345)
"""
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

PYTORCH_DATA_DIR_: Optional[str] = os.getenv("PYTORCH_DATA_DIR")
if PYTORCH_DATA_DIR_ is None:
    print(
        "Please set the enviroment variable `PYTORCH_DATA_DIR`. "
        "It serves as a data dir for training data and models."
    )
    sys.exit(1)
else:
    PYTORCH_DATA_DIR: Path = Path(PYTORCH_DATA_DIR_)
    PYTORCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
del PYTORCH_DATA_DIR_


def MNIST_loaders(train_batch_size: int, test_batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transformation = Compose(
        (
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        )
    )
    train_loader = DataLoader(
        MNIST(
            str(PYTORCH_DATA_DIR),
            train=True,
            download=True,
            transform=transformation,
        ),
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=8,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        MNIST(
            str(PYTORCH_DATA_DIR),
            train=False,
            download=True,
            transform=transformation,
        ),
        shuffle=True,
        batch_size=test_batch_size,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, test_loader


class Layer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        optimizer_config: dict,
        threshold: float,
        device: Optional[torch.device] = None,
        dtype=None,
    ) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        self.activation = nn.ReLU()
        self.optimizer = Layer._get_optimizer(optimizer_config, self.parameters())
        self.threshold = threshold

    @staticmethod
    def _get_optimizer(
        optimizer_config: dict, parameters: Iterator[torch.nn.parameter.Parameter]
    ) -> torch.optim.Optimizer:
        config = {k: v for k, v in optimizer_config.items() if k != "name"}
        if optimizer_config["name"].lower() == "sgd":
            return torch.optim.SGD(parameters, **config)
        elif optimizer_config["name"].lower() == "adam":
            return torch.optim.Adam(parameters, **config)
        elif optimizer_config["name"].lower() == "rmsprop":
            return torch.optim.RMSprop(parameters, **config)
        raise ValueError(f"Unsupported optimizer {optimizer_config['name']}")

    def forward(self, x: Tensor) -> Tensor:
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-5)
        return self.activation(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train_step(self, x_positive: Tensor, x_negative: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
        self,
        dimensions: list[int],
        optimizer_config: dict,
        threshold: float,
        device: Optional[torch.device] = None,
        dtype=None,
    ):
        super().__init__()
        self.layers: list[Layer] = []
        self.device = device if device is not None else torch.device("cpu")
        for index in range(len(dimensions) - 1):
            self.layers.append(
                Layer(
                    dimensions[index],
                    dimensions[index + 1],
                    bias=True,
                    optimizer_config=optimizer_config,
                    threshold=threshold,
                    device=device,
                    dtype=dtype,
                )
            )

    def train_step(self, x_positive: Tensor, x_negative: Tensor) -> float:
        hidden_positive, hidden_negative = torch.flatten(x_positive.to(self.device), 1), torch.flatten(
            x_negative.to(self.device), 1
        )
        losses: list[float] = []
        for index, layer in enumerate(self.layers):
            hidden_positive, hidden_negative, loss = layer.train_step(hidden_positive, hidden_negative)
            losses.append(loss.item())

        return sum(losses) / len(losses)

    def predict(self, x: Tensor) -> Tensor:
        goodness_per_label: list[Tensor] = []
        for label in range(10):
            h = torch.flatten(overlay_label(x, [label] * int(x.shape[0])), 1)
            goodness: list[Tensor] = []
            for index, layer in enumerate(self.layers):
                h = layer(h.to(self.device))
                if index > 0:  # use all but the first hidden layer as in the paper
                    goodness.append(h.pow(2).mean(1))
            goodness_per_label.append(Tensor(sum(goodness)).unsqueeze(1))
        goodness_per_label_t: Tensor = torch.cat(goodness_per_label, 1)
        return goodness_per_label_t.argmax(1)


def overlay_label(images: Tensor, labels: list[int]) -> Tensor:
    assert int(images.shape[0]) == len(labels)
    x = images.clone()
    x[:, :, 0, :10] = 0.0
    x[range(x.shape[0]), :, 0, labels] = 1.0

    return x


class TrainingMode(Enum):
    RANDOM_SUPERVISED = 1


def create_training_data(x: Tensor, y: Tensor, mode: TrainingMode) -> Tuple[Tensor, Tensor]:
    if mode == TrainingMode.RANDOM_SUPERVISED:
        x_positive = overlay_label(x, list(y))
        y_negative = torch.remainder(y + torch.randint(1, 10, y.shape), 10)
        assert not torch.any(y == y_negative)
        x_negative = overlay_label(x, list(y_negative))

    return x_positive, x_negative


if __name__ == "__main__":
    train_loader, test_loader = MNIST_loaders(1_000, 10_000)
    device: torch.device = torch.device("mps")
    print(f"Device: {device}")
    optmizer_config = {
        "name": "Adam",
        "lr": 0.03,
    }
    net = Net([28 * 28, 2000, 2000, 2000, 2000], optimizer_config=optmizer_config, threshold=2.0, device=device)
    for epoch in range(500):
        print(f"Epoch #{epoch+1}")
        total_loss: float = 0.0
        num_steps: int = 0
        for x, y in train_loader:
            x_positive, x_negative = create_training_data(x, y, TrainingMode.RANDOM_SUPERVISED)
            total_loss += net.train_step(x_positive, x_negative)
            num_steps += 1
        print(total_loss / num_steps)

        with torch.no_grad():
            y_true_l: list[Tensor] = []
            y_prediction_l: list[Tensor] = []
            for x, y in test_loader:
                y_true_l.append(y)
                y_prediction_l.append(net.predict(x).cpu())
            y_true: Tensor = torch.cat(y_true_l)
            y_prediction: Tensor = torch.cat(y_prediction_l)
            print(f"Accuracy: {100. * (torch.sum(torch.eq(y_true, y_prediction)) / y_true.shape[0]).item():.2f}%")
            print()
