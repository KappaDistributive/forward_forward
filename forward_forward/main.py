"""
[The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345)
"""
import os
import random
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator, Optional, Tuple

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


def get_optimizer(optimizer_config: dict, parameters: Iterator[torch.nn.parameter.Parameter]) -> torch.optim.Optimizer:
    config = {k: v for k, v in optimizer_config.items() if k != "name"}
    if optimizer_config["name"].lower() == "sgd":
        return torch.optim.SGD(parameters, **config)
    elif optimizer_config["name"].lower() == "adam":
        return torch.optim.Adam(parameters, **config)
    elif optimizer_config["name"].lower() == "rmsprop":
        return torch.optim.RMSprop(parameters, **config)
    raise ValueError(f"Unsupported optimizer {optimizer_config['name']}")


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
        self.optimizer = get_optimizer(optimizer_config, self.parameters())
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.activation(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train_step(self, x_positive: Tensor, x_negative: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        bus_positive = self.forward(x_positive)
        bus_negative = self.forward(x_negative)

        goodness_positive = bus_positive.pow(2).mean(1)
        goodness_negative = bus_negative.pow(2).mean(1)

        # [Logistic Margin Loss](http://juliaml.github.io/LossFunctions.jl/stable/losses/margin/#LogitMarginLoss-1)
        loss = torch.log(
            1 + torch.exp(torch.cat([-goodness_positive + self.threshold, goodness_negative - self.threshold], 0))
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
        sm_optimizer_config: Optional[dict] = None,
        threshold: float = 0.0,
        use_softmax: bool = False,
        num_classes: int = 10,
        device: Optional[torch.device] = None,
        dtype=None,
    ):
        super().__init__()
        self.layers: list[Layer] = []
        self.device = device if device is not None else torch.device("cpu")
        self.use_softmax = use_softmax
        self.num_classes = num_classes
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
        self.linear = None
        self.cross_entropy_loss = None
        self.cross_entropy_optimizer = None
        if use_softmax:
            assert isinstance(sm_optimizer_config, dict)
            self.linear = torch.nn.Linear(sum(dimensions[2:]), self.num_classes, device=self.device, dtype=dtype)
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
            self.cross_entropy_optimizer = get_optimizer(sm_optimizer_config, self.linear.parameters())

    def train_step(self, x_positive: Tensor, x_negative: Tensor, y_positive: Optional[Tensor] = None) -> float:
        hidden_positive, hidden_negative = torch.flatten(x_positive.to(self.device), 1), torch.flatten(
            x_negative.to(self.device), 1
        )
        losses: list[float] = []
        hidden_states: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            hidden_positive, hidden_negative, loss = layer.train_step(hidden_positive, hidden_negative)
            if index > 0:
                hidden_states.append(hidden_positive)
            losses.append(loss.item())
        if self.use_softmax:
            assert isinstance(self.linear, torch.nn.Linear)
            assert isinstance(self.cross_entropy_loss, torch.nn.CrossEntropyLoss)
            assert isinstance(self.cross_entropy_optimizer, torch.optim.Optimizer)
            assert isinstance(y_positive, Tensor)
            x = self.linear(torch.cat(hidden_states, 1))
            x = torch.softmax(x, 1)
            loss = self.cross_entropy_loss(x, y_positive.to(self.device))
            losses.append(loss.item())
            self.cross_entropy_optimizer.zero_grad()
            loss.backward()
            self.cross_entropy_optimizer.step()

        return sum(losses) / len(losses)

    @torch.no_grad()
    def predict(self, x: Tensor, prefer_goodness: bool = False, keep_image: bool = False) -> Tensor:
        if self.use_softmax and (not prefer_goodness):
            x = x.clone()
            if not keep_image:
                x[:, :, 0, :10] = 0.1
            h = torch.flatten(x, 1)
            hidden_states: list[Tensor] = []
            for index, layer in enumerate(self.layers):
                h = layer(h.to(self.device))
                if index > 0:
                    hidden_states.append(h)
            assert isinstance(self.linear, torch.nn.Linear)
            return torch.softmax(self.linear(torch.cat(hidden_states, 1)), 1)
        else:
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
            return torch.softmax(goodness_per_label_t.detach(), dim=1)


def find_hard_negative_label(images: Tensor, labels: Tensor, net: Net, prefer_goodness: bool = False) -> Tensor:
    with torch.no_grad():
        predictions = net.predict(images, prefer_goodness=prefer_goodness)
        predictions[range(images.shape[0]), labels] = 1e-32
        predictions /= torch.sum(predictions, 0)
    return torch.multinomial(predictions, 1).squeeze(1)


def overlay_label(images: Tensor, labels: list[int]) -> Tensor:
    assert int(images.shape[0]) == len(labels)
    x = images.clone()
    x[:, :, 0, :10] = 0.0
    x[range(x.shape[0]), :, 0, labels] = 1.0
    return x


class TrainingMode(Enum):
    RANDOM_SUPERVISED = 1
    HARD_SUPERVISED = 2
    SELF_SUPERVISED = 3


def create_training_data(
    x: Tensor, y: Tensor, net: Net, data_dict: dict[int, list[Tensor]], mode: TrainingMode
) -> Tuple[Tensor, Tensor]:
    if mode == TrainingMode.RANDOM_SUPERVISED:
        x_positive = overlay_label(x, list(y))
        y_negative = torch.remainder(y + torch.randint(1, 10, y.shape), 10)
        assert not torch.any(y == y_negative)
        x_negative = overlay_label(x, list(y_negative))
    elif mode == TrainingMode.HARD_SUPERVISED:
        x_positive = overlay_label(x, list(y))
        y_negative = find_hard_negative_label(x, y, net)
        x_negative = overlay_label(x, list(y_negative))
    elif mode == TrainingMode.SELF_SUPERVISED:
        x_positive = x.clone()
        y_negative = torch.remainder(y + torch.randint(1, 10, y.shape), 10)
        x_negative_temp: list[Tensor] = []
        for index in range(x.shape[0]):
            x_negative_temp.append(random.choice(data_dict[int(y_negative[index])]))
        mask = create_masks((x.shape[0], x.shape[2], x.shape[3]), 15)
        x_negative = torch.cat(x_negative_temp).unsqueeze(1)
        x_negative = mask * x_positive + (~mask * x_negative)
    else:
        raise NotImplementedError()

    return x_positive, x_negative


def create_masks(size: Tuple[int, int, int], num_blurrs: int):
    mask = torch.rand(size).unsqueeze(1)
    filter_weight = (
        Tensor([[0.0, 1.0 / 6.0, 0.0], [1.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0], [0.0, 1.0 / 6.0, 0.0]])
        .unsqueeze(0)
        .unsqueeze(0)
    )
    for _ in range(num_blurrs):
        mask = torch.conv2d(mask, filter_weight, bias=None, stride=1, padding="same", dilation=1, groups=1)

    # mask = mask.squeeze(1)

    return mask >= 0.5


def calculate_accuracy(predictor: Callable[[Tensor], Tensor], images: list[Tensor], labels: list[Tensor]) -> float:
    assert len(images) == len(labels)
    y_pred: list[Tensor] = []
    for x, y in zip(images, labels):
        y_pred.append(predictor(x).argmax(1))
    y_true = torch.cat(labels).cpu()
    return torch.sum(torch.eq(y_true, torch.cat(y_pred))).item() / y_true.shape[0]


def run() -> None:
    train_loader, test_loader = MNIST_loaders(2**10, 2**14)
    device: torch.device = torch.device("mps")
    print(f"Device: {device}")
    training_mode = TrainingMode.HARD_SUPERVISED
    print(f"Training Mode: {training_mode}")
    keep_image: bool = False
    if training_mode == TrainingMode.SELF_SUPERVISED:
        keep_image = True

    xs_test: list[Tensor] = []
    ys_test: list[Tensor] = []
    if training_mode in (TrainingMode.RANDOM_SUPERVISED, TrainingMode.HARD_SUPERVISED):
        for x, y in test_loader:
            xs_test.append(x)
            ys_test.append(y)

    train_dict: defaultdict[int, list[Tensor]] = defaultdict(list)
    if training_mode == TrainingMode.SELF_SUPERVISED:
        for x, y in train_loader:
            assert x.shape[0] == y.shape[0]
            for index in range(x.shape[0]):
                train_dict[int(y[index])].append(x[index])

    optimizer_config = {
        "name": "Adam",
        "lr": 0.03,
    }
    sm_optimizer_config = {
        "name": "Adam",
        "lr": 0.001,
    }
    net = Net(
        [28 * 28, 2000, 2000, 2000],
        use_softmax=True,
        optimizer_config=optimizer_config,
        sm_optimizer_config=sm_optimizer_config,
        threshold=0.0,
        device=device,
    )
    for epoch in range(500):  # training loop
        timer: float = perf_counter()
        num_images: int = 0
        print(f"Epoch #{epoch+1}")
        total_loss: float = 0.0
        num_steps: int = 0
        for x, y in train_loader:
            num_images += int(x.shape[0])
            x_positive, x_negative = create_training_data(x, y, net, train_dict, training_mode)
            total_loss += net.train_step(x_positive, x_negative, y)
            num_steps += 1
        print(f"Loss: {total_loss / num_steps:.8f}")
        print(f"Training: {num_images / (perf_counter() - timer):_.0f} images/second")

        with torch.no_grad():
            timer = perf_counter()
            accuracy: float = 100.0 * calculate_accuracy(
                lambda x: net.predict(x, keep_image=keep_image).cpu(), xs_test, ys_test
            )
            print(f"Accuracy: {accuracy:.2f}%")

            num_images = sum(x.shape[0] for x in xs_test)
            print(f"Evaluation: {num_images / (perf_counter() - timer):_.0f} images/second")

            if (not training_mode == TrainingMode.SELF_SUPERVISED) and net.use_softmax:
                timer = perf_counter()
                accuracy = 100.0 * calculate_accuracy(
                    lambda x: net.predict(x, prefer_goodness=True, keep_image=keep_image).cpu(), xs_test, ys_test
                )
                print(f"Accuracy (goodness): {accuracy:.2f}%")
                num_images = sum(x.shape[0] for x in xs_test)
                print(f"Evaluation (goodness): {num_images / (perf_counter() - timer):_.0f} images/second")

            print("\nLayer Norms:")
            for layer_index, layer in enumerate(net.layers):
                print(f"{layer_index}: {layer.weight.norm().item():.2f}\t{layer.bias.norm().item():.2f}")
            if net.use_softmax:
                assert isinstance(net.linear, torch.nn.Linear)
                print(f"linear: {net.linear.weight.norm().item():.2f}\t{net.linear.bias.norm().item():.2f}")
            print()


if __name__ == "__main__":
    run()
