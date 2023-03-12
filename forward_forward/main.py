import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor

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


def overlay_label(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    x = images.clone()
    x[:, :, 0, :10] = 0.0
    x[range(x.shape[0]), :, 0, y] = 1.0

    return x


if __name__ == "__main__":
    train_loader, test_loader = MNIST_loaders(10, 10)
    for x, y in train_loader:
        print(x.shape, y.shape)
