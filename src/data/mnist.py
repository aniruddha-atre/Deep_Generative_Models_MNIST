# src/data/mnist.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dcgan_dataloader(batch_size: int = 64):
    """
    MNIST dataloader for DCGAN: images normalized to [-1, 1].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                    # [0,1]
        transforms.Normalize((0.5,), (0.5,)),     # -> [-1,1]
    ])

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_loader


def get_dcvae_dataloaders(batch_size: int = 64):
    """
    MNIST dataloaders for DCVAE: images in [0,1], shape (B,1,28,28).
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader
