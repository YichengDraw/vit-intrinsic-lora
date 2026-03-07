"""
Data loading utilities for intrinsic dimension experiments.

Supports:
- MNIST (standard, shuffled pixels, shuffled labels)
- CIFAR-10
- CIFAR-100
- Flowers-102
- ImageNet
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np
import os
import platform


def get_num_workers(requested: int = 4) -> int:
    """
    Get appropriate number of workers for DataLoader.
    On Windows, multiprocessing with DataLoader often causes issues,
    so we default to 0 (main process only).
    """
    if platform.system() == 'Windows':
        return 0  # Avoid multiprocessing issues on Windows
    return requested


# =============================================================================
# MNIST
# =============================================================================

def get_mnist_loaders(
    data_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get standard MNIST data loaders.
    
    Returns:
        (train_loader, test_loader)
    """
    num_workers = get_num_workers(num_workers)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


class ShuffledPixelMNIST(Dataset):
    """MNIST with randomly permuted pixels (same permutation for all images)."""
    
    def __init__(self, mnist_dataset: Dataset, seed: int = 42):
        self.dataset = mnist_dataset
        
        # Generate fixed random permutation
        rng = np.random.RandomState(seed)
        self.perm = torch.from_numpy(rng.permutation(28 * 28))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Flatten, permute, reshape
        img_flat = img.view(-1)
        img_shuffled = img_flat[self.perm].view(1, 28, 28)
        return img_shuffled, label


def get_mnist_shuffled_pixel_loaders(
    data_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST loaders with shuffled pixels.
    
    Paper finding: LeNet loses its advantage over FC on shuffled pixels
    because spatial structure is destroyed.
    """
    num_workers = get_num_workers(num_workers)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    train_shuffled = ShuffledPixelMNIST(train_dataset, seed=seed)
    test_shuffled = ShuffledPixelMNIST(test_dataset, seed=seed)
    
    train_loader = DataLoader(
        train_shuffled, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_shuffled, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


class ShuffledLabelMNIST(Dataset):
    """MNIST with randomly shuffled labels (for measuring memorization)."""
    
    def __init__(self, mnist_dataset: Dataset, shuffle_fraction: float = 1.0, seed: int = 42):
        self.dataset = mnist_dataset
        self.shuffle_fraction = shuffle_fraction
        
        # Shuffle labels
        rng = np.random.RandomState(seed)
        n = len(mnist_dataset)
        n_shuffle = int(n * shuffle_fraction)
        
        # Get all labels
        if hasattr(mnist_dataset, 'targets'):
            self.labels = mnist_dataset.targets.clone()
        else:
            self.labels = torch.tensor([mnist_dataset[i][1] for i in range(n)])
        
        # Randomly permute a fraction of labels
        shuffle_indices = rng.choice(n, n_shuffle, replace=False)
        shuffled_labels = self.labels[shuffle_indices].numpy()
        rng.shuffle(shuffled_labels)
        self.labels[shuffle_indices] = torch.from_numpy(shuffled_labels)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # Ignore original label
        return img, self.labels[idx].item()


def get_mnist_shuffled_label_loaders(
    data_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle_fraction: float = 1.0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST loaders with shuffled labels.
    
    Paper finding: 
    - 100% shuffled: d_int90 ≈ 190,000 (pure memorization)
    - 50% shuffled: d_int90 ≈ 130,000
    - 10% shuffled: d_int90 ≈ 90,000
    - 0% shuffled (original): d_int90 ≈ 750
    """
    num_workers = get_num_workers(num_workers)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    train_shuffled = ShuffledLabelMNIST(train_dataset, shuffle_fraction, seed)
    # Test set keeps original labels for measuring generalization
    
    train_loader = DataLoader(
        train_shuffled, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


# =============================================================================
# CIFAR-10
# =============================================================================

def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders.
    
    Args:
        augment: Whether to use data augmentation for training
    """
    num_workers = get_num_workers(num_workers)
    
    # Normalize with CIFAR-10 statistics
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


# =============================================================================
# CIFAR-100 / Flowers-102 (ViT-friendly 224x224 pipeline)
# =============================================================================

def _build_imagenet_style_transforms(
    image_size: int = 224,
    augment: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build ImageNet normalization transforms for transfer-learning setups.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.67, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, test_transform


def get_cifar100_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 loaders with ViT-friendly preprocessing.
    """
    num_workers = get_num_workers(num_workers)
    train_transform, test_transform = _build_imagenet_style_transforms(
        image_size=image_size,
        augment=augment
    )
    
    train_dataset = datasets.CIFAR100(
        data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        data_dir, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def get_flowers102_loaders(
    data_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 4,
    augment: bool = True,
    image_size: int = 224,
    combine_train_val: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get Oxford Flowers-102 loaders.

    By default, train + val splits are merged for training and test is used for evaluation.
    """
    num_workers = get_num_workers(num_workers)
    train_transform, test_transform = _build_imagenet_style_transforms(
        image_size=image_size,
        augment=augment
    )
    
    train_dataset = datasets.Flowers102(
        root=data_dir, split='train', download=True, transform=train_transform
    )
    val_dataset = datasets.Flowers102(
        root=data_dir, split='val', download=True, transform=train_transform
    )
    test_dataset = datasets.Flowers102(
        root=data_dir, split='test', download=True, transform=test_transform
    )
    
    if combine_train_val:
        train_dataset = ConcatDataset([train_dataset, val_dataset])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


# =============================================================================
# ImageNet
# =============================================================================

def get_imagenet_loaders(
    data_dir: str = './data/imagenet',
    batch_size: int = 256,
    num_workers: int = 8,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet data loaders.
    
    Note: ImageNet must be downloaded separately and placed in data_dir
    with 'train' and 'val' subdirectories.
    """
    num_workers = get_num_workers(num_workers)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"ImageNet not found at {data_dir}. "
            "Please download ImageNet and organize as data_dir/train and data_dir/val"
        )
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
