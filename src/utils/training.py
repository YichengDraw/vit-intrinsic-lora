"""
Training utilities for intrinsic dimension experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Callable
from tqdm import tqdm
import time


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[object] = None
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        scheduler: Optional learning rate scheduler
        
    Returns:
        (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    
    if scheduler is not None:
        scheduler.step()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on test set.
    
    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_subspace_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    optimizer_type: str = 'adam',
    verbose: bool = True,
    early_stop_patience: int = 0
) -> Dict:
    """
    Full training loop for subspace model.
    
    Args:
        model: SubspaceModel to train
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization weight
        optimizer_type: 'adam' or 'sgd'
        verbose: Whether to print progress
        early_stop_patience: Stop if no improvement for this many epochs (0 = disabled)
        
    Returns:
        Dictionary with training history and final metrics
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize theta (subspace parameters)
    if hasattr(model, 'theta'):
        params = [model.theta]
    else:
        params = model.parameters()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }
    
    best_test_acc = 0.0
    patience_counter = 0
    
    iterator = tqdm(range(1, epochs + 1), desc='Training') if verbose else range(1, epochs + 1)
    
    for epoch in iterator:
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        if verbose:
            iterator.set_postfix({
                'train_acc': f'{train_acc:.2f}%',
                'test_acc': f'{test_acc:.2f}%'
            })
        
        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    # Final results
    results = {
        'history': history,
        'best_test_acc': best_test_acc,
        'final_test_acc': history['test_acc'][-1],
        'final_train_acc': history['train_acc'][-1],
        'total_time': sum(history['epoch_times']),
        'epochs_trained': len(history['train_acc'])
    }
    
    return results


def train_subspace_model_quick(
    base_model_fn: Callable,
    subspace_dim: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    projection_type: str = 'dense',
    epochs: int = 30,
    lr: float = 0.01,
    seed: int = 42
) -> float:
    """
    Quick training function that returns just the best test accuracy.
    
    Useful for grid search over subspace dimensions.
    """
    from ..models import SubspaceModel
    
    base_model = base_model_fn()
    model = SubspaceModel(base_model, subspace_dim, projection_type=projection_type, seed=seed)
    
    results = train_subspace_model(
        model, train_loader, test_loader, device,
        epochs=epochs, lr=lr, verbose=False
    )
    
    return results['best_test_acc']
