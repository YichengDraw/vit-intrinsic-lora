"""
Test script to verify PyTorch 1.8+ compatibility.
Run this to verify the environment, projections, subspace path, and CUDA path.
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pytorch_version():
    """Check PyTorch version."""
    print(f"PyTorch version: {torch.__version__}")
    version_tuple = tuple(int(x) for x in torch.__version__.split('.')[:2])
    print(f"Version tuple: {version_tuple}")
    
    if version_tuple >= (2, 0):
        print("[OK] Using PyTorch 2.0+ (functional_call available)")
    elif version_tuple >= (1, 8):
        print("[OK] Using PyTorch 1.8-1.x (using parameter copy fallback)")
    else:
        print("[FAIL] PyTorch version too old, 1.8+ required")
        return False
    return True


def test_projections():
    """Test all projection types."""
    print("\n" + "=" * 50)
    print("Testing Projections")
    print("=" * 50)
    
    from src.projections import DenseProjection, SparseProjection, FastfoodProjection
    
    input_dim = 10
    output_dim = 100
    
    # Test dense
    print("\nDense projection:")
    proj = DenseProjection(input_dim, output_dim)
    x = torch.randn(input_dim)
    y = proj(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (output_dim,), "Dense projection shape mismatch"
    print("  [OK] Dense projection works")
    
    # Test sparse
    print("\nSparse projection:")
    proj = SparseProjection(input_dim, output_dim)
    y = proj(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (output_dim,), "Sparse projection shape mismatch"
    print("  [OK] Sparse projection works")
    
    # Test fastfood
    print("\nFastfood projection:")
    proj = FastfoodProjection(input_dim, output_dim)
    y = proj(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (output_dim,), "Fastfood projection shape mismatch"
    print("  [OK] Fastfood projection works")
    
    return True


def test_subspace_model():
    """Test SubspaceModel wrapper."""
    print("\n" + "=" * 50)
    print("Testing SubspaceModel")
    print("=" * 50)
    
    from src.models import SubspaceModel
    from src.models.fc import fc_mnist
    
    # Create base model
    base_model = fc_mnist(width=64, depth=2)
    print(f"\nBase model parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    
    # Wrap in subspace
    subspace_dim = 100
    model = SubspaceModel(base_model, subspace_dim, projection_type='dense')
    
    # Test forward pass
    print("\nForward pass test:")
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
    y = model(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 10), "Output shape mismatch"
    print("  [OK] Forward pass works")
    
    # Test backward pass
    print("\nBackward pass test:")
    loss = y.sum()
    loss.backward()
    assert model.theta.grad is not None, "Gradient not computed"
    print(f"  Gradient shape: {model.theta.grad.shape}")
    print("  [OK] Backward pass works")
    
    # Test with different projection types
    for proj_type in ['sparse', 'fastfood']:
        print(f"\nTesting {proj_type} projection:")
        base_model = fc_mnist(width=64, depth=2)
        model = SubspaceModel(base_model, subspace_dim, projection_type=proj_type)
        y = model(x)
        loss = y.sum()
        loss.backward()
        print(f"  [OK] {proj_type} works")
    
    return True


def test_training_loop():
    """Test a simple training loop."""
    print("\n" + "=" * 50)
    print("Testing Training Loop")
    print("=" * 50)
    
    from src.models import SubspaceModel
    from src.models.fc import fc_mnist
    
    # Simple test data
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    
    # Create model
    base_model = fc_mnist(width=64, depth=2)
    model = SubspaceModel(base_model, subspace_dim=50)
    
    # Optimizer on theta only
    optimizer = torch.optim.Adam([model.theta], lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining 5 steps:")
    for step in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")
    
    print("  [OK] Training loop works")
    return True


def test_cuda():
    """Test CUDA if available."""
    print("\n" + "=" * 50)
    print("Testing CUDA")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return True
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    from src.models import SubspaceModel
    from src.models.fc import fc_mnist
    
    device = torch.device('cuda')
    base_model = fc_mnist(width=64, depth=2)
    model = SubspaceModel(base_model, subspace_dim=50)
    model.to(device)
    
    x = torch.randn(8, 1, 28, 28, device=device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    print("  [OK] CUDA works")
    return True


def main():
    print("=" * 60)
    print("Intrinsic Dimension - PyTorch Compatibility Test")
    print("=" * 60)
    
    results = []
    
    results.append(("PyTorch Version", test_pytorch_version()))
    results.append(("Projections", test_projections()))
    results.append(("SubspaceModel", test_subspace_model()))
    results.append(("Training Loop", test_training_loop()))
    results.append(("CUDA", test_cuda()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! The code is compatible with your PyTorch version.")
    else:
        print("Some tests failed. Check the output above for details.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


