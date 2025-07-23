#!/usr/bin/env python3
"""Check GPU availability and status."""

import torch
import subprocess
import sys

def check_gpu_status():
    """Check comprehensive GPU status."""
    print("🔍 GPU Status Check")
    print("=" * 50)
    
    # PyTorch CUDA information
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Major/Minor: {props.major}.{props.minor}")
            print(f"  Multi Processor Count: {props.multi_processor_count}")
    else:
        print("GPU Name: None (CUDA not available)")
    
    print("\n🎮 GPU Memory Test")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # Test GPU memory allocation
        try:
            device = torch.device('cuda:0')
            print(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.3f} GB")
            print(f"Current GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.3f} GB")
            
            # Test tensor operations
            test_tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"✅ GPU tensor operation successful: {result.shape}")
            print(f"Memory after test: {torch.cuda.memory_allocated(0) / 1024**3:.3f} GB")
            
            # Clean up
            del test_tensor, result
            torch.cuda.empty_cache()
            print("✅ GPU memory cleaned up")
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
    else:
        print("❌ CUDA not available, skipping GPU memory test")

if __name__ == "__main__":
    check_gpu_status()
