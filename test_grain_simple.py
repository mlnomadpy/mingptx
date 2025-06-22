#!/usr/bin/env python3
"""
Simple test script to verify the Grain-based data loading works with chaining API.
"""

import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

# Import from the current directory structure
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DataConfig, ModelConfig, TrainConfig
from src.dataset import load_text_dataset

def test_grain_data_loading():
    """Test the Grain-based data loading implementation."""
    
    print("Testing Grain-based data loading with chaining API...")
    print(f"JAX devices: {jax.devices()}")
    
    # Configuration with smaller settings for testing
    model_config = ModelConfig(maxlen=128)  # Smaller for faster testing
    data_config = DataConfig(
        dataset_name='roneneldan/TinyStories',
        split='train',
        batch_size=16  # Small batch for testing
    )
    train_config = TrainConfig(num_epochs=1)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Testing with batch_size={data_config.batch_size}, maxlen={model_config.maxlen}")
    
    try:
        print("\nLoading Grain-based data loader...")
        start_time = time.time()
        
        grain_dataset = load_text_dataset(
            data_config, model_config, train_config, 
            data_config.tokenizer_name, tokenizer.pad_token_id
        )
        
        load_time = time.time() - start_time
        print(f"Data loader created in {load_time:.2f} seconds")
        
        # Test iteration
        print("\nTesting data iteration...")
        batch_count = 0
        total_time = 0
        
        for i, batch in enumerate(grain_dataset):
            if i >= 3:  # Test only first 3 batches
                break
                
            batch_start = time.time()
            
            # Verify batch structure
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
                print(f"Batch {i+1}: inputs shape {inputs.shape}, targets shape {targets.shape}")
                
                # Verify the shapes are correct
                expected_input_shape = (model_config.maxlen, data_config.batch_size)
                expected_target_shape = (model_config.maxlen, data_config.batch_size)
                
                if inputs.shape == expected_input_shape and targets.shape == expected_target_shape:
                    print(f"  ✅ Shapes are correct!")
                    
                    # Check that targets are shifted versions of inputs
                    input_sample = inputs[:, 0]  # First sequence
                    target_sample = targets[:, 0]  # First target sequence
                    
                    # Check if target[:-1] matches input[1:]
                    if jnp.array_equal(input_sample[1:], target_sample[:-1]):
                        print(f"  ✅ Target shifting is correct!")
                    else:
                        print(f"  ⚠️  Target shifting might have issues")
                        
                else:
                    print(f"  ❌ Shape mismatch! Expected {expected_input_shape}, got {inputs.shape}")
                
                # Force computation to ensure data is loaded
                _ = jnp.sum(inputs), jnp.sum(targets)
            else:
                print(f"Batch {i+1}: unexpected structure {type(batch)}")
                if hasattr(batch, 'shape'):
                    print(f"  Shape: {batch.shape}")
                return False
            
            batch_time = time.time() - batch_start
            total_time += batch_time
            batch_count += 1
            
            print(f"  Batch processed in {batch_time:.4f}s")
        
        if batch_count > 0:
            avg_time = total_time / batch_count
            print(f"\nProcessed {batch_count} batches")
            print(f"Average batch time: {avg_time:.4f}s")
            print(f"Estimated samples/sec: {data_config.batch_size / avg_time:.0f}")
        
        print("\n✅ Grain data loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during Grain data loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Grain Data Loading Test (Chaining API)")
    print("="*50)
    
    success = test_grain_data_loading()
    
    print("\n" + "="*50)
    print("TEST RESULT")
    print("="*50)
    
    if success:
        print("🎉 The optimized Grain-based data loading is working!")
        print("You can now use the improved data loading pipeline for better performance.")
        print("\nNext steps:")
        print("1. Increase num_threads in ReadOptions for better performance")
        print("2. Adjust cache_size based on your available memory")
        print("3. Increase batch_size for better GPU utilization")
        print("4. Test with your actual training script")
    else:
        print("❌ Grain implementation failed. Check the error messages above.")
        print("Consider using the TensorFlow fallback implementation.") 