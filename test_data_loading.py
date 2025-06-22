#!/usr/bin/env python3
"""
Simple test script to verify the Grain-based data loading works correctly.
"""

import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from src.config import DataConfig, ModelConfig, TrainConfig
from src.dataset import load_text_dataset

def test_grain_data_loading():
    """Test the Grain-based data loading implementation."""
    
    print("Testing Grain-based data loading...")
    print(f"JAX devices: {jax.devices()}")
    
    # Configuration
    model_config = ModelConfig(maxlen=256)
    data_config = DataConfig(
        dataset_name='roneneldan/TinyStories',
        split='train',
        batch_size=128  # Smaller batch for testing
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
        
        grain_loader = load_text_dataset(
            data_config, model_config, train_config, 
            data_config.tokenizer_name, tokenizer.pad_token_id
        )
        
        load_time = time.time() - start_time
        print(f"Data loader created in {load_time:.2f} seconds")
        
        # Test iteration
        print("\nTesting data iteration...")
        batch_count = 0
        total_time = 0
        
        for i, batch in enumerate(grain_loader):
            if i >= 10:  # Test only first 10 batches
                break
                
            batch_start = time.time()
            
            # Verify batch structure
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
                print(f"Batch {i+1}: inputs shape {inputs.shape}, targets shape {targets.shape}")
                
                # Force computation to ensure data is loaded
                _ = jnp.sum(inputs), jnp.sum(targets)
            else:
                print(f"Batch {i+1}: unexpected structure {type(batch)}")
            
            batch_time = time.time() - batch_start
            total_time += batch_time
            batch_count += 1
            
            if i == 0:
                print(f"  First batch processed in {batch_time:.4f}s")
        
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

def test_fallback_data_loading():
    """Test the TensorFlow fallback data loading implementation."""
    
    print("\n" + "="*50)
    print("Testing TensorFlow fallback data loading...")
    
    # Configuration
    model_config = ModelConfig(maxlen=256)
    data_config = DataConfig(
        dataset_name='roneneldan/TinyStories',
        split='train',
        batch_size=128
    )
    train_config = TrainConfig(num_epochs=1)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        from src.dataset import load_text_dataset_tf_fallback
        
        print("Loading TensorFlow fallback data loader...")
        start_time = time.time()
        
        tf_loader = load_text_dataset_tf_fallback(
            data_config, model_config, train_config,
            data_config.tokenizer_name, tokenizer.pad_token_id
        )
        
        load_time = time.time() - start_time
        print(f"TF data loader created in {load_time:.2f} seconds")
        
        # Test iteration
        print("Testing TF data iteration...")
        batch_count = 0
        total_time = 0
        
        for i, batch in enumerate(tf_loader.as_numpy_iterator()):
            if i >= 5:  # Test fewer batches for TF
                break
                
            batch_start = time.time()
            
            # Verify batch structure
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
                print(f"TF Batch {i+1}: inputs shape {inputs.shape}, targets shape {targets.shape}")
                
                # Convert to JAX and force computation
                inputs_jax = jnp.array(inputs)
                targets_jax = jnp.array(targets)
                _ = jnp.sum(inputs_jax), jnp.sum(targets_jax)
            else:
                print(f"TF Batch {i+1}: unexpected structure {type(batch)}")
            
            batch_time = time.time() - batch_start
            total_time += batch_time
            batch_count += 1
        
        if batch_count > 0:
            avg_time = total_time / batch_count
            print(f"TF processed {batch_count} batches")
            print(f"TF average batch time: {avg_time:.4f}s")
        
        print("✅ TensorFlow fallback test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during TF fallback test: {e}")
        return False

if __name__ == "__main__":
    print("Data Loading Performance Test")
    print("="*50)
    
    # Test Grain implementation
    grain_success = test_grain_data_loading()
    
    # Test TensorFlow fallback
    tf_success = test_fallback_data_loading()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Grain implementation: {'✅ PASSED' if grain_success else '❌ FAILED'}")
    print(f"TensorFlow fallback: {'✅ PASSED' if tf_success else '❌ FAILED'}")
    
    if grain_success:
        print("\n🎉 The optimized Grain-based data loading is working!")
        print("You can now use the improved data loading pipeline for better performance.")
    elif tf_success:
        print("\n⚠️  Grain failed but TensorFlow fallback works.")
        print("Consider using load_text_dataset_tf_fallback() for now.")
    else:
        print("\n❌ Both implementations failed. Please check the error messages above.") 