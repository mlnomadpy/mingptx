#!/usr/bin/env python3
"""
Benchmark script to compare data loading performance between 
TensorFlow-based and Grain-based implementations.
"""

import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import argparse

from src.config import DataConfig, ModelConfig, TrainConfig
from src.dataset import load_text_dataset, load_text_dataset_tf_fallback

def benchmark_data_loader(loader, name: str, num_batches: int = 100):
    """Benchmark a data loader by measuring time to iterate through batches."""
    print(f"\n--- Benchmarking {name} ---")
    
    start_time = time.time()
    batch_times = []
    
    try:
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # Simulate some processing (like what happens in training)
            if isinstance(batch, tuple):
                inputs, targets = batch
                # Force computation to ensure data is actually loaded
                _ = jnp.sum(inputs), jnp.sum(targets)
            else:
                # For TF data, convert to JAX arrays
                inputs, targets = batch
                inputs_jax = jnp.array(inputs)
                targets_jax = jnp.array(targets)
                _ = jnp.sum(inputs_jax), jnp.sum(targets_jax)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if (i + 1) % 10 == 0:
                avg_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                print(f"  Batch {i+1}/{num_batches}, Avg time (last 10): {avg_time:.4f}s")
    
    except Exception as e:
        print(f"Error during benchmarking {name}: {e}")
        return None
    
    total_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average batch time: {avg_batch_time:.4f}s")
    print(f"  Batches per second: {len(batch_times) / total_time:.2f}")
    print(f"  Samples per second: {len(batch_times) * 512 / total_time:.0f}")  # Assuming batch_size=512
    
    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'batches_per_second': len(batch_times) / total_time,
        'samples_per_second': len(batch_times) * 512 / total_time
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark data loading performance")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to benchmark")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for testing")
    parser.add_argument("--maxlen", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--skip_tf", action="store_true", help="Skip TensorFlow benchmark")
    parser.add_argument("--skip_grain", action="store_true", help="Skip Grain benchmark")
    args = parser.parse_args()
    
    # Configuration
    model_config = ModelConfig(maxlen=args.maxlen)
    data_config = DataConfig(
        dataset_name='roneneldan/TinyStories',
        split='train',
        batch_size=args.batch_size
    )
    train_config = TrainConfig(num_epochs=1)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("JAX devices:", jax.devices())
    print(f"Benchmarking with batch_size={args.batch_size}, maxlen={args.maxlen}")
    
    results = {}
    
    # Benchmark Grain-based loader (new implementation)
    if not args.skip_grain:
        try:
            print("\n" + "="*50)
            print("Loading Grain-based data loader...")
            grain_loader = load_text_dataset(
                data_config, model_config, train_config, 
                data_config.tokenizer_name, tokenizer.pad_token_id
            )
            results['grain'] = benchmark_data_loader(grain_loader, "Grain-based loader", args.num_batches)
        except Exception as e:
            print(f"Failed to benchmark Grain loader: {e}")
            results['grain'] = None
    
    # Benchmark TensorFlow-based loader (original implementation)
    if not args.skip_tf:
        try:
            print("\n" + "="*50)
            print("Loading TensorFlow-based data loader...")
            tf_loader = load_text_dataset_tf_fallback(
                data_config, model_config, train_config,
                data_config.tokenizer_name, tokenizer.pad_token_id
            )
            # Convert to numpy iterator for fair comparison
            tf_loader_iter = tf_loader.as_numpy_iterator()
            results['tensorflow'] = benchmark_data_loader(tf_loader_iter, "TensorFlow-based loader", args.num_batches)
        except Exception as e:
            print(f"Failed to benchmark TensorFlow loader: {e}")
            results['tensorflow'] = None
    
    # Summary comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*50)
    
    if results.get('grain') and results.get('tensorflow'):
        grain_stats = results['grain']
        tf_stats = results['tensorflow']
        
        speedup = tf_stats['avg_batch_time'] / grain_stats['avg_batch_time']
        throughput_improvement = grain_stats['samples_per_second'] / tf_stats['samples_per_second']
        
        print(f"Grain vs TensorFlow:")
        print(f"  Speedup (batch time): {speedup:.2f}x faster")
        print(f"  Throughput improvement: {throughput_improvement:.2f}x more samples/sec")
        print(f"  Grain avg batch time: {grain_stats['avg_batch_time']:.4f}s")
        print(f"  TF avg batch time: {tf_stats['avg_batch_time']:.4f}s")
        print(f"  Grain samples/sec: {grain_stats['samples_per_second']:.0f}")
        print(f"  TF samples/sec: {tf_stats['samples_per_second']:.0f}")
    
    elif results.get('grain'):
        print("Grain-based loader results:")
        print(f"  Avg batch time: {results['grain']['avg_batch_time']:.4f}s")
        print(f"  Samples/sec: {results['grain']['samples_per_second']:.0f}")
    
    elif results.get('tensorflow'):
        print("TensorFlow-based loader results:")
        print(f"  Avg batch time: {results['tensorflow']['avg_batch_time']:.4f}s")
        print(f"  Samples/sec: {results['tensorflow']['samples_per_second']:.0f}")

if __name__ == "__main__":
    main() 