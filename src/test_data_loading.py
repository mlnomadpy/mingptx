#!/usr/bin/env python3
"""
Performance comparison between TensorFlow and Grain data loading implementations.
"""

import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import numpy as np

from config import DataConfig, ModelConfig, TrainConfig
from dataset import load_text_dataset, load_text_dataset_tf_fallback

def benchmark_data_loader(loader, name: str, num_batches: int = 20, is_tf: bool = False):
    """Benchmark a data loader by measuring detailed performance metrics."""
    print(f"\n--- Benchmarking {name} ---")
    
    start_time = time.time()
    batch_times = []
    memory_usage = []
    
    try:
        iterator = loader.as_numpy_iterator() if is_tf else loader
        
        for i, batch in enumerate(iterator):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # Process batch and measure performance
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
                
                # Convert TF data to JAX if needed
                if is_tf:
                    inputs_jax = jnp.array(inputs)
                    targets_jax = jnp.array(targets)
                else:
                    inputs_jax, targets_jax = inputs, targets
                
                # Force computation to ensure data is actually loaded
                input_sum = jnp.sum(inputs_jax)
                target_sum = jnp.sum(targets_jax)
                
                # Verify shapes and data integrity
                if i == 0:
                    print(f"  Input shape: {inputs_jax.shape}")
                    print(f"  Target shape: {targets_jax.shape}")
                    print(f"  Data type: {inputs_jax.dtype}")
                    
                    # Check target shifting is correct
                    if inputs_jax.shape == targets_jax.shape:
                        sample_input = inputs_jax[:, 0] if len(inputs_jax.shape) > 1 else inputs_jax
                        sample_target = targets_jax[:, 0] if len(targets_jax.shape) > 1 else targets_jax
                        
                        if len(sample_input) > 1:
                            shift_correct = jnp.array_equal(sample_input[1:], sample_target[:-1])
                            print(f"  Target shifting correct: {shift_correct}")
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Estimate memory usage (rough approximation)
                mem_estimate = inputs_jax.nbytes + targets_jax.nbytes
                memory_usage.append(mem_estimate)
                
                if (i + 1) % 5 == 0:
                    avg_time = np.mean(batch_times[-5:])
                    print(f"  Batch {i+1}/{num_batches}, Avg time (last 5): {avg_time:.4f}s")
            else:
                print(f"  Batch {i+1}: unexpected structure {type(batch)}")
                return None
    
    except Exception as e:
        print(f"  Error during benchmarking {name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    total_time = time.time() - start_time
    
    if not batch_times:
        print(f"  No batches processed successfully")
        return None
    
    # Calculate detailed statistics
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    min_batch_time = np.min(batch_times)
    max_batch_time = np.max(batch_times)
    avg_memory = np.mean(memory_usage) / (1024**2)  # MB
    
    # Assume batch size from config
    batch_size = 128  # This should match your config
    samples_per_second = batch_size / avg_batch_time
    
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average batch time: {avg_batch_time:.4f}s ± {std_batch_time:.4f}s")
    print(f"  Min/Max batch time: {min_batch_time:.4f}s / {max_batch_time:.4f}s")
    print(f"  Samples per second: {samples_per_second:.0f}")
    print(f"  Average memory per batch: {avg_memory:.1f} MB")
    print(f"  Throughput: {len(batch_times) / total_time:.2f} batches/sec")
    
    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'std_batch_time': std_batch_time,
        'min_batch_time': min_batch_time,
        'max_batch_time': max_batch_time,
        'samples_per_second': samples_per_second,
        'batches_per_second': len(batch_times) / total_time,
        'avg_memory_mb': avg_memory,
        'num_batches': len(batch_times)
    }

def test_grain_data_loading(num_batches: int = 20):
    """Test the Grain-based data loading implementation."""
    
    print("Testing Grain data loading...")
    
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
        print("Loading Grain data loader...")
        load_start = time.time()
        
        grain_loader = load_text_dataset(
            data_config, model_config, train_config, 
            data_config.tokenizer_name, tokenizer.pad_token_id
        )
        
        load_time = time.time() - load_start
        print(f"Grain loader created in {load_time:.2f} seconds")
        
        # Benchmark performance
        results = benchmark_data_loader(grain_loader, "Grain", num_batches, is_tf=False)
        
        if results:
            results['load_time'] = load_time
            print("✅ Grain data loading test completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during Grain data loading test: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_tf_fallback_data_loading(num_batches: int = 20):
    """Test the TensorFlow fallback data loading implementation."""
    
    print("\nTesting TensorFlow fallback data loading...")
    
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
        print("Loading TensorFlow fallback data loader...")
        load_start = time.time()
        
        tf_loader = load_text_dataset_tf_fallback(
            data_config, model_config, train_config,
            data_config.tokenizer_name, tokenizer.pad_token_id
        )
        
        load_time = time.time() - load_start
        print(f"TensorFlow loader created in {load_time:.2f} seconds")
        
        # Benchmark performance
        results = benchmark_data_loader(tf_loader, "TensorFlow", num_batches, is_tf=True)
        
        if results:
            results['load_time'] = load_time
            print("✅ TensorFlow fallback test completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during TensorFlow fallback test: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(grain_results, tf_results):
    """Compare and analyze the performance results."""
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: Grain vs TensorFlow")
    print("="*60)
    
    if not grain_results and not tf_results:
        print("❌ Both implementations failed - no comparison possible")
        return
    
    if not grain_results:
        print("❌ Grain failed, only TensorFlow results available")
        print_single_results("TensorFlow", tf_results)
        return
    
    if not tf_results:
        print("❌ TensorFlow failed, only Grain results available")
        print_single_results("Grain", grain_results)
        return
    
    # Both implementations worked - do detailed comparison
    print(f"{'Metric':<25} {'Grain':<15} {'TensorFlow':<15} {'Winner':<15}")
    print("-" * 70)
    
    metrics = [
        ('Load Time (s)', 'load_time', 'lower_better'),
        ('Avg Batch Time (s)', 'avg_batch_time', 'lower_better'),
        ('Batch Time Std (s)', 'std_batch_time', 'lower_better'),
        ('Min Batch Time (s)', 'min_batch_time', 'lower_better'),
        ('Max Batch Time (s)', 'max_batch_time', 'lower_better'),
        ('Samples/sec', 'samples_per_second', 'higher_better'),
        ('Batches/sec', 'batches_per_second', 'higher_better'),
        ('Memory/batch (MB)', 'avg_memory_mb', 'lower_better'),
    ]
    
    grain_wins = 0
    tf_wins = 0
    
    for metric_name, key, direction in metrics:
        grain_val = grain_results[key]
        tf_val = tf_results[key]
        
        if direction == 'lower_better':
            winner = "Grain" if grain_val < tf_val else "TensorFlow"
            if grain_val < tf_val:
                grain_wins += 1
            else:
                tf_wins += 1
        else:  # higher_better
            winner = "Grain" if grain_val > tf_val else "TensorFlow"
            if grain_val > tf_val:
                grain_wins += 1
            else:
                tf_wins += 1
        
        print(f"{metric_name:<25} {grain_val:<15.4f} {tf_val:<15.4f} {winner:<15}")
    
    print("-" * 70)
    print(f"{'SCORE':<25} {grain_wins:<15} {tf_wins:<15}")
    
    # Calculate key performance ratios
    speedup = tf_results['avg_batch_time'] / grain_results['avg_batch_time']
    throughput_ratio = grain_results['samples_per_second'] / tf_results['samples_per_second']
    load_time_ratio = grain_results['load_time'] / tf_results['load_time']
    
    print(f"\n📊 KEY PERFORMANCE METRICS:")
    print(f"   Grain is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} per batch")
    print(f"   Grain has {throughput_ratio:.2f}x {'higher' if throughput_ratio > 1 else 'lower'} throughput")
    print(f"   Grain load time is {load_time_ratio:.2f}x {'longer' if load_time_ratio > 1 else 'shorter'}")
    
    # Overall winner
    if grain_wins > tf_wins:
        print(f"\n🏆 OVERALL WINNER: Grain ({grain_wins}/{len(metrics)} metrics)")
        print("   Recommendation: Use Grain for better performance")
    elif tf_wins > grain_wins:
        print(f"\n🏆 OVERALL WINNER: TensorFlow ({tf_wins}/{len(metrics)} metrics)")
        print("   Recommendation: Use TensorFlow fallback for now")
    else:
        print(f"\n🤝 TIE: Both implementations perform similarly")
        print("   Recommendation: Choose based on other factors (ecosystem, features)")

def print_single_results(name, results):
    """Print results for a single implementation."""
    if not results:
        return
    
    print(f"\n{name} Results:")
    print(f"  Load time: {results['load_time']:.2f}s")
    print(f"  Average batch time: {results['avg_batch_time']:.4f}s")
    print(f"  Samples per second: {results['samples_per_second']:.0f}")
    print(f"  Batches per second: {results['batches_per_second']:.2f}")
    print(f"  Memory per batch: {results['avg_memory_mb']:.1f} MB")

def main():
    print("Data Loading Performance Comparison: Grain vs TensorFlow")
    print("="*60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Configuration: batch_size=128, maxlen=256, num_batches=20")
    
    # Test both implementations
    grain_results = test_grain_data_loading(num_batches=20)
    tf_results = test_tf_fallback_data_loading(num_batches=20)
    
    # Compare results
    compare_results(grain_results, tf_results)
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    if grain_results:
        print(f"   For Grain: Consider increasing num_threads and cache_size")
    if tf_results:
        print(f"   For TensorFlow: Already well-optimized with tf.data")
    print(f"   General: Monitor memory usage and adjust batch_size for your hardware")

if __name__ == "__main__":
    main() 