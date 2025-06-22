#!/usr/bin/env python3
"""
Comprehensive comparison between TensorFlow and Grain data loading implementations.
"""

import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import numpy as np

# Import from the current directory structure
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def test_grain_data_loading(num_batches: int = 20, use_optimized: bool = True):
    """Test the Grain-based data loading implementation."""
    
    impl_name = "Optimized Grain" if use_optimized else "Original Grain"
    print(f"Testing {impl_name} data loading...")
    
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
        print(f"Loading {impl_name} data loader...")
        load_start = time.time()
        
        grain_loader = load_text_dataset(
            data_config, model_config, train_config, 
            data_config.tokenizer_name, tokenizer.pad_token_id,
            use_optimized=use_optimized
        )
        
        load_time = time.time() - load_start
        print(f"{impl_name} loader created in {load_time:.2f} seconds")
        
        # Benchmark performance
        results = benchmark_data_loader(grain_loader, impl_name, num_batches, is_tf=False)
        
        if results:
            results['load_time'] = load_time
            print(f"✅ {impl_name} data loading test completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during {impl_name} data loading test: {e}")
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

def compare_three_results(optimized_grain_results, original_grain_results, tf_results):
    """Compare and analyze the performance results for all three implementations."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*80)
    
    results = [
        ("Optimized Grain", optimized_grain_results),
        ("Original Grain", original_grain_results), 
        ("TensorFlow", tf_results)
    ]
    
    # Filter out failed implementations
    valid_results = [(name, res) for name, res in results if res is not None]
    
    if len(valid_results) == 0:
        print("❌ All implementations failed - no comparison possible")
        return
    
    if len(valid_results) == 1:
        name, res = valid_results[0]
        print(f"Only {name} succeeded:")
        print_single_results(name, res)
        return
    
    # Multi-implementation comparison
    print(f"{'Metric':<25}", end="")
    for name, _ in valid_results:
        print(f"{name:<18}", end="")
    print("Winner")
    print("-" * 100)
    
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
    
    scores = {name: 0 for name, _ in valid_results}
    
    for metric_name, key, direction in metrics:
        print(f"{metric_name:<25}", end="")
        
        values = []
        for name, res in valid_results:
            val = res[key]
            values.append((name, val))
            print(f"{val:<18.4f}", end="")
        
        # Determine winner
        if direction == 'lower_better':
            winner_name, winner_val = min(values, key=lambda x: x[1])
        else:  # higher_better
            winner_name, winner_val = max(values, key=lambda x: x[1])
        
        scores[winner_name] += 1
        print(f"{winner_name}")
    
    print("-" * 100)
    print(f"{'FINAL SCORES':<25}", end="")
    for name, _ in valid_results:
        print(f"{scores[name]:<18}", end="")
    
    # Determine overall winner
    winner = max(scores.items(), key=lambda x: x[1])
    print(f"{winner[0]} (Winner)")
    
    # Calculate performance ratios if we have all three
    if len(valid_results) == 3:
        opt_grain = next(res for name, res in valid_results if "Optimized" in name)
        orig_grain = next(res for name, res in valid_results if "Original" in name) 
        tf = next(res for name, res in valid_results if "TensorFlow" in name)
        
        print(f"\n📊 KEY PERFORMANCE COMPARISONS:")
        
        # Optimized Grain vs TensorFlow
        opt_vs_tf_speed = tf['avg_batch_time'] / opt_grain['avg_batch_time']
        opt_vs_tf_throughput = opt_grain['samples_per_second'] / tf['samples_per_second']
        print(f"   Optimized Grain vs TensorFlow:")
        print(f"     - {opt_vs_tf_speed:.2f}x {'faster' if opt_vs_tf_speed > 1 else 'slower'} per batch")
        print(f"     - {opt_vs_tf_throughput:.2f}x {'higher' if opt_vs_tf_throughput > 1 else 'lower'} throughput")
        
        # Optimized vs Original Grain
        opt_vs_orig_speed = orig_grain['avg_batch_time'] / opt_grain['avg_batch_time']
        opt_vs_orig_throughput = opt_grain['samples_per_second'] / orig_grain['samples_per_second']
        print(f"   Optimized Grain vs Original Grain:")
        print(f"     - {opt_vs_orig_speed:.2f}x {'faster' if opt_vs_orig_speed > 1 else 'slower'} per batch")
        print(f"     - {opt_vs_orig_throughput:.2f}x {'higher' if opt_vs_orig_throughput > 1 else 'lower'} throughput")
    
    print(f"\n🏆 RECOMMENDATION:")
    if winner[0] == "Optimized Grain":
        print("   Use the optimized Grain implementation for best performance!")
    elif winner[0] == "TensorFlow":
        print("   TensorFlow still performs best - consider using it for now")
    else:
        print(f"   Use {winner[0]} for your data loading pipeline")

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
    print("Comprehensive Data Loading Performance Comparison")
    print("="*60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Configuration: batch_size=128, maxlen=256, num_batches=20")
    
    # Test all three implementations
    print("\n🚀 Testing Optimized Grain Implementation...")
    optimized_grain_results = test_grain_data_loading(num_batches=20, use_optimized=True)
    
    print("\n📊 Testing Original Grain Implementation...")
    original_grain_results = test_grain_data_loading(num_batches=20, use_optimized=False)
    
    print("\n🔄 Testing TensorFlow Implementation...")
    tf_results = test_tf_fallback_data_loading(num_batches=20)
    
    # Compare all results
    compare_three_results(optimized_grain_results, original_grain_results, tf_results)
    
    # Final recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    if optimized_grain_results:
        print(f"   For Optimized Grain: Further tune cache_size and num_threads based on your hardware")
    if original_grain_results:
        print(f"   For Original Grain: Consider using the optimized version instead")
    if tf_results:
        print(f"   For TensorFlow: Already well-optimized, consider tf.data.experimental.AUTOTUNE")
    print(f"   General: Monitor memory usage and adjust batch_size accordingly")

if __name__ == "__main__":
    main() 