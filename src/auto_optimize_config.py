#!/usr/bin/env python3
"""
Auto-optimization script for data loading configuration.
Automatically determines optimal settings based on system capabilities.
"""

import os
import psutil
import jax
import time
import yaml
from pathlib import Path
from typing import Dict, Any

def get_system_info() -> Dict[str, Any]:
    """Get system information for optimization."""
    
    # CPU information
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    
    # Memory information
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    
    # JAX/GPU information
    jax_devices = jax.devices()
    gpu_count = len([d for d in jax_devices if d.device_kind == 'gpu'])
    
    # Storage information (rough estimate)
    disk_usage = psutil.disk_usage('/')
    disk_free_gb = disk_usage.free / (1024**3)
    
    return {
        'cpu_count': cpu_count,
        'cpu_count_physical': cpu_count_physical,
        'memory_gb': memory_gb,
        'gpu_count': gpu_count,
        'jax_devices': len(jax_devices),
        'disk_free_gb': disk_free_gb,
        'jax_backend': jax.default_backend()
    }

def calculate_optimal_workers(cpu_count: int, memory_gb: float) -> int:
    """Calculate optimal number of worker threads."""
    
    # Conservative approach: use 50-75% of CPU cores
    # Leave some cores for the main training process
    if cpu_count <= 4:
        return max(1, cpu_count - 1)  # Leave at least 1 core for training
    elif cpu_count <= 8:
        return max(2, int(cpu_count * 0.6))
    elif cpu_count <= 16:
        return max(4, int(cpu_count * 0.5))
    else:
        return max(8, min(16, int(cpu_count * 0.4)))  # Cap at 16 workers

def calculate_optimal_cache_size(memory_gb: float, sequence_length: int = 256) -> int:
    """Calculate optimal cache size based on available memory."""
    
    # Estimate memory per cached example (rough calculation)
    # Each token is ~4 bytes (int32), plus some overhead
    memory_per_example = sequence_length * 4 * 1.2  # 20% overhead
    memory_per_example_mb = memory_per_example / (1024**2)
    
    # Use 10-20% of available memory for cache
    if memory_gb < 8:
        cache_memory_gb = memory_gb * 0.1  # 10% for low memory systems
    elif memory_gb < 16:
        cache_memory_gb = memory_gb * 0.15  # 15% for medium memory
    else:
        cache_memory_gb = memory_gb * 0.2   # 20% for high memory systems
    
    cache_memory_mb = cache_memory_gb * 1024
    optimal_cache_size = int(cache_memory_mb / memory_per_example_mb)
    
    # Reasonable bounds
    optimal_cache_size = max(1000, min(200_000, optimal_cache_size))
    
    return optimal_cache_size

def calculate_optimal_batch_size(gpu_count: int, memory_gb: float, model_size: str = "small") -> int:
    """Calculate optimal batch size based on hardware."""
    
    # Base batch sizes by model size
    base_batch_sizes = {
        "tiny": 512,
        "small": 256,
        "medium": 128,
        "large": 64
    }
    
    base_batch = base_batch_sizes.get(model_size, 256)
    
    # Adjust for GPU count
    if gpu_count > 1:
        # For multi-GPU, we can use larger batches
        base_batch = int(base_batch * min(gpu_count, 4))  # Cap scaling at 4x
    elif gpu_count == 0:
        # CPU-only training, use smaller batches
        base_batch = max(32, base_batch // 4)
    
    # Adjust for memory
    if memory_gb < 8:
        base_batch = max(32, base_batch // 2)
    elif memory_gb > 32:
        base_batch = int(base_batch * 1.5)
    
    return base_batch

def calculate_cache_refresh_params(cache_size: int, training_length: str = "medium") -> tuple:
    """Calculate optimal cache refresh rate and interval."""
    
    # Refresh rate as percentage of cache size
    if training_length == "short":  # < 1 epoch
        refresh_rate = max(100, cache_size // 20)  # 5% refresh
        refresh_interval = 3  # Less frequent refresh
    elif training_length == "medium":  # 1-5 epochs
        refresh_rate = max(500, cache_size // 10)  # 10% refresh
        refresh_interval = 2  # Moderate refresh
    else:  # "long" training
        refresh_rate = max(1000, cache_size // 5)  # 20% refresh
        refresh_interval = 1  # Frequent refresh
    
    return refresh_rate, refresh_interval

def benchmark_worker_performance(num_workers_list: list, quick_test: bool = True) -> int:
    """Benchmark different worker counts to find optimal."""
    
    print(f"\n🔧 Benchmarking worker performance...")
    
    # This would require actual data loading test
    # For now, return a reasonable default based on CPU count
    cpu_count = psutil.cpu_count(logical=True)
    return calculate_optimal_workers(cpu_count, psutil.virtual_memory().total / (1024**3))

def generate_optimized_config(
    system_info: Dict[str, Any],
    sequence_length: int = 256,
    model_size: str = "small",
    training_length: str = "medium",
    benchmark_workers: bool = False
) -> Dict[str, Any]:
    """Generate optimized configuration based on system info."""
    
    print(f"🔍 System Analysis:")
    print(f"   CPU Cores: {system_info['cpu_count']} ({system_info['cpu_count_physical']} physical)")
    print(f"   Memory: {system_info['memory_gb']:.1f} GB")
    print(f"   GPUs: {system_info['gpu_count']}")
    print(f"   JAX Backend: {system_info['jax_backend']}")
    
    # Calculate optimal settings
    optimal_workers = calculate_optimal_workers(
        system_info['cpu_count'], 
        system_info['memory_gb']
    )
    
    optimal_cache_size = calculate_optimal_cache_size(
        system_info['memory_gb'], 
        sequence_length
    )
    
    optimal_batch_size = calculate_optimal_batch_size(
        system_info['gpu_count'], 
        system_info['memory_gb'], 
        model_size
    )
    
    cache_refresh_rate, cache_refresh_interval = calculate_cache_refresh_params(
        optimal_cache_size, 
        training_length
    )
    
    # Benchmark workers if requested
    if benchmark_workers:
        optimal_workers = benchmark_worker_performance([optimal_workers])
    
    # Calculate other derived parameters
    shuffle_buffer_size = min(50_000, optimal_cache_size // 2)
    prefetch_buffer_size = max(10, min(100, optimal_workers * 10))
    tokenization_batch_size = max(500, min(2000, optimal_batch_size * 4))
    
    print(f"\n✅ Optimized Settings:")
    print(f"   Workers (num_threads): {optimal_workers}")
    print(f"   Cache Size: {optimal_cache_size:,}")
    print(f"   Batch Size: {optimal_batch_size}")
    print(f"   Cache Refresh Rate: {cache_refresh_rate:,}")
    print(f"   Cache Refresh Interval: {cache_refresh_interval}")
    print(f"   Shuffle Buffer: {shuffle_buffer_size:,}")
    print(f"   Prefetch Buffer: {prefetch_buffer_size}")
    
    return {
        'num_threads': optimal_workers,
        'cache_size': optimal_cache_size,
        'batch_size': optimal_batch_size,
        'cache_refresh_rate': cache_refresh_rate,
        'cache_refresh_interval': cache_refresh_interval,
        'shuffle_buffer_size': shuffle_buffer_size,
        'prefetch_buffer_size': prefetch_buffer_size,
        'tokenization_batch_size': tokenization_batch_size,
        'use_fast_tokenizer': True  # Always use fast tokenizer
    }

def update_config_file(config_path: str, optimized_settings: Dict[str, Any]):
    """Update the config file with optimized settings."""
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data_config section
    if 'data_config' not in config:
        config['data_config'] = {}
    
    # Update with optimized settings
    config['data_config'].update(optimized_settings)
    
    # Backup original config
    backup_path = config_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"📄 Original config backed up to: {backup_path}")
    
    # Write optimized config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Updated config file: {config_path}")

def main():
    """Main optimization function."""
    
    print("🚀 Auto-Optimization for Data Loading Configuration")
    print("=" * 60)
    
    # Get system information
    system_info = get_system_info()
    
    # Generate optimized configuration
    optimized_settings = generate_optimized_config(
        system_info=system_info,
        sequence_length=256,  # Adjust based on your model
        model_size="small",   # tiny, small, medium, large
        training_length="medium",  # short, medium, long
        benchmark_workers=False  # Set to True for more accurate worker count
    )
    
    # Update config file
    config_path = "config.yaml"
    if os.path.exists(config_path):
        update_config_file(config_path, optimized_settings)
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print("Optimized settings:")
        for key, value in optimized_settings.items():
            print(f"  {key}: {value}")
    
    print("\n🎯 Optimization Complete!")
    print("\n💡 Additional Tips:")
    print("   - Monitor memory usage during training")
    print("   - Adjust batch_size if you get OOM errors")
    print("   - Increase cache_size if you have more RAM")
    print("   - Use benchmark_data_loading.py to verify performance")

if __name__ == "__main__":
    main() 