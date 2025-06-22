#!/usr/bin/env python3
"""
Test script to demonstrate dynamic cache behavior in TextDataSource.
This shows how the cache gets updated with new data over time.
"""

import sys
import os
sys.path.append('src')

from config import DataConfig
from dataset import TextDataSource

def test_dynamic_cache():
    """Test that cache updates with new data over time."""
    
    # Create a small cache for testing
    config = DataConfig(
        dataset_name="roneneldan/TinyStories",
        split="train",
        cache_size=100,  # Small cache for quick testing
        cache_refresh_rate=20,  # Refresh 20 entries each time
        cache_refresh_interval=1,  # Refresh every 1 full cache cycle
        shuffle_seed=42,
        shuffle_buffer_size=1000
    )
    
    # Create data source
    data_source = TextDataSource(
        dataset_name=config.dataset_name,
        split=config.split,
        tokenizer_name="gpt2",
        max_length=128,
        d_config=config
    )
    
    print("Initial cache stats:")
    print(data_source.get_cache_stats())
    
    # Store first few examples to check if they change
    initial_examples = []
    for i in range(5):
        initial_examples.append(data_source[i].copy())
    
    print(f"\nFirst 5 examples (first 10 tokens each):")
    for i, example in enumerate(initial_examples):
        print(f"Example {i}: {example[:10]}")
    
    # Access enough examples to trigger cache refresh
    print(f"\nAccessing {config.cache_size * config.cache_refresh_interval} examples to trigger cache refresh...")
    for i in range(config.cache_size * config.cache_refresh_interval):
        _ = data_source[i % len(data_source)]
    
    print("\nCache stats after accessing many examples:")
    print(data_source.get_cache_stats())
    
    # Check if some examples have changed (indicating cache refresh)
    print(f"\nFirst 5 examples after cache refresh (first 10 tokens each):")
    changes_detected = 0
    for i in range(5):
        current_example = data_source[i]
        print(f"Example {i}: {current_example[:10]}")
        if not np.array_equal(current_example, initial_examples[i]):
            changes_detected += 1
            print(f"  -> CHANGED from initial!")
    
    print(f"\nDetected {changes_detected} changes out of 5 examples")
    if changes_detected > 0:
        print("✅ Dynamic cache is working - cache has been updated with new data!")
    else:
        print("⚠️  No changes detected - cache might not be refreshing as expected")

if __name__ == "__main__":
    import numpy as np
    test_dynamic_cache() 