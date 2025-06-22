import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import sys
import time

# Mock configurations to mimic the real ones
@dataclass
class MockDataConfig:
    dataset_name: str = "tatsu-lab/alpaca"
    split: str = "train"
    loader: str = "grain"
    use_cache: bool = False
    use_fast_tokenizer: bool = True
    shuffle_seed: int = 42
    shuffle_buffer_size: int = 1000
    cache_size: int = 1000
    batch_size: int = 8
    tokenization_batch_size: int = 100

@dataclass
class MockModelConfig:
    maxlen: int = 256
    vocab_size: int = 50257

@dataclass
class MockTrainConfig:
    pass

from dataset import load_text_dataset
from transformers import AutoTokenizer

def get_tokenizer_and_pad_id(m_config: MockModelConfig):
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    m_config.vocab_size = tokenizer.vocab_size
    return tokenizer, tokenizer.pad_token_id

def are_batches_different(batch1, batch2):
    """Helper to check if two batches are different."""
    input1, target1 = batch1
    input2, target2 = batch2

    # Convert to numpy if they are not already
    input1, target1 = np.array(input1), np.array(target1)
    input2, target2 = np.array(input2), np.array(target2)

    return not (np.array_equal(input1, input2) and np.array_equal(target1, target2))

def get_batches(loader: str, seed: int, num_batches: int):
    """Helper to get a specified number of batches for a given loader and seed."""
    d_config = MockDataConfig(loader=loader, shuffle_seed=seed)
    m_config = MockModelConfig()
    t_config = MockTrainConfig()
    _, pad_token_id = get_tokenizer_and_pad_id(m_config)
    
    dataset = load_text_dataset(d_config, m_config, t_config, "gpt2", pad_token_id)
    iterator = iter(dataset)
    
    try:
        return [next(iterator) for _ in range(num_batches)]
    except StopIteration:
        return []

def run_dataset_iteration_test(loader: str):
    """
    Tests that the data loader is iterating over the dataset and not returning the same batch.
    """
    print(f"\n--- Testing loader: {loader} ---")
    d_config = MockDataConfig(loader=loader)
    m_config = MockModelConfig()
    t_config = MockTrainConfig()
    
    tokenizer, pad_token_id = get_tokenizer_and_pad_id(m_config)
    
    dataset = load_text_dataset(d_config, m_config, t_config, "gpt2", pad_token_id)
    
    iterator = iter(dataset)
    num_batches_to_check = 5
    
    try:
        batches = [next(iterator) for _ in range(num_batches_to_check)]
    except StopIteration:
        print(f"FAIL: Dataset iterator for loader '{loader}' exhausted before fetching {num_batches_to_check} batches.")
        return

    all_different = True
    for i in range(len(batches) - 1):
        if not are_batches_different(batches[i], batches[i+1]):
            all_different = False
            print(f"Warning: Batches {i} and {i+1} are identical for loader '{loader}'.")

    if not all_different:
        print(f"FAIL: Some batches were identical for loader '{loader}'. This could indicate a problem with data iteration.")
    else:
        print(f"SUCCESS: All {num_batches_to_check} fetched batches for loader '{loader}' are different.")

    # More detailed check: print shapes and some data
    for i, (input_batch, target_batch) in enumerate(batches):
        print(f"Batch {i}:")
        print(f"  Input shape: {input_batch.shape}")
        print(f"  Target shape: {target_batch.shape}")
        # Print a small slice of data to visually inspect
        print(f"  Input sample (first 5 tokens of first sequence): {input_batch[:5, 0]}")
    
    print(f"--- Test for loader {loader} finished ---")

def test_shuffling_and_seeding(loader: str):
    """
    Tests that shuffling is reproducible with the same seed and different with a new seed.
    """
    print(f"\n--- Testing Shuffling and Seeding for loader: {loader} ---")
    num_batches = 3
    
    # Get batches with seed 42
    batches_seed42_run1 = get_batches(loader, 42, num_batches)
    # Get batches with seed 42 again
    batches_seed42_run2 = get_batches(loader, 42, num_batches)
    
    if not batches_seed42_run1 or not batches_seed42_run2 or len(batches_seed42_run1) < num_batches:
        print(f"FAIL: Could not fetch sufficient batches for seed 42 on loader '{loader}'.")
        return

    # Compare the two sets of batches. They should be identical.
    identical_seeds_are_identical = True
    for b1, b2 in zip(batches_seed42_run1, batches_seed42_run2):
        if are_batches_different(b1, b2):
            identical_seeds_are_identical = False
            break
            
    if identical_seeds_are_identical:
        print("SUCCESS: Two runs with the same seed produced identical batches.")
    else:
        print("FAIL: Two runs with the same seed produced different batches.")

    # Get batches with a different seed
    batches_seed99 = get_batches(loader, 99, num_batches)
    if not batches_seed99 or len(batches_seed99) < num_batches:
        print(f"FAIL: Could not fetch sufficient batches for seed 99 on loader '{loader}'.")
        return

    # Compare batches from different seeds. They should be different.
    different_seeds_are_different = False
    if are_batches_different(batches_seed42_run1[0], batches_seed99[0]):
        different_seeds_are_different = True
            
    if different_seeds_are_different:
        print("SUCCESS: Two runs with different seeds produced different batches.")
    else:
        print("FAIL: Two runs with different seeds produced identical batches.")
    
    print(f"--- Shuffling test for loader {loader} finished ---")


def test_performance(loader: str):
    """
    Benchmarks the data loading performance.
    """
    print(f"\n--- Testing Performance for loader: {loader} ---")
    d_config = MockDataConfig(loader=loader, batch_size=32)
    m_config = MockModelConfig()
    t_config = MockTrainConfig()
    
    _, pad_token_id = get_tokenizer_and_pad_id(m_config)
    
    dataset = load_text_dataset(d_config, m_config, t_config, "gpt2", pad_token_id)
    iterator = iter(dataset)
    
    num_batches_to_benchmark = 100
    print(f"Fetching {num_batches_to_benchmark} batches with batch size {d_config.batch_size}...")
    
    start_time = time.time()
    
    batches_fetched = 0
    try:
        for _ in range(num_batches_to_benchmark):
            next(iterator)
            batches_fetched += 1
    except StopIteration:
        print(f"Warning: Dataset exhausted after {batches_fetched} batches.")
        
    end_time = time.time()
    
    duration = end_time - start_time
    if duration > 0 and batches_fetched > 0:
        batches_per_sec = batches_fetched / duration
        examples_per_sec = (batches_fetched * d_config.batch_size) / duration
        print(f"SUCCESS: Fetched {batches_fetched} batches in {duration:.2f} seconds.")
        print(f"  Speed: {batches_per_sec:.2f} batches/sec")
        print(f"  Throughput: {examples_per_sec:.2f} examples/sec")
    elif batches_fetched == 0:
        print("FAIL: Could not fetch any batches to benchmark.")
    else:
        print("FAIL: Test completed too quickly to measure performance.")
        
    print(f"--- Performance test for loader {loader} finished ---")


def test_data_integrity(loader: str):
    """
    Checks the integrity of a single batch - shapes, dtypes, and content relationship.
    """
    print(f"\n--- Testing Data Integrity for loader: {loader} ---")
    d_config = MockDataConfig(loader=loader)
    m_config = MockModelConfig()
    t_config = MockTrainConfig()
    
    _, pad_token_id = get_tokenizer_and_pad_id(m_config)
    
    dataset = load_text_dataset(d_config, m_config, t_config, "gpt2", pad_token_id)
    
    try:
        input_batch, target_batch = next(iter(dataset))
    except StopIteration:
        print(f"FAIL: Could not fetch a batch from loader '{loader}'.")
        return

    # Convert to numpy for consistent checking
    input_batch, target_batch = np.array(input_batch), np.array(target_batch)

    # 1. Check shapes
    expected_shape = (m_config.maxlen, d_config.batch_size)
    if input_batch.shape == expected_shape and target_batch.shape == expected_shape:
        print(f"SUCCESS: Batch shapes are correct: {expected_shape}")
    else:
        print(f"FAIL: Incorrect batch shapes. Got input={input_batch.shape}, target={target_batch.shape}. Expected {expected_shape}.")
        return

    # 2. Check content relationship (target is shifted input)
    # Compare a single sequence from the batch
    input_sequence = input_batch[:, 0]
    target_sequence = target_batch[:, 0]
    
    expected_target_head = input_sequence[1:]
    actual_target_head = target_sequence[:-1]

    if np.array_equal(expected_target_head, actual_target_head):
        print("SUCCESS: Target is correctly shifted from input.")
    else:
        print("FAIL: Target is NOT a correct shift of the input.")
        print(f"  Input sequence (sample): {input_sequence[:10]}...")
        print(f"  Target sequence (sample): {target_sequence[:10]}...")

    # 3. Check padding token in target
    if target_sequence[-1] == pad_token_id:
        print(f"SUCCESS: Target sequence is correctly padded with pad_token_id ({pad_token_id}).")
    else:
        print(f"FAIL: Last token of target sequence is {target_sequence[-1]}, expected pad_token_id {pad_token_id}.")

    print(f"--- Data Integrity test for loader {loader} finished ---")


if __name__ == "__main__":
    loaders_to_test = ["grain", "tf"]
    for loader in loaders_to_test:
        run_dataset_iteration_test(loader)
        test_shuffling_and_seeding(loader)
        test_data_integrity(loader)
        test_performance(loader)
        print("\n" + "="*50 + "\n") 