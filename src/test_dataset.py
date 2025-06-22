import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import sys

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
    return not (np.array_equal(input1, input2) and np.array_equal(target1, target2))

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

if __name__ == "__main__":
    run_dataset_iteration_test("grain")
    run_dataset_iteration_test("tf") 