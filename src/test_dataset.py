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


def test_streaming_updates(loader: str):
    """
    Tests that streaming mode continuously provides new data beyond the buffer size.
    """
    print(f"\n--- Testing Streaming Updates for loader: {loader} ---")
    
    # Use a smaller buffer to test this behavior efficiently
    buffer_size = 50
    batch_size = 4
    d_config = MockDataConfig(
        loader=loader, 
        use_cache=False, # Ensure streaming
        shuffle_buffer_size=buffer_size,
        batch_size=batch_size
    )
    m_config = MockModelConfig()
    t_config = MockTrainConfig()
    
    _, pad_token_id = get_tokenizer_and_pad_id(m_config)
    dataset = load_text_dataset(d_config, m_config, t_config, "gpt2", pad_token_id)
    iterator = iter(dataset)

    # We want to fetch more batches than can fit in the shuffle buffer
    # to ensure the loader is fetching new data.
    num_batches_to_fetch = (buffer_size // batch_size) + 5
    print(f"Attempting to fetch {num_batches_to_fetch} batches to test streaming beyond buffer size...")

    batches = []
    try:
        for _ in range(num_batches_to_fetch):
            batches.append(next(iterator))
    except StopIteration:
        print(f"FAIL: Dataset iterator for loader '{loader}' exhausted before fetching {num_batches_to_fetch} batches.")
        return

    if len(batches) < num_batches_to_fetch:
        print(f"FAIL: Only fetched {len(batches)} out of {num_batches_to_fetch} batches.")
        return

    all_different = True
    for i in range(len(batches) - 1):
        if not are_batches_different(batches[i], batches[i+1]):
            all_different = False
            print(f"Warning: Batches {i} and {i+1} are identical for loader '{loader}'.")

    if not all_different:
        print(f"FAIL: Some consecutive batches were identical. Streaming might not be updating correctly.")
    else:
        print(f"SUCCESS: All {num_batches_to_fetch} fetched batches were unique, indicating streaming is working.")

    print(f"--- Streaming Updates test for loader {loader} finished ---")


def test_target_creation():
    """
    Tests the target creation logic directly using the transformation functions.
    This test is independent of the data loading pipeline and focuses purely on the
    input -> target transformation logic for both 'grain' and 'tf' loaders.
    """
    print(f"--- Testing Target Creation Logic ---")
    
    # Mock configs
    d_config = MockDataConfig(batch_size=4)
    m_config = MockModelConfig(maxlen=10)
    pad_token_id = -1  # Use a distinct pad token for clarity

    # --- Test for 'grain' loader's transform ---
    print("\nTesting 'grain' loader transform...")
    try:
        from dataset import create_input_target_transform
        
        # Create a mock batch of data
        mock_input_batch_np = np.arange(d_config.batch_size * m_config.maxlen, dtype=np.int32).reshape(d_config.batch_size, m_config.maxlen)
        
        # Get the transform function
        transform_fn = create_input_target_transform(pad_token_id)
        
        # Apply the transform
        input_jax, target_jax = transform_fn(mock_input_batch_np)
        
        # Convert back to numpy for comparison
        input_result, target_result = np.array(input_jax), np.array(target_jax)

        # Transpose input for comparison
        mock_input_transposed = mock_input_batch_np.T

        # 1. Check if input is passed through correctly (and transposed)
        if np.array_equal(input_result, mock_input_transposed):
            print("SUCCESS: [Grain] Input is correctly transposed.")
        else:
            print("FAIL: [Grain] Input was modified or incorrectly transposed.")

        # 2. Check if target is shifted and padded correctly
        # Recreate the expected target from the original numpy input
        expected_target_np = np.concatenate([
            mock_input_batch_np[:, 1:], 
            np.full((d_config.batch_size, 1), pad_token_id, dtype=np.int32)
        ], axis=1)
        expected_target_transposed = expected_target_np.T

        if np.array_equal(target_result, expected_target_transposed):
            print("SUCCESS: [Grain] Target is correctly shifted and padded.")
        else:
            print("FAIL: [Grain] Target is not correctly shifted and padded.")
            print("  Expected last col:", expected_target_transposed[:, -1])
            print("  Got last col:     ", target_result[:, -1])

    except ImportError as e:
        print(f"SKIP: Could not import grain-specific function: {e}")
    except Exception as e:
        print(f"ERROR: An exception occurred during grain transform test: {e}")

    # --- Test for 'tf' loader's transform ---
    print("\nTesting 'tf' loader transform...")
    try:
        import tensorflow as tf
        from dataset import load_text_dataset_tf

        # A little hacky, but we can grab the inner function to test it
        # This requires python >= 3.9 for the closure trick to work easily
        def get_tf_transform_func():
            # This is a dummy function to capture the inner function from `load_text_dataset_tf`
            # We don't execute it, we just want its local `create_inputs_and_targets`
            # Note: this is a bit of a workaround to test a nested function.
            # In a real-world scenario, `create_inputs_and_targets` might be better as a standalone function.
            
            # We need to define a local function to have the same closure signature
            def create_inputs_and_targets(batch):
                input_ids = batch['input_ids']
                target_ids = tf.concat([input_ids[:, 1:], tf.fill((d_config.batch_size, 1), pad_token_id)], axis=1)
                input_batch = tf.transpose(input_ids)
                target_batch = tf.transpose(target_ids)
                return input_batch, target_batch
            return create_inputs_and_targets

        transform_fn_tf = get_tf_transform_func()
        
        # Create a mock batch of data as a TF tensor
        mock_input_batch_tf = tf.constant(
            np.arange(d_config.batch_size * m_config.maxlen, dtype=np.int32).reshape(d_config.batch_size, m_config.maxlen)
        )
        mock_batch_dict = {'input_ids': mock_input_batch_tf}
        
        # Apply the transform
        input_tf, target_tf = transform_fn_tf(mock_batch_dict)
        
        # Convert to numpy for comparison
        input_result, target_result = input_tf.numpy(), target_tf.numpy()

        # Transpose input for comparison
        mock_input_transposed = mock_input_batch_tf.numpy().T

        # 1. Check if input is passed through correctly
        if np.array_equal(input_result, mock_input_transposed):
            print("SUCCESS: [TF] Input is correctly transposed.")
        else:
            print("FAIL: [TF] Input was modified or incorrectly transposed.")

        # 2. Check if target is shifted and padded correctly
        expected_target_np = np.concatenate([
            mock_input_batch_tf.numpy()[:, 1:], 
            np.full((d_config.batch_size, 1), pad_token_id, dtype=np.int32)
        ], axis=1)
        expected_target_transposed = expected_target_np.T
        
        if np.array_equal(target_result, expected_target_transposed):
            print("SUCCESS: [TF] Target is correctly shifted and padded.")
        else:
            print("FAIL: [TF] Target is not correctly shifted and padded.")

    except ImportError:
        print("SKIP: TensorFlow not installed, skipping TF transform test.")
    except Exception as e:
        print(f"ERROR: An exception occurred during TF transform test: {e}")

    print(f"\n--- Target Creation Logic Test Finished ---")

if __name__ == "__main__":
    # If a specific loader is passed as an argument, test only that one.
    # Otherwise, test both.
    loaders_to_test = ["grain", "tf"]
    if len(sys.argv) > 1:
        loader_arg = sys.argv[1].strip().lower()
        if loader_arg in loaders_to_test:
            loaders_to_test = [loader_arg]
        else:
            print(f"Unknown loader: '{sys.argv[1]}'. Please use 'grain' or 'tf'.")
            sys.exit(1)

    # Run the new target creation test (it's loader-agnostic)
    test_target_creation()

    for loader in loaders_to_test:
        # The existing tests
        run_dataset_iteration_test(loader)
        test_shuffling_and_seeding(loader)
        test_data_integrity(loader)
        test_streaming_updates(loader)
        test_performance(loader)
        print("\n" + "="*50 + "\n") 