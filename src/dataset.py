import jax
import jax.numpy as jnp
import grain.python as grain
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import lru_cache
import numpy as np
from typing import Iterator, Dict, Any
import threading

from config import DataConfig, ModelConfig, TrainConfig

class TextDataSource(grain.RandomAccessDataSource):
    """Efficient data source for streaming datasets with dynamic cache updates."""
    
    def __init__(self, dataset_name: str, split: str, tokenizer_name: str, max_length: int, d_config: DataConfig):
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        self.d_config = d_config
        
        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=d_config.use_fast_tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset in streaming mode
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.dataset = self.dataset.shuffle(seed=d_config.shuffle_seed, buffer_size=d_config.shuffle_buffer_size)
        
        # Create an iterator for the dataset
        self.dataset_iter = iter(self.dataset)
        
        # Dynamic cache configuration
        self._cache_size = d_config.cache_size
        self._data_cache = []
        self._total_accessed = 0
        self._cache_generation = 0
        self._lock = threading.Lock()
        
        # Populate initial cache
        self._populate_cache()
    
    def _populate_cache(self):
        """Populate cache with tokenized data."""
        print(f"Populating cache (generation {self._cache_generation}) with {self._cache_size} examples...")
        self._data_cache.clear()
        
        examples_added = 0
        try:
            for example in self.dataset_iter:
                if examples_added >= self._cache_size:
                    break
                
                # Tokenize directly to numpy arrays
                tokens = self.tokenizer(
                    example["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="np"
                )
                
                # Store just the input_ids as a flat array
                self._data_cache.append(tokens['input_ids'].squeeze(0).astype(np.int32))
                examples_added += 1
                
        except StopIteration:
            # If we run out of data, recreate the iterator
            print("Reached end of dataset, reshuffling and recreating iterator...")
            # Reshuffle with a new seed to ensure different data ordering in next pass
            self.dataset = self.dataset.shuffle(seed=self.d_config.shuffle_seed + self._cache_generation, buffer_size=self.d_config.shuffle_buffer_size)
            self.dataset_iter = iter(self.dataset)
            # Try to fill the remaining cache
            for example in self.dataset_iter:
                if examples_added >= self._cache_size:
                    break
                
                tokens = self.tokenizer(
                    example["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="np"
                )
                
                self._data_cache.append(tokens['input_ids'].squeeze(0).astype(np.int32))
                examples_added += 1
        
        print(f"Cache populated with {len(self._data_cache)} examples")
        self._cache_generation += 1
    
    def __len__(self):
        return len(self._data_cache)
    
    def __getitem__(self, index):
        with self._lock:
            self._total_accessed += 1
            
            # Periodically refresh the entire cache to see new data
            if self._total_accessed > 0 and self._total_accessed % self._cache_size == 0:
                self._populate_cache()
            
            # Ensure cache is not empty before accessing
            if not self._data_cache:
                raise IndexError("Data cache is empty. Possible exhaustion of the dataset.")
            
            return self._data_cache[index % len(self._data_cache)]

    def get_cache_stats(self):
        """Get statistics about the cache for debugging."""
        return {
            'cache_size': len(self._data_cache),
            'total_accessed': self._total_accessed,
            'cache_generation': self._cache_generation,
        }

# This is a streaming data source for Grain
class StreamingTextDataSource(grain.RandomAccessDataSource):
    """A streaming data source for Grain that tokenizes on the fly."""
    def __init__(self, dataset_name: str, split: str, tokenizer_name: str, max_length: int, d_config: DataConfig):
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        self.d_config = d_config
        self._generation = 0

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=d_config.use_fast_tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset in streaming mode and create an iterator
        self.hf_dataset = load_dataset(dataset_name, split=split, streaming=True)
        self._reseed_and_reshuffle()
        
        # Internal buffer to hold a chunk of the dataset
        self.buffer = []
        self._populate_buffer()
    
    def _reseed_and_reshuffle(self):
        """Reshuffles the dataset with a new seed."""
        # Use a new seed for each pass through the data
        seed = self.d_config.shuffle_seed + self._generation
        print(f"Reshuffling dataset with seed {seed}")
        self.hf_dataset = self.hf_dataset.shuffle(seed=seed, buffer_size=self.d_config.shuffle_buffer_size)
        self.hf_iterator = iter(self.hf_dataset)
        self._generation += 1

    def _populate_buffer(self):
        """Fills the internal buffer with new examples from the dataset iterator."""
        self.buffer.clear()
        try:
            for _ in range(self.d_config.shuffle_buffer_size):
                self.buffer.append(next(self.hf_iterator))
        except StopIteration:
            print("Dataset iterator exhausted. Reshuffling...")
            self._reseed_and_reshuffle()
            # Try to populate again after reshuffling
            for _ in range(self.d_config.shuffle_buffer_size):
                try:
                    self.buffer.append(next(self.hf_iterator))
                except StopIteration:
                    # If it's still exhausted, the dataset is likely smaller than buffer size
                    break
        
        if not self.buffer:
            raise IndexError("Unable to populate data buffer. The dataset might be empty or too small.")

    def __len__(self):
        # The effective length is the size of our current buffer
        return len(self.buffer)

    def __getitem__(self, index):
        # When grain asks for an index, it's from the current buffer.
        # If index is out of bounds, it means grain has exhausted our buffer and we need to refill it.
        # This simplistic check might not be robust for all of grain's access patterns,
        # but for a standard sequential iteration it works.
        if index >= len(self.buffer):
            self._populate_buffer()
        
        # Get example from the in-memory buffer
        example = self.buffer[index]
        tokens = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np"
        )
        return tokens['input_ids'].squeeze(0).astype(np.int32)

def create_input_target_transform(pad_token_id: int):
    """Transform that creates input/target pairs efficiently."""
    
    def transform(batch):
        # batch is now a numpy array of shape (batch_size, seq_len)
        input_ids = np.array(batch)
        
        # Create targets by shifting input (vectorized operation)
        targets = np.concatenate([
            input_ids[:, 1:], 
            np.full((input_ids.shape[0], 1), pad_token_id, dtype=np.int32)
        ], axis=1)
        
        # Convert to JAX arrays and transpose for model expectations
        # Model expects (seq_len, batch_size)
        input_batch = jnp.array(input_ids).T
        target_batch = jnp.array(targets).T
        
        return input_batch, target_batch
    
    return transform

def load_text_dataset(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer_name: str, pad_token_id: int):
    """Loads dataset using the loader specified in the config."""
    loader_name = str(d_config.loader).strip().lower()

    if loader_name == 'grain':
        return load_text_dataset_grain(d_config, m_config, t_config, tokenizer_name, pad_token_id)
    elif loader_name == 'tf':
        return load_text_dataset_tf(d_config, m_config, t_config, tokenizer_name, pad_token_id)
    else:
        raise ValueError(f"Unknown data loader: '{d_config.loader}'. Must be 'grain' or 'tf'.")

def load_text_dataset_grain(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer_name: str, pad_token_id: int):
    """
    Loads and prepares a text dataset for training with JAX using Grain.
    - Uses JAX-native data loading for better performance
    - Efficient tokenization and batching
    - Optimized memory usage
    """
    
    # Create data source based on whether caching is enabled
    if d_config.use_cache:
        print("Using caching Grain data source.")
        data_source = TextDataSource(
            dataset_name=d_config.dataset_name,
            split=d_config.split,
            tokenizer_name=tokenizer_name,
            max_length=m_config.maxlen,
            d_config=d_config
        )
    else:
        print("Using streaming Grain data source.")
        data_source = StreamingTextDataSource(
            dataset_name=d_config.dataset_name,
            split=d_config.split,
            tokenizer_name=tokenizer_name,
            max_length=m_config.maxlen,
            d_config=d_config
        )

    # Create input/target transformation
    transform = create_input_target_transform(pad_token_id)
    
    # Create dataset using Grain's chaining API
    dataset = (
        grain.MapDataset.source(data_source)
        .shuffle(seed=d_config.shuffle_seed)
        .batch(batch_size=d_config.batch_size)
        .map(transform)
    )
    
    return dataset

# Main function for TensorFlow-based loading
def load_text_dataset_tf(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer_name: str, pad_token_id: int):
    """
    TensorFlow-based data loading pipeline, that works with older and newer versions of `datasets`.
    Supports both streaming and shuffling.
    """
    import tensorflow as tf
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=d_config.use_fast_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # streaming=True returns an IterableDataset
    hf_dataset = load_dataset(d_config.dataset_name, split=d_config.split, streaming=True)

    def tokenize_function(examples):
        # We'll handle tensor conversion in the generator.
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=m_config.maxlen,
        )

    # The map function is applied on-the-fly.
    tokenized_dataset = hf_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"]
    )
    
    # We only need 'input_ids' for the model input.
    # The 'attention_mask' is also available if your model uses it.
    tokenized_dataset = tokenized_dataset.remove_columns(["attention_mask"])

    # Use a generator to feed data to tf.data.Dataset, which is compatible with IterableDataset
    def data_generator():
        for example in tokenized_dataset:
            yield {'input_ids': example['input_ids']}

    # Define the output signature for the generator
    output_signature = {
        'input_ids': tf.TensorSpec(shape=(m_config.maxlen,), dtype=tf.int32)
    }

    tf_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )

    # Enable caching if specified in config
    if d_config.use_cache:
        print("Enabling tf.data caching.")
        tf_dataset = tf_dataset.cache()

    # Shuffle, batch, and create targets
    if d_config.shuffle_buffer_size and d_config.shuffle_buffer_size > 0:
        tf_dataset = tf_dataset.shuffle(d_config.shuffle_buffer_size)
    
    tf_dataset = tf_dataset.batch(d_config.batch_size, drop_remainder=True)

    def create_inputs_and_targets(batch):
        input_ids = batch['input_ids']
        # Create target by shifting input
        target_ids = tf.concat([input_ids[:, 1:], tf.fill((d_config.batch_size, 1), pad_token_id)], axis=1)
        # The model expects (maxlen, batch_size), so we transpose.
        input_batch = tf.transpose(input_ids)
        target_batch = tf.transpose(target_ids)
        return input_batch, target_batch

    tf_dataset = tf_dataset.map(
        create_inputs_and_targets, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return tf_dataset 