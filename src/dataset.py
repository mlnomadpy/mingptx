import jax
import jax.numpy as jnp
import grain.python as grain
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import lru_cache
import numpy as np
from typing import Iterator, Dict, Any

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
        self._cache_refresh_rate = d_config.cache_refresh_rate
        self._cache_refresh_interval = d_config.cache_refresh_interval
        self._total_accessed = 0
        self._cache_generation = 0
        
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
            print("Reached end of dataset, recreating iterator...")
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
    
    def _refresh_cache_partially(self):
        """Refresh a portion of the cache with new data."""
        if len(self._data_cache) == 0:
            self._populate_cache()
            return
            
        refresh_count = min(self._cache_refresh_rate, len(self._data_cache))
        print(f"Refreshing {refresh_count} cache entries...")
        
        # Remove old entries from random positions
        import random
        random.seed(self.d_config.shuffle_seed + self._cache_generation)
        indices_to_replace = random.sample(range(len(self._data_cache)), refresh_count)
        
        # Add new entries
        new_entries = []
        try:
            for _ in range(refresh_count):
                example = next(self.dataset_iter)
                tokens = self.tokenizer(
                    example["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="np"
                )
                new_entries.append(tokens['input_ids'].squeeze(0).astype(np.int32))
        except StopIteration:
            # Recreate iterator if we reach the end
            self.dataset_iter = iter(self.dataset)
            remaining = refresh_count - len(new_entries)
            for _ in range(remaining):
                try:
                    example = next(self.dataset_iter)
                    tokens = self.tokenizer(
                        example["text"],
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="np"
                    )
                    new_entries.append(tokens['input_ids'].squeeze(0).astype(np.int32))
                except StopIteration:
                    break
        
        # Replace old entries with new ones
        for i, new_entry in zip(indices_to_replace[:len(new_entries)], new_entries):
            self._data_cache[i] = new_entry
    
    def __len__(self):
        return len(self._data_cache)
    
    def __getitem__(self, index):
        self._total_accessed += 1
        
        # Periodically refresh part of the cache to see new data
        if self._total_accessed % (self._cache_size * self._cache_refresh_interval) == 0:
            self._refresh_cache_partially()
        
        return self._data_cache[index % len(self._data_cache)]

    def get_cache_stats(self):
        """Get statistics about the cache for debugging."""
        return {
            'cache_size': len(self._data_cache),
            'total_accessed': self._total_accessed,
            'cache_generation': self._cache_generation,
            'refresh_rate': self._cache_refresh_rate,
            'refresh_interval': self._cache_refresh_interval
        }

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
    """
    Loads and prepares a text dataset for training with JAX using Grain.
    - Uses JAX-native data loading for better performance
    - Efficient tokenization and batching
    - Optimized memory usage
    """
    
    # Create data source
    data_source = TextDataSource(
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
    
    # Convert to iterable dataset for training
    iter_dataset = dataset.to_iter_dataset(
        grain.ReadOptions(num_threads=d_config.num_threads, prefetch_buffer_size=d_config.prefetch_buffer_size)
    )
    
    return iter_dataset

# Backward compatibility function for TensorFlow-based loading (if needed)
def load_text_dataset_tf_fallback(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer_name: str, pad_token_id: int):
    """
    Fallback to TensorFlow-based loading if Grain has issues.
    This is the original implementation with some optimizations.
    """
    import tensorflow as tf
    
    # streaming=True returns an IterableDataset
    hf_dataset = load_dataset(d_config.dataset_name, split=d_config.split, streaming=True)

    # Shuffle the dataset. For streaming, this uses a buffer of elements.
    hf_dataset = hf_dataset.shuffle(seed=d_config.shuffle_seed, buffer_size=d_config.shuffle_buffer_size)

    @lru_cache(maxsize=None)
    def get_tokenizer(name):
        """
        Loads and caches the tokenizer.
        Each worker process will have its own cached tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=d_config.use_fast_tokenizer)  # Use config value
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def tokenize_function(examples):
        # The tokenizer returns TensorFlow tensors.
        tokenizer = get_tokenizer(tokenizer_name)
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=m_config.maxlen,
            return_tensors="tf",
        )

    # The map function is applied on-the-fly to batches of examples.
    tokenized_dataset = hf_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=d_config.tokenization_batch_size,  # Use config value instead of hardcoded 1000
        remove_columns=hf_dataset.column_names
    )
    
    # We only need 'input_ids' for the model, so we remove other columns.
    # The training script expects batches with 'input_ids'.
    tokenized_dataset = tokenized_dataset.remove_columns(["attention_mask"])

    # Create a tf.data.Dataset from the Hugging Face IterableDataset.
    def data_generator():
        for record in tokenized_dataset:
            yield record

    # The output from the generator will be a dictionary. The training loop
    # expects 'input_ids'.
    output_signature = {
        'input_ids': tf.TensorSpec(shape=(m_config.maxlen,), dtype=tf.int64)
    }

    tf_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )

    # Batch and prefetch the dataset for performance.
    tf_dataset = tf_dataset.batch(d_config.batch_size, drop_remainder=True)
    
    # Create the padding tensor once to avoid overhead in the map function.
    pad_tensor = tf.constant(pad_token_id, shape=(d_config.batch_size, 1), dtype=tf.int64)

    def create_inputs_and_targets(batch):
        input_ids = batch['input_ids']
        # Create target by shifting input, reusing the pre-made pad_tensor.
        target_ids = tf.concat([input_ids[:, 1:], pad_tensor], axis=1)
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