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
    """Efficient data source for streaming datasets."""
    
    def __init__(self, dataset_name: str, split: str, tokenizer_name: str, max_length: int):
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        
        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset in streaming mode
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10_000)
        
        # Convert to list for random access (cache a reasonable amount)
        self._cache_size = 100_000  # Adjust based on memory
        self._data_cache = []
        self._populate_cache()
    
    def _populate_cache(self):
        """Populate cache with tokenized data."""
        print(f"Populating cache with {self._cache_size} examples...")
        for i, example in enumerate(self.dataset):
            if i >= self._cache_size:
                break
            
            # Tokenize directly to numpy arrays
            tokens = self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="np"
            )
            
            self._data_cache.append({
                'input_ids': tokens['input_ids'].squeeze(0).astype(np.int32)
            })
        
        print(f"Cache populated with {len(self._data_cache)} examples")
    
    def __len__(self):
        return len(self._data_cache)
    
    def __getitem__(self, index):
        return self._data_cache[index % len(self._data_cache)]

def create_input_target_transform(pad_token_id: int):
    """Transform that creates input/target pairs efficiently."""
    
    def transform(batch: Dict[str, np.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        input_ids = batch['input_ids']  # Shape: (batch_size, seq_len)
        
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
        max_length=m_config.maxlen
    )
    
    # Create sampler with shuffling
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=True,
        seed=42,
        num_epochs=t_config.num_epochs
    )
    
    # Create data loader with optimizations
    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        worker_count=4,  # Adjust based on CPU cores
        worker_buffer_size=100,  # Prefetch buffer per worker
        read_options=grain.ReadOptions(
            num_threads=2,  # Threads per worker for I/O
            prefetch_buffer_size=50
        )
    )
    
    # Apply batching
    loader = loader.batch(d_config.batch_size, drop_remainder=True)
    
    # Apply input/target transformation
    transform = create_input_target_transform(pad_token_id)
    loader = loader.map(transform)
    
    return loader

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
    hf_dataset = hf_dataset.shuffle(seed=42, buffer_size=10_000)

    @lru_cache(maxsize=None)
    def get_tokenizer(name):
        """
        Loads and caches the tokenizer.
        Each worker process will have its own cached tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)  # Use fast tokenizer
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
        batch_size=1000,  # Larger batch size for tokenization
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