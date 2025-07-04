import jax
import jax.numpy as jnp
import grain.python as grain
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import lru_cache
import numpy as np
from typing import Iterator, Dict, Any
import threading
import tiktoken

from config import DataConfig, ModelConfig, TrainConfig

def encode_with_tiktoken(text, tokenizer, max_length):
    """Encodes text using tiktoken, handling padding and truncation."""
    tokens = tokenizer.encode(text)
    tokens = tokens[:max_length]  # Truncate
    
    padding_needed = max_length - len(tokens)
    padded_tokens = tokens + [tokenizer.eot_token] * padding_needed
    
    attention_mask = [1] * len(tokens) + [0] * padding_needed
    
    return {
        'input_ids': np.array(padded_tokens, dtype=np.int32),
        'attention_mask': np.array(attention_mask, dtype=np.int32)
    }

# This is a streaming data source for Grain
class StreamingTextDataSource(grain.RandomAccessDataSource):
    """A streaming data source for Grain that tokenizes on the fly."""
    def __init__(self, dataset_name: str, split: str, tokenizer: Any, max_length: int, d_config: DataConfig):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.d_config = d_config
        self._generation = 0

        # Check tokenizer type
        self.is_tiktoken = isinstance(self.tokenizer, tiktoken.Encoding)

        if not self.is_tiktoken:
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

        if self.is_tiktoken:
            return encode_with_tiktoken(example["text"], self.tokenizer, self.max_length)
        else:
            tokens = self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="np"
            )
            return {
                'input_ids': tokens['input_ids'].squeeze(0).astype(np.int32),
                'attention_mask': tokens['attention_mask'].squeeze(0).astype(np.int32)
            }

def create_input_target_transform(pad_token_id: int):
    """Transform that returns input ids and attention masks, targets created in train."""
    
    def transform(batch):
        # batch is now a list of dicts with 'input_ids' and 'attention_mask'
        input_ids = np.array([item['input_ids'] for item in batch])
        attention_mask = np.array([item['attention_mask'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    return transform

def load_text_dataset(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer: Any, pad_token_id: int):
    """Loads dataset using the loader specified in the config."""
    loader_name = str(d_config.loader).strip().lower()

    if loader_name == 'grain':
        return load_text_dataset_grain(d_config, m_config, t_config, tokenizer, pad_token_id)
    elif loader_name == 'tf':
        return load_text_dataset_tf(d_config, m_config, t_config, tokenizer, pad_token_id)
    else:
        raise ValueError(f"Unknown data loader: '{d_config.loader}'. Must be 'grain' or 'tf'.")

def load_text_dataset_grain(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer: Any, pad_token_id: int):
    """
    Loads and prepares a text dataset for training with JAX using Grain.
    - Uses JAX-native data loading for better performance
    - Efficient tokenization and batching
    - Optimized memory usage
    """
    
    # Create data source based on whether caching is enabled
    print("Using streaming Grain data source.")
    data_source = StreamingTextDataSource(
        dataset_name=d_config.dataset_name,
        split=d_config.split,
        tokenizer=tokenizer,
        max_length=m_config.maxlen,
        d_config=d_config
    )

    # Create input/target transformation
    transform = create_input_target_transform(pad_token_id)
    
    # Create dataset using Grain's chaining API
    dataset = (
        grain.MapDataset.source(data_source)
        # .shuffle(seed=d_config.shuffle_seed)  # NOTE: This shuffles a small buffer and is incorrect. The source is shuffled, but the buffer logic is flawed.
        .batch(batch_size=d_config.batch_size)
        .map(transform)
    )
    
    return dataset

# Main function for TensorFlow-based loading
def load_text_dataset_tf(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer: Any, pad_token_id: int):
    """
    TensorFlow-based data loading pipeline, that works with older and newer versions of `datasets`.
    Supports both streaming and shuffling.
    """
    import tensorflow as tf
    from transformers import AutoTokenizer
    is_tiktoken = isinstance(tokenizer, tiktoken.Encoding)

    # streaming=True returns an IterableDataset
    hf_dataset = load_dataset(d_config.dataset_name, split=d_config.split, streaming=True)

    def tokenize_function(examples):
        if is_tiktoken:
            # We need to process examples one by one for tiktoken
            tokenized_examples = [encode_with_tiktoken(text, tokenizer, m_config.maxlen) for text in examples["text"]]
            # And then stack them
            return {
                'input_ids': np.array([ex['input_ids'] for ex in tokenized_examples]),
                'attention_mask': np.array([ex['attention_mask'] for ex in tokenized_examples])
            }
        else:
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
    
    # Keep both 'input_ids' and 'attention_mask' for proper masking
    # Note: We now keep the attention_mask instead of removing it

    # Use a generator to feed data to tf.data.Dataset, which is compatible with IterableDataset
    def data_generator():
        for example in tokenized_dataset:
            yield {
                'input_ids': example['input_ids'],
                'attention_mask': example['attention_mask']
            }

    # Define the output signature for the generator
    output_signature = {
        'input_ids': tf.TensorSpec(shape=(m_config.maxlen,), dtype=tf.int32),
        'attention_mask': tf.TensorSpec(shape=(m_config.maxlen,), dtype=tf.int32)
    }

    tf_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )

    # Shuffle, batch, and create targets
    if d_config.shuffle_buffer_size and d_config.shuffle_buffer_size > 0:
        tf_dataset = tf_dataset.shuffle(d_config.shuffle_buffer_size)
    
    tf_dataset = tf_dataset.batch(d_config.batch_size, drop_remainder=True)

    def extract_inputs_and_masks(batch):
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }

    tf_dataset = tf_dataset.map(extract_inputs_and_masks, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return tf_dataset 