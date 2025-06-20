import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import lru_cache

from config import DataConfig, ModelConfig, TrainConfig

def load_text_dataset(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer_name: str):
    """
    Loads and prepares a text dataset for training with JAX.
    - Uses streaming for large datasets.
    - Tokenizes and batches data efficiently.
    - Returns a tf.data.Dataset.
    """
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
        tokenizer = AutoTokenizer.from_pretrained(name)
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
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return tf_dataset 