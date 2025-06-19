import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer

from config import DataConfig, ModelConfig, TrainConfig

def load_text_dataset(d_config: DataConfig, m_config: ModelConfig, t_config: TrainConfig, tokenizer: AutoTokenizer):
    """
    Loads and prepares a text dataset for training with JAX.
    - Uses streaming for large datasets.
    - Tokenizes and batches data efficiently.
    - Returns a tf.data.Dataset.
    """
    hf_dataset = load_dataset(d_config.dataset_name, split=d_config.split, streaming=True)

    def tokenize_function(examples):
        # The tokenizer now returns TensorFlow tensors.
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=m_config.maxlen,
            return_tensors="tf",
        )

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    
    # We only need 'input_ids' for the model input.
    # The 'attention_mask' is also available if your model uses it.
    tokenized_dataset = tokenized_dataset.remove_columns(["text", "attention_mask"])

    # Convert to a tf.data.Dataset
    tf_dataset = tokenized_dataset.to_tf_dataset(
        columns=["input_ids"],
        batch_size=d_config.batch_size,
        shuffle=True, # Shuffle the dataset. For streaming, this uses a buffer.
        prefetch=tf.data.experimental.AUTOTUNE,
        drop_remainder=True,
    )
    
    return tf_dataset 