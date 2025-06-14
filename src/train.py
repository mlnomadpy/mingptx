import os
import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint as orbax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from transformers import AutoTokenizer
import argparse

from config import ProjectConfig, ModelConfig, DataConfig, TrainConfig
from dataset import load_text_dataset
from model import MiniGPT
from log import Logger, visualize_and_log_loss, flatten_for_logging, get_flat_determinants

def setup_mesh():
    devices = jax.devices()
    num_devices = len(devices)
    if num_devices > 1:
        # For multi-device setup, create a 2D mesh, e.g., for data and model parallelism
        # This is a general setup, you might need to tune the mesh shape
        mesh_shape = (jax.local_device_count(), num_devices // jax.local_device_count())
        return Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Train a mini-GPT model.")
    
    default_config = ProjectConfig()

    # Model args
    parser.add_argument("--maxlen", type=int, default=default_config.model_config.maxlen, help="Maximum sequence length.")
    parser.add_argument("--vocab_size", type=int, default=default_config.model_config.vocab_size, help="Vocabulary size.")
    parser.add_argument("--embed_dim", type=int, default=default_config.model_config.embed_dim, help="Embedding dimensionality.")
    parser.add_argument("--num_heads", type=int, default=default_config.model_config.num_heads, help="Number of attention heads.")
    parser.add_argument("--feed_forward_dim", type=int, default=default_config.model_config.feed_forward_dim, help="Dimensionality of the feed-forward network.")
    parser.add_argument("--num_transformer_blocks", type=int, default=default_config.model_config.num_transformer_blocks, help="Number of transformer blocks.")
    parser.add_argument("--dropout_rate", type=float, default=default_config.model_config.dropout_rate, help="Dropout rate.")

    # Data args
    parser.add_argument("--dataset_name", type=str, default=default_config.data_config.dataset_name, help="Hugging Face dataset name.")
    parser.add_argument("--split", type=str, default=default_config.data_config.split, help="Dataset split to use.")
    parser.add_argument("--batch_size", type=int, default=default_config.data_config.batch_size, help="Batch size for training.")
    parser.add_argument("--tokenizer_name", type=str, default=default_config.data_config.tokenizer_name, help="Tokenizer to use.")

    # Train args
    parser.add_argument("--num_epochs", type=int, default=default_config.train_config.num_epochs, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=default_config.train_config.learning_rate, help="Learning rate for the optimizer.")
    parser.add_argument("--log_interval", type=int, default=default_config.train_config.log_interval, help="Interval for logging metrics.")
    parser.add_argument("--text_log_interval", type=int, default=default_config.train_config.text_log_interval, help="Interval for logging generated text.")
    parser.add_argument("--use_wandb", type=lambda x: (str(x).lower() == 'true'), default=default_config.train_config.use_wandb, help="Whether to use wandb for logging.")
    parser.add_argument("--checkpoint_dir", type=str, default=default_config.train_config.checkpoint_dir, help="Directory to save checkpoints.")
    
    args = parser.parse_args()
    
    config = ProjectConfig(
        model_config=ModelConfig(
            maxlen=args.maxlen,
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            feed_forward_dim=args.feed_forward_dim,
            num_transformer_blocks=args.num_transformer_blocks,
            dropout_rate=args.dropout_rate
        ),
        data_config=DataConfig(
            dataset_name=args.dataset_name,
            split=args.split,
            batch_size=args.batch_size,
            tokenizer_name=args.tokenizer_name
        ),
        train_config=TrainConfig(
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            log_interval=args.log_interval,
            text_log_interval=args.text_log_interval,
            use_wandb=args.use_wandb,
            checkpoint_dir=args.checkpoint_dir
        )
    )
    return config

def main():
    config = parse_args()
    mesh = setup_mesh()

    logger = Logger(project_name="mingptx", config={
        "model": config.model_config,
        "data": config.data_config,
        "train": config.train_config
    }, use_wandb=config.train_config.use_wandb)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.data_config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    text_dl = load_text_dataset(config.data_config, config.model_config, config.train_config, tokenizer)

    # Create model
    rngs = nnx.Rngs(0)
    model = MiniGPT(config.model_config, mesh, rngs=rngs)
    
    # Optimizer
    optimizer = nnx.Optimizer(model, optax.adam(config.train_config.learning_rate))
    
    # Metrics
    metrics_manager = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    
    # Loss function
    def loss_fn(mdl, batch):
        logits = mdl(batch[0], training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
        return loss, logits

    # Training step
    @nnx.jit
    def train_step(mdl, opt, mets, b):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, _), grads = grad_fn(mdl, b)
        mets.update(loss=loss)
        opt.update(grads)
        grad_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x) if x is not None else 0.0, grads)
        return grad_norms

    # Initial text generation
    start_prompt = "Once upon a time"
    generated_text = model.generate_text(
        max_tokens=config.model_config.maxlen, 
        start_prompt=start_prompt,
        tokenizer=tokenizer
    )
    logger.log_text("initial_generated_text", generated_text)

    # Training loop
    metrics_history = {'train_loss': []}
    prep_target_batch = jax.vmap(lambda tokens: jnp.concatenate((tokens[1:], jnp.array([tokenizer.pad_token_id]))))
    step = 0

    for epoch in range(config.train_config.num_epochs):
        start_time = time.time()
        for batch in iter(text_dl):
            if len(batch) % len(jax.devices()) != 0:
                continue
            
            input_batch = jnp.array(jnp.array(batch).T)
            target_batch = prep_target_batch(input_batch)
            
            batch_data = (input_batch, target_batch)
            if mesh:
                batch_data = jax.device_put(batch_data, NamedSharding(mesh, P('batch', None)))
            
            grad_norms = train_step(model, optimizer, metrics_manager, batch_data)

            if (step + 1) % config.train_config.log_interval == 0:
                computed_metrics = metrics_manager.compute()
                loss_value = computed_metrics['loss'].item()
                metrics_history['train_loss'].append(loss_value)
                
                log_metrics = {'train_loss': loss_value}
                
                flat_grad_norms = flatten_for_logging(grad_norms, prefix='grads')
                log_metrics.update(flat_grad_norms)

                flat_determinants = get_flat_determinants(model)
                log_metrics.update(flat_determinants)
                
                logger.log_metrics(log_metrics, step=step + 1)
                metrics_manager.reset()

                elapsed_time = time.time() - start_time
                # print(f"Step {step + 1}, Loss: {loss_value}, Elapsed Time: {elapsed_time:.2f} seconds")
                
                start_time = time.time()
            step += 1

            if (step + 1) % config.train_config.text_log_interval == 0:
                generated_text = model.generate_text(
                    max_tokens=config.model_config.maxlen, 
                    start_prompt=start_prompt,
                    tokenizer=tokenizer
                )
                logger.log_text("generated_text", generated_text, step=step + 1)
                visualize_and_log_loss(metrics_history, logger, step=step + 1)

    # Final text generation
    final_text = model.generate_text(
        max_tokens=config.model_config.maxlen, 
        start_prompt=start_prompt,
        tokenizer=tokenizer
    )
    logger.log_text("final_generated_text", final_text, step=step)

    # Save checkpoint
    state = nnx.state(model)
    checkpointer = orbax.PyTreeCheckpointer()
    save_dir = config.train_config.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)
    checkpointer.save(os.path.join(save_dir, 'model_checkpoint'), state)
    print(f"Checkpoint saved to {save_dir}")
    
    logger.finish()

if __name__ == "__main__":
    main()