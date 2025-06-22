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
import yaml

from config import ProjectConfig, ModelConfig, DataConfig, TrainConfig
from dataset import load_text_dataset
from model import create_model
from optimizer import create_optimizer
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
    
    # Add an argument for the config file
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    
    # Temporarily parse for the config file path
    config_args, _ = parser.parse_known_args()

    # Load config from YAML file
    try:
        with open(config_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_args.config}. Using default values.")
        yaml_config = {}

    # Create a new parser that will consume all arguments, and add back the config argument
    parser = argparse.ArgumentParser(description="Train a mini-GPT model.")
    parser.add_argument("--config", type=str, default=config_args.config, help="Path to the YAML config file.")
    
    default_config = ProjectConfig()

    # Helper to get config values from YAML or defaults
    def get_config_value(section, key, default_value):
        return yaml_config.get(section, {}).get(key, default_value)

    # Model args
    model_config_defaults = default_config.model_config
    parser.add_argument("--model_name", type=str, default=get_config_value("model_config", "model_name", model_config_defaults.model_name), help="Name of the model to train.")
    parser.add_argument("--maxlen", type=int, default=get_config_value("model_config", "maxlen", model_config_defaults.maxlen), help="Maximum sequence length.")
    parser.add_argument("--vocab_size", type=int, default=get_config_value("model_config", "vocab_size", model_config_defaults.vocab_size), help="Vocabulary size.")
    parser.add_argument("--embed_dim", type=int, default=get_config_value("model_config", "embed_dim", model_config_defaults.embed_dim), help="Embedding dimensionality.")
    parser.add_argument("--num_heads", type=int, default=get_config_value("model_config", "num_heads", model_config_defaults.num_heads), help="Number of attention heads.")
    parser.add_argument("--feed_forward_dim", type=int, default=get_config_value("model_config", "feed_forward_dim", model_config_defaults.feed_forward_dim), help="Dimensionality of the feed-forward network.")
    parser.add_argument("--num_transformer_blocks", type=int, default=get_config_value("model_config", "num_transformer_blocks", model_config_defaults.num_transformer_blocks), help="Number of transformer blocks.")
    parser.add_argument("--dropout_rate", type=float, default=get_config_value("model_config", "dropout_rate", model_config_defaults.dropout_rate), help="Dropout rate.")
    parser.add_argument("--dropconnect_rate", type=float, default=get_config_value("model_config", "dropconnect_rate", model_config_defaults.dropconnect_rate), help="Dropconnect rate.")
    parser.add_argument("--use_dropconnect", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("model_config", "use_dropconnect", model_config_defaults.use_dropconnect), help="Whether to use dropconnect.")
    parser.add_argument("--use_softermax", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("model_config", "use_softermax", model_config_defaults.use_softermax), help="Whether to use softermax.")
    parser.add_argument("--power", type=float, default=get_config_value("model_config", "power", model_config_defaults.power), help="Power for softermax.")

    # Data args
    data_config_defaults = default_config.data_config
    parser.add_argument("--dataset_name", type=str, default=get_config_value("data_config", "dataset_name", data_config_defaults.dataset_name), help="Hugging Face dataset name.")
    parser.add_argument("--split", type=str, default=get_config_value("data_config", "split", data_config_defaults.split), help="Dataset split to use.")
    parser.add_argument("--validation_split_name", type=str, default=get_config_value("data_config", "validation_split_name", data_config_defaults.validation_split_name), help="Dataset validation split to use.")
    parser.add_argument("--batch_size", type=int, default=get_config_value("data_config", "batch_size", data_config_defaults.batch_size), help="Batch size for training.")
    parser.add_argument("--tokenizer_name", type=str, default=get_config_value("data_config", "tokenizer_name", data_config_defaults.tokenizer_name), help="Tokenizer to use.")
    parser.add_argument("--loader", type=str, default=get_config_value("data_config", "loader", data_config_defaults.loader), help="Data loader to use ('grain' or 'tf').")
    parser.add_argument("--use_cache", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("data_config", "use_cache", data_config_defaults.use_cache), help="Whether to use caching.")
    parser.add_argument("--shuffle_seed", type=int, default=get_config_value("data_config", "shuffle_seed", data_config_defaults.shuffle_seed), help="Seed for dataset shuffling.")
    parser.add_argument("--shuffle_buffer_size", type=int, default=get_config_value("data_config", "shuffle_buffer_size", data_config_defaults.shuffle_buffer_size), help="Buffer size for dataset shuffling.")
    parser.add_argument("--cache_size", type=int, default=get_config_value("data_config", "cache_size", data_config_defaults.cache_size), help="Size of data cache (for 'grain' loader).")
    parser.add_argument("--num_threads", type=int, default=get_config_value("data_config", "num_threads", data_config_defaults.num_threads), help="Number of threads for data loading.")
    parser.add_argument("--prefetch_buffer_size", type=int, default=get_config_value("data_config", "prefetch_buffer_size", data_config_defaults.prefetch_buffer_size), help="Prefetch buffer size for data loading.")
    parser.add_argument("--tokenization_batch_size", type=int, default=get_config_value("data_config", "tokenization_batch_size", data_config_defaults.tokenization_batch_size), help="Batch size for tokenization.")
    parser.add_argument("--use_fast_tokenizer", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("data_config", "use_fast_tokenizer", data_config_defaults.use_fast_tokenizer), help="Whether to use fast tokenizer.")

    # Train args
    train_config_defaults = default_config.train_config
    parser.add_argument("--optimizer_name", type=str, default=get_config_value("train_config", "optimizer_name", train_config_defaults.optimizer_name), help="Optimizer to use (e.g., 'adam', 'adamw').")
    parser.add_argument("--num_epochs", type=int, default=get_config_value("train_config", "num_epochs", train_config_defaults.num_epochs), help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=get_config_value("train_config", "learning_rate", train_config_defaults.learning_rate), help="Learning rate for the optimizer.")
    parser.add_argument("--lr_warmup_steps", type=int, default=get_config_value("train_config", "lr_warmup_steps", train_config_defaults.lr_warmup_steps), help="Number of warmup steps for learning rate schedule.")
    parser.add_argument("--lr_num_decay_steps", type=int, default=get_config_value("train_config", "lr_num_decay_steps", train_config_defaults.lr_num_decay_steps), help="Number of decay steps for learning rate schedule.")
    parser.add_argument("--weight_decay", type=float, default=get_config_value("train_config", "weight_decay", train_config_defaults.weight_decay), help="Weight decay for the optimizer.")
    parser.add_argument("--grad_clip_value", type=float, default=get_config_value("train_config", "grad_clip_value", train_config_defaults.grad_clip_value), help="Value for gradient clipping.")
    parser.add_argument("--log_interval", type=int, default=get_config_value("train_config", "log_interval", train_config_defaults.log_interval), help="Interval for logging metrics.")
    parser.add_argument("--text_log_interval", type=int, default=get_config_value("train_config", "text_log_interval", train_config_defaults.text_log_interval), help="Interval for logging generated text.")
    parser.add_argument("--use_wandb", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("train_config", "use_wandb", train_config_defaults.use_wandb), help="Whether to use wandb for logging.")
    parser.add_argument("--checkpoint_dir", type=str, default=get_config_value("train_config", "checkpoint_dir", train_config_defaults.checkpoint_dir), help="Directory to save checkpoints.")
    parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("train_config", "debug", train_config_defaults.debug), help="Enable or disable debug prints.")
    parser.add_argument("--run_generation", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("train_config", "run_generation", train_config_defaults.run_generation), help="Whether to run text generation.")
    parser.add_argument("--log_determinants", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("train_config", "log_determinants", train_config_defaults.log_determinants), help="Whether to log matrix determinants.")
    parser.add_argument("--log_gradients", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("train_config", "log_gradients", train_config_defaults.log_gradients), help="Whether to log gradient norms.")
    
    # Now parse all arguments
    args = parser.parse_args()
    
    config = ProjectConfig(
        model_config=ModelConfig(
            model_name=args.model_name,
            maxlen=args.maxlen,
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            feed_forward_dim=args.feed_forward_dim,
            num_transformer_blocks=args.num_transformer_blocks,
            dropout_rate=args.dropout_rate,
            dropconnect_rate=args.dropconnect_rate,
            use_dropconnect=args.use_dropconnect,
            use_softermax=args.use_softermax,
            power=args.power
        ),
        data_config=DataConfig(
            dataset_name=args.dataset_name,
            split=args.split,
            validation_split_name=args.validation_split_name,
            batch_size=args.batch_size,
            tokenizer_name=args.tokenizer_name,
            loader=args.loader,
            use_cache=args.use_cache,
            shuffle_seed=args.shuffle_seed,
            shuffle_buffer_size=args.shuffle_buffer_size,
            cache_size=args.cache_size,
            num_threads=args.num_threads,
            prefetch_buffer_size=args.prefetch_buffer_size,
            tokenization_batch_size=args.tokenization_batch_size,
            use_fast_tokenizer=args.use_fast_tokenizer,
        ),
        train_config=TrainConfig(
            optimizer_name=args.optimizer_name,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_num_decay_steps=args.lr_num_decay_steps,
            weight_decay=args.weight_decay,
            grad_clip_value=args.grad_clip_value,
            log_interval=args.log_interval,
            text_log_interval=args.text_log_interval,
            use_wandb=args.use_wandb,
            checkpoint_dir=args.checkpoint_dir,
            debug=args.debug,
            run_generation=args.run_generation,
            log_determinants=args.log_determinants,
            log_gradients=args.log_gradients
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
    tokenizer = AutoTokenizer.from_pretrained(config.data_config.tokenizer_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    text_dl = load_text_dataset(config.data_config, config.model_config, config.train_config, config.data_config.tokenizer_name, tokenizer.pad_token_id)

    # Create model
    rngs = nnx.Rngs(0)
    model = create_model(config.model_config.model_name, config.model_config, mesh, rngs=rngs)
    
    # Count trainable parameters
    params = nnx.state(model, nnx.Param)
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Number of model parameters: {num_params / 1e6:.2f}M")
    logger.log_metrics({'num_params': num_params}, step=0)

    # Optimizer
    optimizer = create_optimizer(model, config)
    
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

        grad_norms = None
        if config.train_config.log_gradients:
            grad_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x) if x is not None else 0.0, grads)
        
        return grad_norms

    # Initial text generation
    start_prompt = "The difference between the mind and the cosmos is"
    if config.train_config.run_generation:
        generated_text = model.generate_text(
            max_tokens=config.model_config.maxlen, 
            start_prompt=start_prompt,
            tokenizer=tokenizer
        )
        logger.log_text("initial_generated_text", generated_text)

    # Training loop
    metrics_history = {'train_loss': []}

    step = 0

    for epoch in range(config.train_config.num_epochs):
        start_time = time.time()
        for batch_data in text_dl:

            if mesh:
                # Shard the batch dimension (axis 1) across the 'batch' mesh axis
                batch_data = jax.device_put(batch_data, NamedSharding(mesh, P(None, 'batch')))
            
            grad_norms = train_step(model, optimizer, metrics_manager, batch_data)

            if (step + 1) % config.train_config.log_interval == 0:
                computed_metrics = metrics_manager.compute()
                loss_value = computed_metrics['loss'].item()
                metrics_history['train_loss'].append(loss_value)
                
                elapsed_time = time.time() - start_time
                
                log_metrics = {'train_loss': loss_value, 'elapsed_time': elapsed_time}
                
                if config.train_config.log_gradients and grad_norms is not None:
                    flat_grad_norms = flatten_for_logging(grad_norms, prefix='grads')
                    log_metrics.update(flat_grad_norms)

                if config.train_config.log_determinants:
                    flat_determinants = get_flat_determinants(model, debug=config.train_config.debug)
                    log_metrics.update(flat_determinants)
                
                logger.log_metrics(log_metrics, step=step + 1)
                metrics_manager.reset()

                # print(f"Step {step + 1}, Loss: {loss_value}, Elapsed Time: {elapsed_time:.2f} seconds")
                
                start_time = time.time()
            step += 1

            if (step + 1) % config.train_config.text_log_interval == 0 and config.train_config.run_generation:
                generated_text = model.generate_text(
                    max_tokens=config.model_config.maxlen, 
                    start_prompt=start_prompt,
                    tokenizer=tokenizer
                )
                logger.log_text("generated_text", generated_text, step=step + 1)

            if (step + 1) % config.train_config.text_log_interval == 0:
                visualize_and_log_loss(metrics_history, logger, step=step + 1)

    # Final text generation
    if config.train_config.run_generation:
        final_text = model.generate_text(
            max_tokens=config.model_config.maxlen, 
            start_prompt=start_prompt,
            tokenizer=tokenizer
        )
        logger.log_text("final_generated_text", final_text, step=step)

    # Save checkpoint
    state = nnx.state(model, nnx.Param)
    checkpointer = orbax.PyTreeCheckpointer()
    save_dir = os.path.abspath(config.train_config.checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'model_checkpoint')
    checkpointer.save(checkpoint_path, state)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Load the model from the checkpoint to verify it
    print("\nVerifying checkpoint by loading and generating text...")
    
    # Create a new model instance for testing
    test_rngs = nnx.Rngs(1)
    test_model = create_model(config.model_config.model_name, config.model_config, mesh, rngs=test_rngs)
    
    # Create a template state object with the same structure as the model to be restored
    # This ensures that the sharding information is correctly applied during restoration
    abstract_state = jax.eval_shape(lambda: nnx.state(test_model, nnx.Param))
    
    # Load the checkpoint using the abstract state as a template
    restored_state = checkpointer.restore(checkpoint_path, item=abstract_state)
    
    # Update the new model with the loaded state
    nnx.update(test_model, restored_state)
    
    test_generated_text = test_model.generate_text(
        max_tokens=config.model_config.maxlen,
        start_prompt=start_prompt,
        tokenizer=tokenizer
    )
    logger.log_text("test_generated_text", test_generated_text, step=step)
    print("--- Text generated from loaded model ---")
    print(test_generated_text)
    print("--- Checkpoint verification complete ---")
    
    logger.finish()

if __name__ == "__main__":
    main()