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
    
    default_config = ProjectConfig()

    # Model args
    parser.add_argument("--model_name", type=str, default="mini-gpt", help="Name of the model to train.")
    parser.add_argument("--maxlen", type=int, default=default_config.model_config.maxlen, help="Maximum sequence length.")
    parser.add_argument("--vocab_size", type=int, default=default_config.model_config.vocab_size, help="Vocabulary size.")
    parser.add_argument("--embed_dim", type=int, default=default_config.model_config.embed_dim, help="Embedding dimensionality.")
    parser.add_argument("--num_heads", type=int, default=default_config.model_config.num_heads, help="Number of attention heads.")
    parser.add_argument("--feed_forward_dim", type=int, default=default_config.model_config.feed_forward_dim, help="Dimensionality of the feed-forward network.")
    parser.add_argument("--num_transformer_blocks", type=int, default=default_config.model_config.num_transformer_blocks, help="Number of transformer blocks.")
    parser.add_argument("--dropout_rate", type=float, default=default_config.model_config.dropout_rate, help="Dropout rate.")
    parser.add_argument("--dropconnect_rate", type=float, default=default_config.model_config.dropconnect_rate, help="Dropconnect rate.")
    parser.add_argument("--use_dropconnect", type=lambda x: (str(x).lower() == 'true'), default=default_config.model_config.use_dropconnect, help="Whether to use dropconnect.")

    # Data args
    parser.add_argument("--dataset_name", type=str, default=default_config.data_config.dataset_name, help="Hugging Face dataset name.")
    parser.add_argument("--split", type=str, default=default_config.data_config.split, help="Dataset split to use.")
    parser.add_argument("--batch_size", type=int, default=default_config.data_config.batch_size, help="Batch size for training.")
    parser.add_argument("--tokenizer_name", type=str, default=default_config.data_config.tokenizer_name, help="Tokenizer to use.")

    # Train args
    parser.add_argument("--optimizer_name", type=str, default=default_config.train_config.optimizer_name, help="Optimizer to use (e.g., 'adam', 'adamw').")
    parser.add_argument("--num_epochs", type=int, default=default_config.train_config.num_epochs, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=default_config.train_config.learning_rate, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=default_config.train_config.weight_decay, help="Weight decay for the optimizer.")
    parser.add_argument("--log_interval", type=int, default=default_config.train_config.log_interval, help="Interval for logging metrics.")
    parser.add_argument("--text_log_interval", type=int, default=default_config.train_config.text_log_interval, help="Interval for logging generated text.")
    parser.add_argument("--use_wandb", type=lambda x: (str(x).lower() == 'true'), default=default_config.train_config.use_wandb, help="Whether to use wandb for logging.")
    parser.add_argument("--checkpoint_dir", type=str, default=default_config.train_config.checkpoint_dir, help="Directory to save checkpoints.")
    parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), default=default_config.train_config.debug, help="Enable or disable debug prints.")
    
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
            use_dropconnect=args.use_dropconnect
        ),
        data_config=DataConfig(
            dataset_name=args.dataset_name,
            split=args.split,
            batch_size=args.batch_size,
            tokenizer_name=args.tokenizer_name
        ),
        train_config=TrainConfig(
            optimizer_name=args.optimizer_name,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            log_interval=args.log_interval,
            text_log_interval=args.text_log_interval,
            use_wandb=args.use_wandb,
            checkpoint_dir=args.checkpoint_dir,
            debug=args.debug
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
    # Correctly vmap over the batch dimension (axis 1) of (maxlen, batch_size) arrays
    prep_target_batch = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([tokenizer.pad_token_id]))), 
        in_axes=1, 
        out_axes=1
    )
    step = 0

    for epoch in range(config.train_config.num_epochs):
        start_time = time.time()
        for batch in text_dl.as_numpy_iterator():
            # batch is a dict {'input_ids': array} with shape (batch_size, maxlen)
            # The model expects (maxlen, batch_size), so we transpose.
            input_batch = jnp.array(batch['input_ids']).T
            
            # Create target by shifting input
            target_batch = prep_target_batch(input_batch)
            
            batch_data = (input_batch, target_batch)
            
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
                
                flat_grad_norms = flatten_for_logging(grad_norms, prefix='grads')
                log_metrics.update(flat_grad_norms)

                flat_determinants = get_flat_determinants(model, debug=config.train_config.debug)
                log_metrics.update(flat_determinants)
                
                logger.log_metrics(log_metrics, step=step + 1)
                metrics_manager.reset()

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
    
    # Generate text with the loaded model to verify
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