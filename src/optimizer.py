import optax
import flax.nnx as nnx

def create_optimizer(model, config):
    """
    Creates an optimizer based on the provided configuration.
    Supports 'adam' and 'adamw'.
    Applies gradient clipping and a learning rate schedule.
    """
    # Create a learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.train_config.learning_rate,
        warmup_steps=config.train_config.lr_warmup_steps,
        decay_steps=config.train_config.lr_num_decay_steps,
        end_value=config.train_config.learning_rate / 10.0
    )

    # Chain the optimizer with gradient clipping
    optimizer_chain = []
    if hasattr(config.train_config, 'grad_clip_value') and config.train_config.grad_clip_value and config.train_config.grad_clip_value > 0:
        optimizer_chain.append(optax.clip_by_global_norm(config.train_config.grad_clip_value))

    if config.train_config.optimizer_name == 'adam':
        optimizer_chain.append(optax.adam(
            learning_rate=lr_schedule
        ))
    elif config.train_config.optimizer_name == 'adamw':
        optimizer_chain.append(optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.train_config.weight_decay
        ))
    else:
        raise ValueError(f"Unsupported optimizer: {config.train_config.optimizer_name}")

    optimizer = optax.chain(*optimizer_chain)
    
    return nnx.Optimizer(model, optimizer) 