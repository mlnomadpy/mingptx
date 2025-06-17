import optax
import flax.nnx as nnx

def create_optimizer(model, config):
    """
    Creates an optimizer based on the provided configuration.
    Supports 'adam' and 'adamw'.
    """
    if config.train_config.optimizer_name == 'adam':
        return nnx.Optimizer(model, optax.adam(
            learning_rate=config.train_config.learning_rate
        ))
    elif config.train_config.optimizer_name == 'adamw':
        return nnx.Optimizer(model, optax.adamw(
            learning_rate=config.train_config.learning_rate,
            weight_decay=config.train_config.weight_decay
        ))
    else:
        raise ValueError(f"Unsupported optimizer: {config.train_config.optimizer_name}") 