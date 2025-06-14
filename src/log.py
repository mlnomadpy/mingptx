import wandb
import matplotlib.pyplot as plt
import jax.tree_util as jtu
import jax.numpy as jnp
import flax.nnx as nnx

class Logger:
    def __init__(self, project_name, config, use_wandb=True):
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=project_name, config=config)

    def log_text(self, key, text, step=None):
        if self.use_wandb:
            log_data = {key: text}
            if step is not None:
                wandb.log(log_data, step=step)
            else:
                wandb.log(log_data)

    def log_metrics(self, metrics, step):
        print_str = f"Step {step}: "
        log_dict = {}
        for k, v in metrics.items():
            # Ensure value is a number before formatting
            if isinstance(v, (int, float)):
                print_str += f"{k}: {v:.4f} | "
            else:
                print_str += f"{k}: {v} | "
            log_dict[k] = v
        
        # print(print_str.strip().strip('|').strip())
        
        if self.use_wandb:
            wandb.log(log_dict, step=step)

    def log_image(self, key, image_path, step=None):
        if self.use_wandb:
            log_data = {key: wandb.Image(image_path)}
            if step is not None:
                wandb.log(log_data, step=step)
            else:
                wandb.log(log_data)

    def finish(self):
        if self.use_wandb:
            wandb.finish()

def visualize_and_log_loss(metrics_history, logger, step):
    plt.plot(metrics_history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    loss_plot_path = 'training_loss.png'
    plt.savefig(loss_plot_path)
    plt.close() # prevent displaying the plot locally
    
    logger.log_image("training_loss_plot", loss_plot_path, step=step)

def flatten_for_logging(pytree, prefix='grads'):
    """Flattens a PyTree and formats the keys for wandb logging."""
    flat_metrics = {}
    
    def format_key(key_path):
        """Creates a readable string from the key path returned by tree_flatten_with_path."""
        keys = []
        for p in key_path:
            if isinstance(p, jtu.GetAttrKey):
                keys.append(p.name)
            elif isinstance(p, jtu.DictKey):
                keys.append(str(p.key))
            elif isinstance(p, jtu.SequenceKey):
                keys.append(str(p.idx))
            else:
                keys.append(str(p)) # Fallback for other key types
        return ".".join(keys)

    leaves, _ = jtu.tree_flatten_with_path(pytree)
    
    for path, value in leaves:
        if value is not None:
            log_key = f"{prefix}/{format_key(path)}"
            flat_metrics[log_key] = value.item()
            
    return flat_metrics

def get_key_name_from_path_entry(key_entry):
    """Safely extracts a string representation from a JAX PyTree KeyEntry."""
    if isinstance(key_entry, jtu.GetAttrKey):
        return key_entry.name
    elif isinstance(key_entry, jtu.DictKey):
        return str(key_entry.key)
    elif isinstance(key_entry, jtu.SequenceKey):
        return str(key_entry.idx)
    return str(key_entry)

def get_flat_determinants(model: nnx.Module, debug: bool = False):
    """
    Calculates the log-abs-determinant of the Gramian matrix for kernel and embedding weights
    and returns them as a flat dictionary for logging.
    """
    if debug:
        print("\n[Debug] Inside get_flat_determinants...")
    model_state = nnx.state(model)
    flat_determinants = {}
    
    leaves, _ = jtu.tree_flatten_with_path(model_state)
    if debug:
        print(f"[Debug] Found {len(leaves)} leaves in model state.")
    
    for path, leaf in leaves:
        path_str_for_debug = ".".join(get_key_name_from_path_entry(p) for p in path)
        
        # Check for 2D arrays and that the path is long enough to have a parent key
        if isinstance(leaf, jnp.ndarray) and leaf.ndim == 2 and len(path) >= 2:
            # The parameter name ('kernel', 'embedding') is typically the second to last key
            key_to_check = get_key_name_from_path_entry(path[-2]).lower()
            if debug:
                print(f"[Debug] Found 2D array at path: {path_str_for_debug} | Key to check: '{key_to_check}' | Shape: {leaf.shape}")
            
            if 'kernel' in key_to_check or 'embedding' in key_to_check:
                if debug:
                    print(f"  -> MATCH FOUND for '{key_to_check}'. Calculating determinant.")
                gramian = leaf.T @ leaf
                gramian += jnp.eye(gramian.shape[0]) * 1e-8
                _sign, logabsdet = jnp.linalg.slogdet(gramian)
                
                log_key = f"determinants/{path_str_for_debug}"
                
                flat_determinants[log_key] = logabsdet.item()
    
    if debug:
        if not flat_determinants:
            print("[Debug] WARNING: No kernel or embedding weights were matched. Determinant dict is empty.")
        else:
            print(f"[Debug] Calculated determinants: {flat_determinants}")
        print("[Debug] Exiting get_flat_determinants.\n")
        
    return flat_determinants 