import wandb
import matplotlib.pyplot as plt
import jax.tree_util as jtu

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