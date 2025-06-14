import wandb
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, project_name, config, use_wandb=True):
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=project_name, config=config)

    def log_text(self, key, text, step=None):
        print(f"{key}:\n{text}\n")
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
        
        print(print_str.strip().strip('|').strip())
        
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