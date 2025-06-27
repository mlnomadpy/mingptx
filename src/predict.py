import os
import jax
import flax.nnx as nnx
import orbax.checkpoint as orbax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from transformers import AutoTokenizer
import argparse
import yaml

from config import ProjectConfig, ModelConfig
from model import create_model

def setup_mesh():
    devices = jax.devices()
    num_devices = len(devices)
    if num_devices > 1:
        mesh_shape = (jax.local_device_count(), num_devices // jax.local_device_count())
        return Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Predict with a mini-GPT model.")
    
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
    parser = argparse.ArgumentParser(description="Predict with a mini-GPT model.")
    parser.add_argument("--config", type=str, default=config_args.config, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory of the model checkpoint.")
    parser.add_argument("--prompt", type=str, default=None, help="A single prompt to run.")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max new tokens to generate.")
    
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

    # Data args (only tokenizer)
    data_config_defaults = default_config.data_config
    parser.add_argument("--tokenizer_name", type=str, default=get_config_value("data_config", "tokenizer_name", data_config_defaults.tokenizer_name), help="Tokenizer to use.")
    
    # Train args for checkpoint dir
    train_config_defaults = default_config.train_config
    parser.add_argument("--default_checkpoint_dir", type=str, default=get_config_value("train_config", "checkpoint_dir", train_config_defaults.checkpoint_dir), help="Default directory for checkpoints.")
    
    # Now parse all arguments
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.default_checkpoint_dir
    
    model_config = ModelConfig(
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
    )
    
    return args, model_config

def main():
    args, model_config = parse_args()
    mesh = setup_mesh()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    rngs = nnx.Rngs(0)
    model = create_model(model_config.model_name, model_config, mesh, rngs=rngs)
    
    # Load checkpoint
    checkpointer = orbax.PyTreeCheckpointer()
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_checkpoint')
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    abstract_state = jax.eval_shape(lambda: nnx.state(model, nnx.Param))
    
    try:
        restored_state = checkpointer.restore(checkpoint_path, item=abstract_state)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure you have a trained model checkpoint in the specified directory.")
        return
        
    nnx.update(model, restored_state)
    
    print("Model loaded successfully.")
    
    if args.prompt:
        print(f"Generating text for prompt: '{args.prompt}'")
        generated_text = model.generate_text(
            max_tokens=args.max_tokens,
            start_prompt=args.prompt,
            tokenizer=tokenizer
        )
        print("\n--- Generated Text ---")
        print(generated_text)
        print("---------------------\n")
    else:
        # Interactive loop
        print("\nEnter a prompt to generate text (or 'quit' to exit).")
        while True:
            try:
                start_prompt = input(">> ")
            except (KeyboardInterrupt, EOFError):
                # Handle Ctrl+C, Ctrl+D
                print("\nExiting.")
                break
            
            if start_prompt.lower() == 'quit':
                break
            
            generated_text = model.generate_text(
                max_tokens=args.max_tokens, 
                start_prompt=start_prompt,
                tokenizer=tokenizer
            )
            print("\n--- Generated Text ---")
            print(generated_text)
            print("---------------------\n")

if __name__ == "__main__":
    main() 