# mingptx

A modular and extensible implementation of a mini-GPT, powered by JAX and Flax. This project is designed for simplicity, performance, and scalability, making it an ideal starting point for transformer-based language modeling research and development.

## Features

- **Modular Design**: Clear, separate modules for configuration, data, and model architecture.
- **JAX and Flax (NNX)**: High-performance numerical computing and elegant neural network modeling.
- **YAML-based Configuration**: Easily manage experiments using `config.yaml` with command-line overrides.
- **Hugging Face Integration**: Seamlessly use `datasets` and `tokenizers`.
- **Customizable Models**: A base GPT class makes it easy to create and experiment with new architectures.
- **Custom Loss Functions**: Includes a "Softermax" loss function for experimentation.
- **Advanced Logging**: Deep dive into your training with detailed logs for determinants, gradients, and batch stats, all integrated with Weights & Biases.
- **Multi-Device Support**: Natively supports training on multiple GPUs/TPUs.
- **In-training Text Generation**: Monitor model progress by generating text samples during training.

## Project Structure

```
mingptx/
├── src/
│   ├── train.py                    # Main training script
│   ├── predict.py                  # Script for text generation
│   ├── model.py                    # Model loader
│   ├── optimizer.py                # Optimizer creation logic
│   ├── dataset.py                  # Data loading and preprocessing
│   ├── config.py                   # Dataclasses for configuration
│   ├── log.py                      # Logging utilities
│   ├── losses/                     # Custom loss functions
│   │   └── softermax...entropy.py
│   └── models/                     # GPT model implementations
│       ├── gpt.py                  # Base GPT model
│       ├── linear/
│       │   ├── base.py
│       │   └── mingpt.py
│       └── aether/
│           ├── base.py
│           └── aethergpt.py
├── config.yaml                     # Main configuration file
├── train.sh                        # Example script to run training
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or later
- JAX and the required dependencies for your hardware (CPU, GPU, or TPU). Follow the [official JAX installation guide](https://github.com/google/jax#installation).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/mingptx.git
    cd mingptx
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

To start the training process, simply run the `train.sh` script:

```bash
bash train.sh
```

This will start training with the default configuration specified in `config.yaml`.

## Demos & Pre-trained Models

You can try `mingptx` in your browser or download pre-trained models from the following links:

- **Colab Notebook**: [Link to Colab]
- **Kaggle Notebook**: [Link to Kaggle]
- **Hugging Face Model**: [Link to Hugging Face]

## Configuration

The primary configuration for the project is managed through the `config.yaml` file. This file is organized into three main sections: `model_config`, `data_config`, and `train_config`.

You can override any setting in `config.yaml` by passing it as a command-line argument to `train.py`. For example, to change the learning rate and batch size:

```bash
python src/train.py --learning_rate 0.0002 --batch_size 64
```

The `train.sh` script provides a convenient way to pass these arguments. For example:
```bash
bash train.sh --learning_rate 0.0002 --optimizer_name adamw
```

## Text Generation

To generate text with a trained model, use the `predict.py` script. It can run in an interactive mode or take a single prompt.

```bash
# Interactive mode
python src/predict.py --checkpoint_dir /path/to/your/checkpoint

# Single prompt
python src/predict.py --checkpoint_dir /path/to/your/checkpoint --prompt "Hello, world!" --max_tokens 100
```

## Utilities

This project includes several utility scripts in the `src/` directory:
- `auto_optimize_config.py`: Helps in finding an optimal batch size that maximizes VRAM usage without causing OOM errors.
- `benchmark_data_loading.py`: Benchmarks the performance of different data loading configurations.

## Weights & Biases

To use Weights & Biases for experiment tracking, make sure you have an account and are logged in. The training script will automatically log metrics, generated text samples, and training plots to your `wandb` dashboard. You can disable this feature by setting `use_wandb: False` in `config.yaml`.
