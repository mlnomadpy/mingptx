# mingptx

A modular and extensible implementation of a mini-GPT, powered by JAX and Flax. This project is designed for simplicity, performance, and scalability, making it an ideal starting point for transformer-based language modeling research and development.

## Features

- **Modular Design**: The project is structured into clear, separate modules for configuration, data handling, and model architecture, promoting code reusability and maintainability.
- **JAX and Flax**: Leverages the power of JAX for high-performance numerical computing and Flax (NNX) for elegant neural network modeling.
- **Weights & Biases Integration**: Comes with built-in support for `wandb` to track experiments, log metrics, and visualize model performance.
- **Hugging Face Datasets**: Seamlessly integrates with the Hugging Face `datasets` library, allowing you to train on a wide variety of text datasets with minimal effort.
- **Extensible GPT Model**: The GPT model is designed with a base class, making it easy to create and experiment with new transformer architectures.
- **Multi-Device Support**: Includes support for training on multiple devices (TPUs/GPUs) using JAX's data and model parallelism features.

## Project Structure

```
mingptx/
├── src/
│   ├── train.py          # Main training script
│   ├── model.py          # GPT model definition
│   ├── dataset.py        # Data loading and preprocessing
│   └── config.py         # Configuration for model, data, and training
├── train.sh              # Example script to run training
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
    *Note: A `requirements.txt` file will be created in the next step.*

### How to Run

To start the training process, simply run the `train.sh` script:

```bash
bash train.sh
```

This will start the training with the default configuration specified in `src/config.py`. You can easily modify the configurations in this file to experiment with different hyperparameters, datasets, or model architectures.

## Weights & Biases

To use Weights & Biases for experiment tracking, make sure you have an account and are logged in. The training script will automatically log metrics, generated text samples, and training plots to your `wandb` dashboard. You can disable this feature by setting `use_wandb = False` in `src/config.py`.
