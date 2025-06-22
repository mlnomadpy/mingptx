#!/bin/bash
# Example of running the training script.
#
# The script uses 'config.yaml' by default for configuration.
# You can specify a different config file using the --config argument:
#   bash train.sh --config my_experiment_config.yaml
#
# You can also override specific parameters from the command line.
# For example, to change the batch size and number of epochs:
#   bash train.sh --batch_size 256 --num_epochs 5
#
# The following example shows how to override the model name and number of transformer blocks:
# bash train.sh --model_name "micro-gpt" --num_transformer_blocks 1

python src/train.py "$@" 