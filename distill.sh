#!/bin/bash
# Example of running the distillation script.
#
# The script uses 'config.yaml' by default for configuration.
# You can specify a different config file using the --config argument:
#   bash distill.sh --config my_distill_config.yaml
#
# You can also override specific parameters from the command line.
# For example, to change the student model's embedding dimension:
#   bash distill.sh --embed_dim 128
#
# To specify the teacher model and distillation parameters:
#   bash distill.sh --teacher_model_name "gpt2-medium" --distillation_alpha 0.3 --distillation_temperature 2.5

python src/distill.py "$@" 