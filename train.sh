#!/bin/bash
# Example of running the training script with all parameters configured from the command line.

# To train micro-gpt, for example, you can run:
# bash train.sh --model_name "micro-gpt" --num_transformer_blocks 1

# Default model_name if not provided
MODEL_NAME=${1:-"micro-aethergpt"}

python src/train.py \
    --model_name "$MODEL_NAME" \
    --maxlen 256 \
    --vocab_size 50257 \
    --embed_dim 256 \
    --num_heads 8 \
    --feed_forward_dim 256 \
    --num_transformer_blocks 8 \
    --dropout_rate 0.1 \
    --use_dropconnect True \
    --dataset_name "roneneldan/TinyStories" \
    --split "train" \
    --batch_size 256 \
    --tokenizer_name "gpt2" \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --log_interval 1 \
    --text_log_interval 200 \
    --use_wandb True \
    --checkpoint_dir "checkpoints" \
    --debug False 