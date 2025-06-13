#!/bin/bash
# Example of running the training script with all parameters configured from the command line.
python src/train.py \
    --maxlen 256 \
    --vocab_size 50257 \
    --embed_dim 256 \
    --num_heads 8 \
    --feed_forward_dim 256 \
    --num_transformer_blocks 8 \
    --dropout_rate 0.1 \
    --dataset_name "roneneldan/TinyStories" \
    --split "train" \
    --batch_size 256 \
    --tokenizer_name "gpt2" \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --log_interval 200 \
    --use_wandb True \
    --checkpoint_dir "checkpoints" 