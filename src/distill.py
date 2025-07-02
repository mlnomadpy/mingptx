import os
import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint as orbax
import grain.python as grain
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
import argparse
import yaml
import numpy as np

from config import ProjectConfig, ModelConfig, DataConfig, TrainConfig, DistillConfig
from dataset import load_text_dataset
from model import create_model
from optimizer import create_optimizer
from log import Logger, visualize_and_log_loss, flatten_for_logging, get_flat_determinants

def setup_mesh():
    devices = jax.devices()
    num_devices = len(devices)
    if num_devices > 1:
        mesh_shape = (jax.local_device_count(), num_devices // jax.local_device_count())
        return Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Distill a GPT model.")
    
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    
    config_args, _ = parser.parse_known_args()

    try:
        with open(config_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_args.config}. Using default values.")
        yaml_config = {}

    parser = argparse.ArgumentParser(description="Distill a GPT model.")
    parser.add_argument("--config", type=str, default=config_args.config, help="Path to the YAML config file.")
    
    default_config = ProjectConfig()

    def get_config_value(section, key, default_value):
        return yaml_config.get(section, {}).get(key, default_value)

    # Model args (Student)
    model_config_defaults = default_config.model_config
    parser.add_argument("--model_name", type=str, default=get_config_value("model_config", "model_name", model_config_defaults.model_name), help="Name of the student model to train.")
    # ... (rest of model args are for student)
    parser.add_argument("--maxlen", type=int, default=get_config_value("model_config", "maxlen", model_config_defaults.maxlen), help="Maximum sequence length.")
    parser.add_argument("--vocab_size", type=int, default=get_config_value("model_config", "vocab_size", model_config_defaults.vocab_size), help="Vocabulary size.")
    parser.add_argument("--embed_dim", type=int, default=get_config_value("model_config", "embed_dim", model_config_defaults.embed_dim), help="Embedding dimensionality.")
    parser.add_argument("--num_heads", type=int, default=get_config_value("model_config", "num_heads", model_config_defaults.num_heads), help="Number of attention heads.")
    parser.add_argument("--feed_forward_dim", type=int, default=get_config_value("model_config", "feed_forward_dim", model_config_defaults.feed_forward_dim), help="Dimensionality of the feed-forward network.")
    parser.add_argument("--num_transformer_blocks", type=int, default=get_config_value("model_config", "num_transformer_blocks", model_config_defaults.num_transformer_blocks), help="Number of transformer blocks.")
    parser.add_argument("--dropout_rate", type=float, default=get_config_value("model_config", "dropout_rate", model_config_defaults.dropout_rate), help="Dropout rate.")

    # Data args
    data_config_defaults = default_config.data_config
    parser.add_argument("--dataset_name", type=str, default=get_config_value("data_config", "dataset_name", data_config_defaults.dataset_name), help="Hugging Face dataset name.")
    parser.add_argument("--split", type=str, default=get_config_value("data_config", "split", data_config_defaults.split), help="Dataset split to use.")
    parser.add_argument("--batch_size", type=int, default=get_config_value("data_config", "batch_size", data_config_defaults.batch_size), help="Batch size for training.")
    parser.add_argument("--tokenizer_name", type=str, default=get_config_value("data_config", "tokenizer_name", data_config_defaults.tokenizer_name), help="Tokenizer to use (should match teacher).")

    # Train args
    train_config_defaults = default_config.train_config
    parser.add_argument("--optimizer_name", type=str, default=get_config_value("train_config", "optimizer_name", train_config_defaults.optimizer_name), help="Optimizer to use (e.g., 'adam', 'adamw').")
    parser.add_argument("--num_epochs", type=int, default=get_config_value("train_config", "num_epochs", train_config_defaults.num_epochs), help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=get_config_value("train_config", "learning_rate", train_config_defaults.learning_rate), help="Learning rate for the optimizer.")
    parser.add_argument("--log_interval", type=int, default=get_config_value("train_config", "log_interval", train_config_defaults.log_interval), help="Interval for logging metrics.")
    parser.add_argument("--use_wandb", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("train_config", "use_wandb", train_config_defaults.use_wandb), help="Whether to use wandb for logging.")
    parser.add_argument("--checkpoint_dir", type=str, default=get_config_value("train_config", "checkpoint_dir", train_config_defaults.checkpoint_dir), help="Directory to save checkpoints.")
    parser.add_argument("--start_prompt", type=str, default=get_config_value("train_config", "start_prompt", train_config_defaults.start_prompt), help="Start prompt for text generation.")
    parser.add_argument("--text_log_interval", type=int, default=get_config_value("train_config", "text_log_interval", train_config_defaults.text_log_interval), help="Interval for logging generated text.")
    parser.add_argument("--run_generation", type=lambda x: (str(x).lower() == 'true'), default=get_config_value("train_config", "run_generation", train_config_defaults.run_generation), help="Whether to run text generation.")
    
    # Distill args
    distill_config_defaults = default_config.distill_config
    parser.add_argument("--teacher_model_name", type=str, default=get_config_value("distill_config", "teacher_model_name", distill_config_defaults.teacher_model_name), help="Name of the teacher model (from Hugging Face).")
    parser.add_argument("--distillation_alpha", type=float, default=get_config_value("distill_config", "distillation_alpha", distill_config_defaults.distillation_alpha), help="Weight for the distillation loss component.")
    parser.add_argument("--distillation_temperature", type=float, default=get_config_value("distill_config", "distillation_temperature", distill_config_defaults.distillation_temperature), help="Temperature for softening probability distributions.")

    args = parser.parse_args()
    
    config = ProjectConfig(
        model_config=ModelConfig(
            model_name=args.model_name,
            maxlen=args.maxlen,
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            feed_forward_dim=args.feed_forward_dim,
            num_transformer_blocks=args.num_transformer_blocks,
            dropout_rate=args.dropout_rate,
        ),
        data_config=DataConfig(
            dataset_name=args.dataset_name,
            split=args.split,
            batch_size=args.batch_size,
            tokenizer_name=args.tokenizer_name,
        ),
        train_config=TrainConfig(
            optimizer_name=args.optimizer_name,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            log_interval=args.log_interval,
            use_wandb=args.use_wandb,
            checkpoint_dir=args.checkpoint_dir,
            start_prompt=args.start_prompt,
            text_log_interval=args.text_log_interval,
            run_generation=args.run_generation,
        ),
        distill_config=DistillConfig(
            teacher_model_name=args.teacher_model_name,
            distillation_alpha=args.distillation_alpha,
            distillation_temperature=args.distillation_temperature,
        )
    )
    return config

def main():
    config = parse_args()
    mesh = setup_mesh()

    logger = Logger(project_name="mingptx-distill", config={
        "student_model": config.model_config,
        "data": config.data_config,
        "train": config.train_config,
        "distill": config.distill_config
    }, use_wandb=config.train_config.use_wandb)

    # Load tokenizer from teacher model
    tokenizer = AutoTokenizer.from_pretrained(config.distill_config.teacher_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure student vocab size matches tokenizer
    config.model_config.vocab_size = tokenizer.vocab_size

    # Load data
    text_dl = load_text_dataset(config.data_config, config.model_config, config.train_config, config.data_config.tokenizer_name, tokenizer.pad_token_id)

    # Load Teacher Model
    print(f"Loading teacher model: {config.distill_config.teacher_model_name}")
    teacher_model = FlaxAutoModelForCausalLM.from_pretrained(config.distill_config.teacher_model_name)
    teacher_params = teacher_model.params
    
    # Create Student Model
    rngs = nnx.Rngs(0)
    student_model = create_model(config.model_config.model_name, config.model_config, mesh, rngs=rngs)
    
    params = nnx.state(student_model, nnx.Param)
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Number of student model parameters: {num_params / 1e6:.2f}M")
    logger.log_metrics({'student_num_params': num_params}, step=0)

    optimizer = create_optimizer(student_model, config)
    metrics_manager = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        distill_loss=nnx.metrics.Average('distill_loss'),
        student_loss=nnx.metrics.Average('student_loss')
    )
    
    def loss_fn(student, teacher_prms, batch, alpha, temp):
        inputs, targets, attention_mask = batch
        # HF models expect (batch, seq_len), our model expects (seq_len, batch)
        # Transpose inputs for teacher
        teacher_inputs = inputs.T
        teacher_attention_mask = attention_mask.T

        # Get teacher logits (no gradients)
        teacher_outputs = teacher_model(
            input_ids=teacher_inputs, 
            attention_mask=teacher_attention_mask,
            params=teacher_prms
        )
        teacher_logits = jax.lax.stop_gradient(teacher_outputs.logits.transpose((1, 0, 2))) # Back to (seq_len, batch, vocab)

        # Get student logits
        student_logits = student(inputs, training=True)

        # Create comprehensive mask combining attention mask and target validity
        # attention_mask: 1 for real tokens, 0 for padding
        # We want to mask out: padding tokens AND the last position (no next token to predict)
        sequence_mask = attention_mask.astype(bool)
        
        # For causal LM, we don't predict the token after the last real token
        # Shift sequence_mask to align with targets (targets are inputs shifted left)
        target_mask = jnp.concatenate([sequence_mask[1:], jnp.zeros((1, sequence_mask.shape[1]), dtype=bool)], axis=0)
        
        # Additional safety: ensure targets are valid (not padding tokens)
        valid_target_mask = targets != tokenizer.pad_token_id
        
        # Combine all masking conditions
        final_mask = target_mask & valid_target_mask
        
        # Count valid positions for proper normalization
        num_valid_tokens = jnp.sum(final_mask)
        
        # Avoid division by zero
        num_valid_tokens = jnp.maximum(num_valid_tokens, 1.0)
        
        # Distillation loss (KL divergence) - only on valid positions
        soft_teacher_logits = jax.nn.log_softmax(teacher_logits / temp)
        soft_student_logits = jax.nn.log_softmax(student_logits / temp)
        
        # KL divergence: KL(P_teacher || P_student) = sum(P_teacher * log(P_teacher / P_student))
        kl_div = jnp.sum(jnp.exp(soft_teacher_logits) * (soft_teacher_logits - soft_student_logits), axis=-1)
        distill_loss = jnp.sum(kl_div * final_mask) / num_valid_tokens

        # Student loss (cross-entropy) - only on valid positions
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(student_logits, targets)
        student_loss = jnp.sum(ce_loss * final_mask) / num_valid_tokens
        
        # Total loss
        total_loss = alpha * distill_loss + (1.0 - alpha) * student_loss
        
        # Return additional metrics for monitoring
        metrics = {
            'num_valid_tokens': num_valid_tokens,
            'mask_ratio': num_valid_tokens / (final_mask.shape[0] * final_mask.shape[1])
        }
        
        return total_loss, (distill_loss, student_loss, metrics)

    @nnx.jit
    def train_step(stdnt, tchr_prms, opt, mets, b, alpha, temp):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, (d_loss, s_loss, metrics)), grads = grad_fn(stdnt, tchr_prms, b, alpha, temp)
        mets.update(loss=loss, distill_loss=d_loss, student_loss=s_loss)
        opt.update(grads)
        return metrics  # Return metrics for logging

    prep_target_batch = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([tokenizer.pad_token_id]))), 
        in_axes=1, out_axes=1
    )

    # Initial generation comparison
    if config.train_config.run_generation:
        print("\n=== Initial Generation Comparison ===")
        try:
            student_initial = student_model.generate_text(
                max_tokens=50, start_prompt=config.train_config.start_prompt, tokenizer=tokenizer
            )
            print(f"Initial Student: {student_initial}")
            logger.log_text("initial_student_generation", student_initial, step=0)
        except Exception as e:
            print(f"Initial student generation failed: {e}")
        
        try:
            input_ids = tokenizer.encode(config.train_config.start_prompt, return_tensors="jax")
            teacher_initial_outputs = teacher_model.generate(
                input_ids=input_ids, params=teacher_params, max_new_tokens=50,
                do_sample=True, temperature=1.0, top_k=10,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )
            teacher_initial = tokenizer.decode(teacher_initial_outputs.sequences[0], skip_special_tokens=True)
            print(f"Initial Teacher: {teacher_initial}")
            logger.log_text("initial_teacher_generation", teacher_initial, step=0)
        except Exception as e:
            print(f"Initial teacher generation failed: {e}")
        print("=== End Initial Generation ===\n")

    step = 0
    for epoch in range(config.train_config.num_epochs):
        data_iterator = text_dl.as_numpy_iterator()
        
        for batch_data in data_iterator:
            # Extract input_ids and attention_mask from the batch
            input_batch = jnp.array(batch_data['input_ids']).T
            attention_mask = jnp.array(batch_data['attention_mask']).T
            
            target_batch = prep_target_batch(input_batch)
            batch = (input_batch, target_batch, attention_mask)

            if mesh:
                batch = jax.device_put(batch, NamedSharding(mesh, P(None, 'batch')))

            step_metrics = train_step(
                student_model, teacher_params, optimizer, metrics_manager, batch, 
                config.distill_config.distillation_alpha, 
                config.distill_config.distillation_temperature
            )

            if (step + 1) % config.train_config.log_interval == 0:
                computed_metrics = metrics_manager.compute()
                log_metrics = {k: v.item() for k, v in computed_metrics.items()}
                
                # Add the additional metrics from the latest step
                log_metrics.update({
                    'num_valid_tokens': float(step_metrics['num_valid_tokens']),
                    'mask_ratio': float(step_metrics['mask_ratio'])
                })
                
                logger.log_metrics(log_metrics, step=step + 1)
                metrics_manager.reset()
                print(f"Step {step + 1}, Loss: {log_metrics['loss']:.4f}, Distill Loss: {log_metrics['distill_loss']:.4f}, Student Loss: {log_metrics['student_loss']:.4f}, Mask Ratio: {log_metrics['mask_ratio']:.3f}")

            # Periodic generation comparison
            if (step + 1) % config.train_config.text_log_interval == 0 and config.train_config.run_generation:
                print(f"\n=== Generation Comparison at Step {step + 1} ===")
                try:
                    student_text = student_model.generate_text(
                        max_tokens=50, start_prompt=config.train_config.start_prompt, tokenizer=tokenizer
                    )
                    print(f"Student: {student_text}")
                    logger.log_text("student_generation", student_text, step=step + 1)
                except Exception as e:
                    print(f"Student generation failed: {e}")
                
                try:
                    input_ids = tokenizer.encode(config.train_config.start_prompt, return_tensors="jax")
                    teacher_outputs = teacher_model.generate(
                        input_ids=input_ids, params=teacher_params, max_new_tokens=50,
                        do_sample=True, temperature=1.0, top_k=10,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
                    )
                    teacher_text = tokenizer.decode(teacher_outputs.sequences[0], skip_special_tokens=True)
                    print(f"Teacher: {teacher_text}")
                    logger.log_text("teacher_generation", teacher_text, step=step + 1)
                except Exception as e:
                    print(f"Teacher generation failed: {e}")
                print("=== End Generation Comparison ===\n")

            step += 1

    # Save student model checkpoint
    state = nnx.state(student_model, nnx.Param)
    checkpointer = orbax.PyTreeCheckpointer()
    save_dir = os.path.abspath(config.train_config.checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'{config.model_config.model_name}_distilled')
    checkpointer.save(checkpoint_path, state)
    print(f"Distilled student model saved to {checkpoint_path}")
    
    # Generation test to compare teacher and student models
    print("\n=== Generation Comparison Test ===")
    test_prompt = config.train_config.start_prompt
    max_gen_tokens = 50  # Generate 50 tokens for comparison
    
    print(f"Test prompt: '{test_prompt}'")
    print(f"Generating {max_gen_tokens} tokens...\n")
    
    # Generate text with student model
    print("Student model generation:")
    try:
        student_text = student_model.generate_text(
            max_tokens=max_gen_tokens,
            start_prompt=test_prompt,
            tokenizer=tokenizer
        )
        print(f"Student: {student_text}")
    except Exception as e:
        print(f"Student generation failed: {e}")
        student_text = "Failed to generate"
    
    # Generate text with teacher model
    print("\nTeacher model generation:")
    try:
        # Tokenize the prompt
        input_ids = tokenizer.encode(test_prompt, return_tensors="jax")
        
        # Generate with teacher model (HuggingFace)
        teacher_outputs = teacher_model.generate(
            input_ids=input_ids,
            params=teacher_params,
            max_new_tokens=max_gen_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode the generated text
        teacher_text = tokenizer.decode(teacher_outputs.sequences[0], skip_special_tokens=True)
        print(f"Teacher: {teacher_text}")
    except Exception as e:
        print(f"Teacher generation failed: {e}")
        teacher_text = "Failed to generate"
    
    # Log the comparison
    logger.log_metrics({
        'generation_test_completed': 1,
        'student_generation_length': len(student_text),
        'teacher_generation_length': len(teacher_text)
    }, step=step)
    
    logger.log_text("final_student_generation", student_text, step=step)
    logger.log_text("final_teacher_generation", teacher_text, step=step)
    
    print("\n=== End Generation Comparison ===")
    
    logger.finish()

if __name__ == "__main__":
    main() 