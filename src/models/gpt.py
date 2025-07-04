import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import Mesh

from config import ModelConfig

class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(num_embeddings=config.vocab_size, features=config.embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=config.maxlen, features=config.embed_dim, rngs=rngs)

    def __call__(self, x):
        # x.shape is (batch_size, seq_len)
        positions = jnp.arange(0, x.shape[1])[None, :] # Shape: (1, seq_len)
        position_embedding = self.pos_emb(positions)   # Shape: (1, seq_len, embed_dim)
        token_embedding = self.token_emb(x)           # Shape: (batch_size, seq_len, embed_dim)
        # Broadcast position_embedding across the batch dimension
        return token_embedding + position_embedding

class GPT(nnx.Module):
    def __init__(self, config: ModelConfig, mesh: Mesh):
        self.config = config
        self.mesh = mesh

    def __call__(self, inputs, attention_mask=None, training: bool = False):
        raise NotImplementedError

    def generate_text(self, max_tokens: int, start_prompt: str, tokenizer, pad_token_id, top_k=10):
        is_tiktoken = hasattr(tokenizer, 'eot_token')
        eos_token_id = tokenizer.eot_token if is_tiktoken else tokenizer.eos_token_id

        if not is_tiktoken and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        start_tokens = tokenizer.encode(start_prompt)
        if not is_tiktoken:
            start_tokens = start_tokens.tolist()

        def sample_from(logits):
            logits, indices = jax.lax.top_k(logits, k=top_k)
            # Convert logits to probabilities
            logits = nnx.softmax(logits)
            return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

        def generate_step(tokens):
            pad_len = self.config.maxlen - len(tokens)
            sample_index = len(tokens) - 1
            if pad_len < 0:
                x = jnp.array(tokens[:self.config.maxlen])
                attention_mask = jnp.ones_like(x)
                sample_index = self.config.maxlen - 1
            elif pad_len > 0:
                x = jnp.array(tokens + [pad_token_id] * pad_len)
                attention_mask = jnp.array([1] * len(tokens) + [0] * pad_len)
            else:
                x = jnp.array(tokens)
                attention_mask = jnp.ones_like(x)

            x = x[None, :] # Add batch dimension -> (1, seq_len)
            attention_mask = attention_mask[None, :] # Add batch dimension
            logits = self(x, attention_mask=attention_mask, training=False)
            next_token = sample_from(logits[0, sample_index])
            return next_token

        generated_tokens = []
        for _ in range(max_tokens):
            next_token = generate_step(start_tokens + generated_tokens)
            if next_token == eos_token_id:
                break
            generated_tokens.append(int(next_token))
        
        return tokenizer.decode(start_tokens + generated_tokens) 
    
