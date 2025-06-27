import flax.nnx as nnx
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from config import ModelConfig
from model import register_model
from models.aether.base import TransformerBlock
from models.gpt import GPT, TokenAndPositionEmbedding
from nmn.nnx.nmn import YatNMN
from nmn.nnx.squashers.softermax import softermax
import jax.numpy as jnp
import jax

@register_model("mini-aethergpt")
class MiniGPT(GPT):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        super().__init__(config, mesh)
        self.embedding_layer = TokenAndPositionEmbedding(config, rngs=rngs)
        self.transformer_blocks = [
            TransformerBlock(config, mesh, rngs=rngs) for _ in range(config.num_transformer_blocks)
        ]
        
        # Handle partitioning based on whether mesh is available
        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.orthogonal(), NamedSharding(mesh, P(None, 'model')))
            alpha_init = nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model')))
            bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model')))
        else:
            kernel_init = nnx.initializers.orthogonal()
            alpha_init = nnx.initializers.ones_init()
            bias_init = nnx.initializers.zeros_init()
        
        self.output_layer = YatNMN(
            in_features=config.embed_dim,
            out_features=config.vocab_size,
            kernel_init=kernel_init,
            alpha_init=alpha_init,
            bias_init=bias_init,
            use_bias=False,
            rngs=rngs
        )

    def __call__(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        return self.output_layer(x)
    
    def generate_text(self, max_tokens: int, start_prompt: str, tokenizer, top_k=10):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        start_tokens = tokenizer.encode(start_prompt, return_tensors="jax")[0].tolist()
        
        def sample_from(logits):
            logits, indices = jax.lax.top_k(logits, k=top_k)
            logits = softermax(logits)
            return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

        def generate_step(tokens):
            pad_len = self.config.maxlen - len(tokens)
            sample_index = len(tokens) - 1
            if pad_len < 0:
                x = jnp.array(tokens[:self.config.maxlen])
                sample_index = self.config.maxlen - 1
            elif pad_len > 0:
                x = jnp.array(tokens + [tokenizer.pad_token_id] * pad_len)
            else:
                x = jnp.array(tokens)

            x = x[None, :]
            logits = self(x, training=False)
            next_token = sample_from(logits[0][sample_index])
            return next_token

        generated_tokens = []
        for _ in range(max_tokens):
            next_token = generate_step(start_tokens + generated_tokens)
            if next_token == tokenizer.eos_token_id:
                break
            generated_tokens.append(int(next_token))
        
        return tokenizer.decode(start_tokens + generated_tokens) 

@register_model("micro-aethergpt")
class MicroGPT(GPT):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        super().__init__(config, mesh)
        self.embedding_layer = TokenAndPositionEmbedding(config, rngs=rngs)
        self.transformer_block = TransformerBlock(config, mesh, rngs=rngs)
        
        # Handle partitioning based on whether mesh is available
        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.orthogonal(), NamedSharding(mesh, P(None, 'model')))
            alpha_init = nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model')))
            bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model')))
        else:
            kernel_init = nnx.initializers.orthogonal()
            alpha_init = nnx.initializers.ones_init()
            bias_init = nnx.initializers.zeros_init()
        
        self.output_layer = YatNMN(
            in_features=config.embed_dim,
            out_features=config.vocab_size,
            kernel_init=kernel_init,
            alpha_init=alpha_init,
            bias_init=bias_init,
            use_bias=False,
            rngs=rngs
        )

    def __call__(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x, training=training)
        return self.output_layer(x) 
    
  
    def generate_text(self, max_tokens: int, start_prompt: str, tokenizer, top_k=10):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        start_tokens = tokenizer.encode(start_prompt, return_tensors="jax")[0].tolist()
        
        def sample_from(logits):
            logits, indices = jax.lax.top_k(logits, k=top_k)
            logits = softermax(logits)
            return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

        def generate_step(tokens):
            pad_len = self.config.maxlen - len(tokens)
            sample_index = len(tokens) - 1
            if pad_len < 0:
                x = jnp.array(tokens[:self.config.maxlen])
                sample_index = self.config.maxlen - 1
            elif pad_len > 0:
                x = jnp.array(tokens + [tokenizer.pad_token_id] * pad_len)
            else:
                x = jnp.array(tokens)

            x = x[None, :]
            logits = self(x, training=False)
            next_token = sample_from(logits[0][sample_index])
            return next_token

        generated_tokens = []
        for _ in range(max_tokens):
            next_token = generate_step(start_tokens + generated_tokens)
            if next_token == tokenizer.eos_token_id:
                break
            generated_tokens.append(int(next_token))
        
        return tokenizer.decode(start_tokens + generated_tokens) 
