import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from config import ModelConfig
from nmn.nnx.yatattention import MultiHeadAttention
from nmn.nnx.nmn import YatNMN

def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class TransformerBlock(nnx.Module):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        self.mha = MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.embed_dim,
            use_dropconnect=config.use_dropconnect,
            dropconnect_rate=config.dropout_rate,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
            alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )

        self.non_linear1 = YatNMN(
            in_features=config.embed_dim,
            out_features=config.feed_forward_dim,
            use_dropconnect=config.use_dropconnect,
            drop_rate=config.dropout_rate,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
            alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )
        self.non_linear2 = YatNMN(
            in_features=config.feed_forward_dim,
            out_features=config.embed_dim,
            use_dropconnect=config.use_dropconnect,
            drop_rate=config.dropout_rate,
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
            alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )

    def __call__(self, inputs, training: bool = False):
        _, seq_len, _ = inputs.shape
        mask = causal_attention_mask(seq_len)
        attention_output = self.mha(inputs_q=inputs, mask=mask, decode=False)
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = inputs + attention_output
        
        ffn_output = self.non_linear1(out1)
        ffn_output = self.non_linear2(ffn_output)
        return out1 + ffn_output

class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(num_embeddings=config.vocab_size, features=config.embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=config.maxlen, features=config.embed_dim, rngs=rngs)

    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        position_embedding = self.pos_emb(positions)
        token_embedding = self.token_emb(x)
        return token_embedding + position_embedding

class GPT(nnx.Module):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        self.config = config
        self.mesh = mesh
        self.rngs = rngs

    def __call__(self, inputs, training: bool = False):
        raise NotImplementedError

    def generate_text(self, max_tokens: int, start_prompt: str, tokenizer, top_k=10):
        start_tokens = tokenizer.encode(start_prompt, return_tensors="jax")[0].tolist()
        
        def sample_from(logits):
            logits, indices = jax.lax.top_k(logits, k=top_k)
            logits = nnx.softmax(logits)
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