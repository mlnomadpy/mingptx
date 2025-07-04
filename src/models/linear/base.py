import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from config import ModelConfig

def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class TransformerBlock(nnx.Module):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        # Handle partitioning based on whether mesh is available
        if mesh is not None:
            mha_kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model')))
            mha_bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model')))
            layer_norm_scale_init = nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P('model')))
            layer_norm1_bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model')))
            linear_kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model')))
            linear_bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model')))
            layer_norm2_bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model')))
        else:
            mha_kernel_init = nnx.initializers.xavier_uniform()
            mha_bias_init = nnx.initializers.zeros_init()
            layer_norm_scale_init = nnx.initializers.ones_init()
            layer_norm1_bias_init = nnx.initializers.zeros_init()
            linear_kernel_init = nnx.initializers.xavier_uniform()
            linear_bias_init = nnx.initializers.zeros_init()
            layer_norm2_bias_init = nnx.initializers.zeros_init()

        self.mha = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.embed_dim,
            kernel_init=mha_kernel_init,
            bias_init=mha_bias_init,
            rngs=rngs
        )
        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.layer_norm1 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=config.embed_dim,
            scale_init=layer_norm_scale_init,
            bias_init=layer_norm1_bias_init,
            rngs=rngs
        )
        self.linear1 = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.feed_forward_dim,
            kernel_init=linear_kernel_init,
            bias_init=linear_bias_init,
            rngs=rngs
        )
        self.linear2 = nnx.Linear(
            in_features=config.feed_forward_dim,
            out_features=config.embed_dim,
            kernel_init=linear_kernel_init,
            bias_init=linear_bias_init,
            rngs=rngs
        )
        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=config.embed_dim,
            scale_init=layer_norm_scale_init,
            bias_init=layer_norm2_bias_init,
            rngs=rngs
        )

    def __call__(self, inputs, attention_mask=None, training: bool = False):
        batch_size, seq_len, _ = inputs.shape
        causal_mask = causal_attention_mask(seq_len)

        if attention_mask is not None:
            # The attention_mask from the input is broadcastable to the causal_mask,
            # so we can combine them with a logical AND.
            # (batch_size, seq_len) -> (batch_size, 1, seq_len)
            padding_mask = attention_mask[:, None, :]
            combined_mask = causal_mask & padding_mask
        else:
            combined_mask = causal_mask

        attention_output = self.mha(inputs_q=inputs, mask=combined_mask, decode=False)
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = self.layer_norm1(inputs + attention_output)
        
        ffn_output = self.linear1(out1)
        ffn_output = nnx.relu(ffn_output)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        return self.layer_norm2(out1 + ffn_output)
