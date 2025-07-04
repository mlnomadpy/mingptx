import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from config import ModelConfig
from nmn.nnx.yatattention import MultiHeadAttention
from nmn.nnx.nmn import YatNMN
from nmn.nnx.squashers.softer_sigmoid import softer_sigmoid

def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class TransformerBlock(nnx.Module):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        self.use_activation = config.use_activation
        self.use_linear_out = config.use_linear_out
        # Handle partitioning based on whether mesh is available
        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.orthogonal(), NamedSharding(mesh, P(None, 'model')))
            alpha_init = nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model')))
            bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model')))
        else:
            kernel_init = nnx.initializers.orthogonal()
            alpha_init = nnx.initializers.ones_init()
            bias_init = nnx.initializers.zeros_init()

        self.mha = MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.embed_dim,
            use_dropconnect=config.use_dropconnect,
            use_softermax = config.use_softermax,
            power = config.power,
            dropconnect_rate=config.dropconnect_rate,
            kernel_init=kernel_init,
            alpha_init=alpha_init,
            bias_init=bias_init,
            rngs=rngs
        )
        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        self.non_linear1 = YatNMN(
            in_features=config.embed_dim,
            out_features=config.embed_dim,
            use_dropconnect=config.use_dropconnect,
            drop_rate=config.dropconnect_rate,
            kernel_init=kernel_init,
            alpha_init=alpha_init,
            bias_init=bias_init,
            rngs=rngs
        )
        self.out_linear1 = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.embed_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs
        )

        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)


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


        attention_output = self.mha(inputs_q=inputs, mask=combined_mask, decode=False, deterministic=not training)
        if self.use_activation:
            attention_output = softer_sigmoid(attention_output)

        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = inputs + attention_output
        
        ffn_output = self.non_linear1(out1)
        if self.use_activation:
            attention_output = softer_sigmoid(ffn_output)
            
        if self.use_linear_out:
            ffn_output = self.out_linear1(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        return out1 + ffn_output