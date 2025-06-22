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
            use_softermax = config.use_softermax,
            power = config.power,
            dropconnect_rate=config.dropconnect_rate,
            kernel_init=nnx.with_partitioning(nnx.initializers.orthogonal(), NamedSharding(mesh, P(None, 'model'))),
            alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )
        self.dropout1 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        self.non_linear1 = YatNMN(
            in_features=config.embed_dim,
            # out_features=config.feed_forward_dim,
            out_features=config.embed_dim,
            use_dropconnect=config.use_dropconnect,
            drop_rate=config.dropconnect_rate,
            kernel_init=nnx.with_partitioning(nnx.initializers.orthogonal(), NamedSharding(mesh, P(None, 'model'))),
            alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )
        # self.non_linear2 = YatNMN(
        #     in_features=config.feed_forward_dim,
        #     out_features=config.embed_dim,
        #     use_dropconnect=config.use_dropconnect,
        #     drop_rate=config.dropconnect_rate,
        #     kernel_init=nnx.with_partitioning(nnx.initializers.orthogonal(), NamedSharding(mesh, P(None, 'model'))),
        #     alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
        #     bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
        #     rngs=rngs
        # )

        self.dropout2 = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)


    def __call__(self, inputs, training: bool = False):
        _, seq_len, _ = inputs.shape
        mask = causal_attention_mask(seq_len)
        attention_output = self.mha(inputs_q=inputs, mask=mask, decode=False, deterministic=not training)
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = inputs + attention_output
        
        ffn_output = self.non_linear1(out1)
        # ffn_output = self.non_linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        return out1 + ffn_output