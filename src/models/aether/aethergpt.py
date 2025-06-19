import flax.nnx as nnx
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from config import ModelConfig
from model import register_model
from models.aether.base import TransformerBlock
from models.gpt import GPT, TokenAndPositionEmbedding
from nmn.nnx.nmn import YatNMN

@register_model("mini-aethergpt")
class MiniGPT(GPT):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        super().__init__(config, mesh)
        self.embedding_layer = TokenAndPositionEmbedding(config, rngs=rngs)
        self.transformer_blocks = [
            TransformerBlock(config, mesh, rngs=rngs) for _ in range(config.num_transformer_blocks)
        ]
        self.output_layer = YatNMN(
            in_features=config.embed_dim,
            out_features=config.vocab_size,
            kernel_init=nnx.with_partitioning(nnx.initializers.orthogonal(), NamedSharding(mesh, P(None, 'model'))),
            alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )

    def __call__(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        return self.output_layer(x)

@register_model("micro-aethergpt")
class MicroGPT(GPT):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        super().__init__(config, mesh)
        self.embedding_layer = TokenAndPositionEmbedding(config, rngs=rngs)
        self.transformer_block = TransformerBlock(config, mesh, rngs=rngs)
        self.output_layer = YatNMN(
            in_features=config.embed_dim,
            out_features=config.vocab_size,
            kernel_init=nnx.with_partitioning(nnx.initializers.orthogonal(), NamedSharding(mesh, P(None, 'model'))),
            alpha_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            rngs=rngs
        )

    def __call__(self, inputs, training: bool = False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x, training=training)
        return self.output_layer(x) 