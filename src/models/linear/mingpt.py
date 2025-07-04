import flax.nnx as nnx
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from config import ModelConfig
from model import register_model
from models.linear.base import TransformerBlock
from models.gpt import GPT, TokenAndPositionEmbedding


@register_model("mini-gpt")
class MiniGPT(GPT):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        super().__init__(config, mesh)
        self.embedding_layer = TokenAndPositionEmbedding(config, rngs=rngs)
        self.transformer_blocks = [
            TransformerBlock(config, mesh, rngs=rngs) for _ in range(config.num_transformer_blocks)
        ]
        
        # Handle partitioning based on whether mesh is available
        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model')))
            bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model')))
        else:
            kernel_init = nnx.initializers.xavier_uniform()
            bias_init = nnx.initializers.zeros_init()
            
        self.output_layer = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.vocab_size,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs
        )

    def __call__(self, inputs, attention_mask=None, training: bool = False):
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask=attention_mask, training=training)
        return self.output_layer(x)

@register_model("micro-gpt")
class MicroGPT(GPT):
    def __init__(self, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
        super().__init__(config, mesh)
        self.embedding_layer = TokenAndPositionEmbedding(config, rngs=rngs)
        self.transformer_block = TransformerBlock(config, mesh, rngs=rngs)
        
        # Handle partitioning based on whether mesh is available
        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model')))
            bias_init = nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model')))
        else:
            kernel_init = nnx.initializers.xavier_uniform()
            bias_init = nnx.initializers.zeros_init()
            
        self.output_layer = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.vocab_size,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs
        )

    def __call__(self, inputs, attention_mask=None, training: bool = False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x, attention_mask=attention_mask, training=training)
        return self.output_layer(x) 