import flax.nnx as nnx
from jax.sharding import Mesh

from config import ModelConfig

_MODELS = {}

def register_model(name: str):
    def decorator(cls):
        _MODELS[name] = cls
        return cls
    return decorator

def create_model(name: str, config: ModelConfig, mesh: Mesh, *, rngs: nnx.Rngs):
    if name not in _MODELS:
        raise ValueError(f"Unknown model name: {name}")
    return _MODELS[name](config, mesh, rngs=rngs)

# Import models to register them
from models.linear import mingpt 
from models.aether import aethergpt