import jax.numpy as jnp
import chex
from typing import Union

def softermax_cross_entropy_with_integer_labels(
    logits: chex.Array,
    labels: chex.Array,
    n: float = 1.0,
    epsilon: float = 1e-6,
    axis: int = -1
) -> chex.Array:
    """
    Softermax cross-entropy loss with integer labels.

    Args:
        logits: Non-negative logits, shape [..., num_classes].
        labels: Integer labels, shape matching logits except at `axis`.
        n: Sharpness exponent (n > 0).
        epsilon: Stability constant.
        axis: Axis of the class dimension.

    Returns:
        Loss array of shape [...], reduced along `axis`.
    """
    chex.assert_type([logits], float)
    chex.assert_type([labels], int)
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)

    # Bring class axis to the last dimension for easier indexing
    logits = jnp.moveaxis(logits, axis, -1)
    logits_pow = jnp.power(logits + 1e-12, n)
    logits_sum = jnp.sum(logits_pow, axis=-1)

    # Gather correct class logits
    gather_indices = jnp.expand_dims(labels, axis=-1)
    correct_logits = jnp.take_along_axis(logits, gather_indices, axis=-1).squeeze(axis=-1)

    loss = -n * jnp.log(correct_logits + 1e-12) + jnp.log(logits_sum + epsilon)
    return loss
