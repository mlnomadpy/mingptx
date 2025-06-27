import jax
import jax.numpy as jnp
import chex
from typing import Union

def softermax_cross_entropy_with_integer_labels(
    logits: chex.Array,
    labels: chex.Array,
    n: float = 1.0,
    epsilon: float = 1e-9,
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

    # Ensure logits are non-negative.
    logits = jax.nn.relu(logits)

    # Bring class axis to the last dimension for easier indexing
    logits = jnp.moveaxis(logits, axis, -1)

    # Add epsilon before power to ensure stability for zero logits.
    logits_plus_epsilon = logits + epsilon
    logits_pow = jnp.power(logits_plus_epsilon, n)
    logits_sum = jnp.sum(logits_pow, axis=-1)

    # Gather correct class logits
    gather_indices = jnp.expand_dims(labels, axis=-1)
    correct_logits_plus_epsilon = jnp.take_along_axis(logits_plus_epsilon, gather_indices, axis=-1).squeeze(axis=-1)
    
    # The epsilon is already incorporated in logits_sum via logits_pow
    log_normalizers = jnp.log(logits_sum)
    # log(x^n) = n*log(x)
    log_correct_logits = n * jnp.log(correct_logits_plus_epsilon)

    loss = log_normalizers - log_correct_logits
    return loss

def softermax_cross_entropy_with_one_hot_labels(
    logits: chex.Array,
    labels: chex.Array,
    n: float = 1.0,
    epsilon: float = 1e-9,
    axis: int = -1
) -> chex.Array:
    """
    Softermax cross-entropy loss with one-hot labels.

    Args:
        logits: Non-negative logits, shape [..., num_classes].
        labels: One-hot labels, shape matching logits.
        n: Sharpness exponent (n > 0).
        epsilon: Stability constant.
        axis: Axis of the class dimension.

    Returns:
        Loss array of shape [...], reduced along `axis`.
    """
    chex.assert_type([logits, labels], float)
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)

    # Ensure logits are non-negative.
    logits = jax.nn.relu(logits)

    # Bring class axis to the last dimension for easier indexing
    logits = jnp.moveaxis(logits, axis, -1)
    labels = jnp.moveaxis(labels, axis, -1)

    # Add epsilon before power to ensure stability for zero logits.
    logits_plus_epsilon = logits + epsilon
    logits_pow = jnp.power(logits_plus_epsilon, n)
    logits_sum = jnp.sum(logits_pow, axis=-1)

    # Get the correct logit using the one-hot labels
    correct_logits_plus_epsilon = jnp.sum(logits_plus_epsilon * labels, axis=-1)
    
    # The epsilon is already incorporated in logits_sum via logits_pow
    log_normalizers = jnp.log(logits_sum)
    # log(x^n) = n*log(x)
    log_correct_logits = n * jnp.log(correct_logits_plus_epsilon)

    loss = log_normalizers - log_correct_logits
    return loss
