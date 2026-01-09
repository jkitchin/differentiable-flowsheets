"""Numerical utilities for safe, differentiable operations.

This module provides safe versions of common mathematical operations
that prevent numerical issues (NaN, Inf) during JAX autodifferentiation.

All functions are JAX-compatible and can be JIT-compiled.
"""

import jax
import jax.numpy as jnp
from jax import Array

from difflow.constants import EPS_DIVISION, EPS_LOG, EPS_SQRT


@jax.jit
def safe_divide(
    numerator: Array,
    denominator: Array,
    eps: float = EPS_DIVISION,
) -> Array:
    """Safe division that prevents division by zero.

    When |denominator| < eps, uses eps with the sign of denominator.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        eps: Minimum absolute value for denominator

    Returns:
        numerator / safe_denominator

    Example:
        >>> safe_divide(1.0, 0.0)  # Returns 1e10 instead of inf
        >>> safe_divide(1.0, -1e-15)  # Returns -1e10, preserving sign
    """
    # Preserve sign while ensuring minimum magnitude
    safe_denom = jnp.where(
        jnp.abs(denominator) < eps,
        jnp.sign(denominator + eps) * eps,
        denominator,
    )
    return numerator / safe_denom


@jax.jit
def safe_log(x: Array, eps: float = EPS_LOG) -> Array:
    """Safe logarithm that prevents log(0) and log(negative).

    Clips input to minimum of eps before taking log.

    Args:
        x: Input array
        eps: Minimum value for input

    Returns:
        log(max(x, eps))

    Example:
        >>> safe_log(0.0)  # Returns log(1e-15) ≈ -34.5 instead of -inf
        >>> safe_log(-1.0)  # Returns log(1e-15) instead of NaN
    """
    return jnp.log(jnp.maximum(x, eps))


@jax.jit
def safe_sqrt(x: Array, eps: float = EPS_SQRT) -> Array:
    """Safe square root that prevents sqrt(negative).

    Clips input to minimum of eps before taking sqrt.

    Args:
        x: Input array
        eps: Minimum value for input

    Returns:
        sqrt(max(x, eps))

    Example:
        >>> safe_sqrt(-1.0)  # Returns sqrt(1e-20) ≈ 1e-10 instead of NaN
    """
    return jnp.sqrt(jnp.maximum(x, eps))


@jax.jit
def safe_power(base: Array, exponent: Array, eps: float = EPS_SQRT) -> Array:
    """Safe power operation for non-integer exponents.

    For fractional exponents, clips base to eps to avoid negative bases.

    Args:
        base: Base array
        exponent: Exponent array
        eps: Minimum value for base

    Returns:
        max(base, eps) ** exponent
    """
    safe_base = jnp.maximum(base, eps)
    return jnp.power(safe_base, exponent)


@jax.jit
def safe_exp(x: Array, max_exp: float = 100.0) -> Array:
    """Safe exponential that prevents overflow.

    Clips exponent to prevent exp overflow.

    Args:
        x: Exponent array
        max_exp: Maximum allowed exponent (default 100, exp(100) ≈ 2.7e43)

    Returns:
        exp(clip(x, -max_exp, max_exp))
    """
    return jnp.exp(jnp.clip(x, -max_exp, max_exp))


@jax.jit
def smooth_max(a: Array, b: Array, alpha: float = 10.0) -> Array:
    """Differentiable approximation to max(a, b).

    Uses log-sum-exp for smooth, differentiable maximum.

    Args:
        a: First array
        b: Second array
        alpha: Smoothing parameter (larger = sharper, more like true max)

    Returns:
        Smooth approximation to max(a, b)
    """
    # log-sum-exp trick for numerical stability
    m = jnp.maximum(a, b)
    return m + jnp.log(jnp.exp(alpha * (a - m)) + jnp.exp(alpha * (b - m))) / alpha


@jax.jit
def smooth_min(a: Array, b: Array, alpha: float = 10.0) -> Array:
    """Differentiable approximation to min(a, b).

    Uses negative log-sum-exp for smooth, differentiable minimum.

    Args:
        a: First array
        b: Second array
        alpha: Smoothing parameter (larger = sharper, more like true min)

    Returns:
        Smooth approximation to min(a, b)
    """
    return -smooth_max(-a, -b, alpha)


@jax.jit
def smooth_clamp(x: Array, low: float, high: float, alpha: float = 10.0) -> Array:
    """Differentiable approximation to clamp/clip.

    Args:
        x: Input array
        low: Lower bound
        high: Upper bound
        alpha: Smoothing parameter

    Returns:
        Smooth approximation to clip(x, low, high)
    """
    return smooth_min(smooth_max(x, jnp.asarray(low), alpha), jnp.asarray(high), alpha)


@jax.jit
def safe_arccos(x: Array, eps: float = 1e-7) -> Array:
    """Safe arccosine that handles edge cases.

    Clips input to [-1+eps, 1-eps] to avoid derivative issues at boundaries.

    Args:
        x: Input array (should be in [-1, 1])
        eps: Margin from boundaries

    Returns:
        arccos(clip(x, -1+eps, 1-eps))
    """
    return jnp.arccos(jnp.clip(x, -1.0 + eps, 1.0 - eps))
