"""Liquid-phase activity-coefficient models: Wilson, Margules, van Laar.

Every activity model in difflow follows one small protocol: a pytree with a
``gamma(x, T) -> Array`` method (a plain callable ``f(x, T)`` also works).
:func:`activity_gamma` dispatches on it, so :class:`~difflow.units.flash.Flash`
and the phase-diagram helpers in :mod:`difflow.phase_diagrams` accept any of
:class:`~difflow.units.lle.NRTLParams`, :class:`~difflow.units.lle.UNIQUACParams`,
:class:`WilsonParams`, :class:`MargulesParams` and :class:`VanLaarParams`.

All models are pure JAX and differentiable in ``x``, ``T`` and the parameters.

Notes
-----
Wilson's equation cannot predict liquid-liquid splitting: for any parameters
the Wilson Gibbs energy of mixing is convex, so the model describes a single
liquid phase. Use NRTL or UNIQUAC (or Margules / van Laar, which can) for
partially miscible systems.

References
----------
- Wilson, G.M. J. Am. Chem. Soc. 86, 127 (1964).
- Margules, M. Sitzungsber. Akad. Wiss. Wien, Math.-Naturwiss. Kl. II 104,
  1243 (1895).
- van Laar, J.J. Z. Phys. Chem. 72, 723 (1910); 83, 599 (1913).
- Prausnitz, Lichtenthaler, de Azevedo. Molecular Thermodynamics of
  Fluid-Phase Equilibria, 3e, Ch. 6-7.
- Smith, Van Ness, Abbott. Introduction to Chemical Engineering
  Thermodynamics, 7e, Ch. 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

#: Gas constant (J/mol/K).
R_GAS = 8.314462618


@runtime_checkable
class ActivityModel(Protocol):
    """Protocol of an activity-coefficient model.

    A model is a pytree exposing ``gamma(x, T)`` that returns the activity
    coefficients (same order as ``x``) of a liquid at mole fractions ``x`` and
    temperature ``T`` (K).
    """

    def gamma(self, x: Array, T: Array) -> Array:  # pragma: no cover
        ...


def activity_gamma(model, x: Array, T: Array | float) -> Array:
    """Activity coefficients from any supported activity model.

    Args:
        model: An object with a ``gamma(x, T)`` method (the
            :class:`ActivityModel` protocol, which includes ``NRTLParams`` and
            ``UNIQUACParams``) or a plain callable ``model(x, T)``.
        x: Liquid mole fractions.
        T: Temperature (K).

    Returns:
        Array of activity coefficients, same shape as ``x``.

    Raises:
        TypeError: If ``model`` is neither.
    """
    x = jnp.asarray(x)
    T = jnp.asarray(T)
    g = getattr(model, "gamma", None)
    if callable(g):
        return g(x, T)
    if callable(model):
        return model(x, T)
    raise TypeError(
        f"{type(model).__name__} is not an activity model: it needs a "
        "gamma(x, T) method or must be callable as model(x, T)"
    )


def _binary_check(species: tuple[str, ...], name: str) -> None:
    if len(species) != 2:
        raise ValueError(f"{name} is a binary model; got {len(species)} species")


@partial(jax.tree_util.register_dataclass,
         data_fields=["V", "dlam"], meta_fields=["species"])
@dataclass(frozen=True)
class WilsonParams:
    r"""Wilson activity-coefficient model.

    .. math::
        \Lambda_{ij} = \frac{V_j}{V_i}\exp\!\left(-\frac{\lambda_{ij}-\lambda_{ii}}{RT}\right),
        \qquad
        \ln\gamma_i = 1 - \ln\sum_j x_j\Lambda_{ij}
        - \sum_k \frac{x_k\Lambda_{ki}}{\sum_j x_j\Lambda_{kj}}

    Wilson's equation cannot predict liquid-liquid splitting (its Gibbs
    energy of mixing is always convex); use NRTL/UNIQUAC for partially
    miscible systems.

    Attributes:
        species: Species names, in order.
        V: Pure-liquid molar volumes (n,); any volume unit, only ratios enter.
        dlam: Energy parameters ``lambda_ij - lambda_ii`` (n, n) in J/mol;
            the diagonal is ignored (it is zero by definition).
    """
    species: tuple[str, ...]
    V: Array
    dlam: Array

    @classmethod
    def binary(cls, species, V, dlam12: float, dlam21: float) -> "WilsonParams":
        """Binary Wilson parameters from ``lambda12-lambda11`` and ``lambda21-lambda22`` (J/mol)."""
        _binary_check(tuple(species), "WilsonParams.binary")
        return cls(tuple(species), jnp.asarray(V, dtype=float),
                   jnp.array([[0.0, dlam12], [dlam21, 0.0]]))

    def Lambda(self, T: Array | float) -> Array:
        """The Wilson ``Lambda_ij`` matrix at temperature ``T`` (diagonal 1)."""
        n = len(self.species)
        V = jnp.asarray(self.V)
        dl = jnp.asarray(self.dlam) * (1.0 - jnp.eye(n))
        return (V[None, :] / V[:, None]) * jnp.exp(-dl / (R_GAS * jnp.asarray(T)))

    def gamma(self, x: Array, T: Array | float) -> Array:
        """Activity coefficients at mole fractions ``x`` and temperature ``T`` (K)."""
        L = self.Lambda(T)
        s = L @ x                      # s_k = sum_j x_j L_kj
        return jnp.exp(1.0 - jnp.log(s) - (L.T @ (x / s)))


@partial(jax.tree_util.register_dataclass,
         data_fields=["A12", "A21"], meta_fields=["species"])
@dataclass(frozen=True)
class MargulesParams:
    r"""Binary Margules activity model (two- and three-suffix).

    Three-suffix (two-parameter) form:

    .. math::
        \ln\gamma_1 = x_2^2\left[A_{12} + 2(A_{21}-A_{12})x_1\right],\qquad
        \ln\gamma_2 = x_1^2\left[A_{21} + 2(A_{12}-A_{21})x_2\right]

    With ``A12 == A21 == A`` it reduces to the two-suffix (one-parameter)
    form :math:`\ln\gamma_1 = A x_2^2`, :math:`\ln\gamma_2 = A x_1^2`; build it
    with :meth:`two_suffix`. ``A12 = ln gamma_1^inf`` and
    ``A21 = ln gamma_2^inf``. The parameters are dimensionless and
    temperature-independent. Binary only.

    Attributes:
        species: The two species names.
        A12: ``ln gamma_1`` at infinite dilution of 1 in 2.
        A21: ``ln gamma_2`` at infinite dilution of 2 in 1.
    """
    species: tuple[str, ...]
    A12: Array
    A21: Array

    def __post_init__(self):
        _binary_check(tuple(self.species), "MargulesParams")

    @classmethod
    def two_suffix(cls, species, A: float) -> "MargulesParams":
        """Two-suffix (one-parameter) Margules model, ``ln gamma_1 = A x_2^2``."""
        return cls(tuple(species), jnp.asarray(A, dtype=float), jnp.asarray(A, dtype=float))

    def gamma(self, x: Array, T: Array | float = 0.0) -> Array:
        """Activity coefficients ``[gamma_1, gamma_2]`` (``T`` is unused)."""
        x1, x2 = x[0], x[1]
        A12, A21 = self.A12, self.A21
        lg1 = x2**2 * (A12 + 2.0 * (A21 - A12) * x1)
        lg2 = x1**2 * (A21 + 2.0 * (A12 - A21) * x2)
        return jnp.exp(jnp.stack([lg1, lg2]))


@partial(jax.tree_util.register_dataclass,
         data_fields=["A12", "A21"], meta_fields=["species"])
@dataclass(frozen=True)
class VanLaarParams:
    r"""Binary van Laar activity model.

    .. math::
        \ln\gamma_1 = A_{12}\left(1+\frac{A_{12}x_1}{A_{21}x_2}\right)^{-2}
        = \frac{A_{12}A_{21}^2x_2^2}{(A_{12}x_1+A_{21}x_2)^2},\qquad
        \ln\gamma_2 = \frac{A_{21}A_{12}^2x_1^2}{(A_{12}x_1+A_{21}x_2)^2}

    ``A12 = ln gamma_1^inf`` and ``A21 = ln gamma_2^inf``. The second form
    is used in the code so that the pure-component limits are finite.
    ``A12`` and ``A21`` must have the same sign (otherwise the denominator
    can vanish). Parameters are dimensionless and temperature-independent.
    Binary only.

    Attributes:
        species: The two species names.
        A12: ``ln gamma_1`` at infinite dilution of 1 in 2.
        A21: ``ln gamma_2`` at infinite dilution of 2 in 1.
    """
    species: tuple[str, ...]
    A12: Array
    A21: Array

    def __post_init__(self):
        _binary_check(tuple(self.species), "VanLaarParams")

    def gamma(self, x: Array, T: Array | float = 0.0) -> Array:
        """Activity coefficients ``[gamma_1, gamma_2]`` (``T`` is unused)."""
        x1, x2 = x[0], x[1]
        A12, A21 = self.A12, self.A21
        den = (A12 * x1 + A21 * x2) ** 2
        lg1 = A12 * A21**2 * x2**2 / den
        lg2 = A21 * A12**2 * x1**2 / den
        return jnp.exp(jnp.stack([lg1, lg2]))


#: Models defined here, for introspection.
ACTIVITY_MODELS: dict[str, Callable] = {
    "wilson": WilsonParams,
    "margules": MargulesParams,
    "van_laar": VanLaarParams,
}
