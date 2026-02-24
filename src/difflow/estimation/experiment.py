"""Experiment dataclass for parameter estimation.

An Experiment holds the inputs, observed outputs, and optional uncertainties
for a single experimental measurement used in parameter estimation.
"""

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin


@dataclass
class Experiment(ParamsMixin):
    """A single experimental observation for parameter estimation.

    Attributes:
        inputs: Model inputs (flexible dict, passed to user's model function)
        observed: Measured outputs {name: value}
        uncertainties: Optional 1-sigma measurement uncertainties per output
        name: Optional experiment label
        metadata: Optional extra information
    """

    inputs: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, float | Array] = field(default_factory=dict)
    uncertainties: dict[str, float | Array] | None = None
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output_names(self) -> list[str]:
        """Names of observed outputs."""
        return list(self.observed.keys())

    @property
    def observed_array(self) -> Array:
        """Observed values as a JAX array (ordered by output_names)."""
        return jnp.array([float(self.observed[k]) for k in self.output_names])

    @property
    def weights(self) -> Array:
        """Inverse-variance weights for weighted least squares.

        Returns ones if no uncertainties are provided.
        """
        if self.uncertainties is None:
            return jnp.ones(len(self.observed))
        names = self.output_names
        sigmas = jnp.array([float(self.uncertainties.get(k, 1.0)) for k in names])
        return 1.0 / (sigmas ** 2)
