"""Experiment dataclass for parameter estimation.

An Experiment holds the inputs, observed outputs, and optional uncertainties
for a single experimental measurement used in parameter estimation.

The same object doubles as a *candidate* for experiment design, i.e. a run
that has not happened yet. Such a candidate has inputs and a list of the
outputs that would be measured, but no observed values; build one with
:meth:`Experiment.candidate`. Everything the Fisher information needs --
the model inputs, which outputs are measured, and their 1-sigma
uncertainties -- is known before the run, which is exactly why a design can
be scored in advance.

Note that ``measures`` is the *measurement-set* question, not the
experimental-condition one: when
:func:`~difflow.estimation.check_identifiability` fails, the fix is a new
kind of measurement, and you test a proposed one by adding it to
``measures`` and rerunning the check. The plant-side version of the same
question -- which *instrument* to install -- is
:func:`difflow.reconciliation.design.sensor_ranking`.

Scope: this is difflow's own design-of-experiments path, and it exists for
models that only exist as differentiable JAX functions (a difflow flowsheet,
recycles and all). It selects runs from a **candidate list**. The much
larger ``discopt-doe`` plugin (importable as ``discopt.doe``, a separate
distribution) does continuous design optimization over a box, profile
likelihood, model discrimination, estimability ranking, classical and
screening designs and Bayesian optimization -- but it needs a symbolic
``discopt.modeling`` model and cannot ingest a JAX flowsheet. See
``docs/experiment-design.md`` for the division of labour and when to reach
for which.
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
        measured: Names of the outputs that are (or would be) measured.
            Only needed for a *candidate* experiment, whose ``observed``
            is empty because the run has not happened yet. When ``None``
            the measured outputs are the keys of ``observed``.
    """

    inputs: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, float | Array] = field(default_factory=dict)
    uncertainties: dict[str, float | Array] | None = None
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    measured: list[str] | None = None

    @classmethod
    def candidate(
        cls,
        inputs: dict[str, Any],
        measures: list[str],
        uncertainties: dict[str, float | Array] | None = None,
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "Experiment":
        """Build an experiment that has not been run yet.

        Args:
            inputs: Model inputs for the proposed run.
            measures: Names of the outputs that would be measured.
            uncertainties: Optional 1-sigma uncertainties per measured
                output; missing entries default to 1.0.
            name: Optional label.
            metadata: Optional extra information.

        Returns:
            An Experiment with empty ``observed`` and ``measured`` set.

        Example:
            >>> c = Experiment.candidate({'T': 320.0}, ['y'], {'y': 0.05})
            >>> c.measured_names
            ['y']
            >>> c.is_candidate
            True
        """
        return cls(
            inputs=dict(inputs),
            observed={},
            uncertainties=uncertainties,
            name=name,
            metadata=dict(metadata or {}),
            measured=list(measures),
        )

    @property
    def is_candidate(self) -> bool:
        """True when nothing has been observed yet (a proposed run)."""
        return not self.observed

    @property
    def output_names(self) -> list[str]:
        """Names of observed outputs."""
        return list(self.observed.keys())

    @property
    def measured_names(self) -> list[str]:
        """Names of the outputs that are or would be measured.

        Falls back to ``output_names`` when ``measured`` is not set, so a
        recorded experiment and a candidate are handled uniformly.
        """
        if self.measured is not None:
            return list(self.measured)
        return self.output_names

    @property
    def sigma_array(self) -> Array:
        """1-sigma uncertainties over ``measured_names`` (ones if unknown)."""
        names = self.measured_names
        if self.uncertainties is None:
            return jnp.ones(len(names))
        return jnp.array([float(self.uncertainties.get(k, 1.0)) for k in names])

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
