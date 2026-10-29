"""Evaluate a flowsheet across a grid of parameter values, with gradients.

:func:`sweep` solves the flowsheet at every point of a grid over one or
more :class:`SweepAxis` (``"<unit>.<param>"`` or ``"feed:<stream>.<field>"``)
with ``jax.vmap``, and records each output and its exact derivative with
respect to every axis --- difflow has them for free. The result is a
:class:`SweepResult`: plain arrays, ready to plot or to ship.

    result = sweep(fs, axes=[SweepAxis("reactor.V", 0.2, 5.0)],
                   outputs={"product": lambda s: get_flows(s["out"])["B"]})

(The browser app that comes with the difflow book turns a sweep into a
self-contained interactive page; the computation is here.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from difflow.params_mixin import ParamsMixin


@dataclass
class SweepAxis(ParamsMixin):
    """One parameter to vary, and the range to vary it over.

    Attributes:
        key: flowsheet parameter, in the dot notation
            :meth:`~difflow.flowsheet.Flowsheet.make_objective_fn` uses
            --- ``"<unit>.<param>"``, e.g. ``"reactor.V"``.
        lo, hi: inclusive bounds.
        n: grid points. The page interpolates between them, so this
            sets how faithful it is; 21 is usually plenty for a smooth
            response.
        label: axis label; defaults to ``key``.
        units: shown after the value.
    """

    key: str
    lo: float
    hi: float
    n: int = 21
    label: str | None = None
    units: str = ""

    def __post_init__(self):
        if self.n < 2:
            raise ValueError(f"axis {self.key!r} needs at least 2 points")
        if not self.hi > self.lo:
            raise ValueError(f"axis {self.key!r}: hi must exceed lo")

    @property
    def title(self) -> str:
        """The label shown on the page, defaulting to the key."""
        return self.label or self.key

    def values(self):
        """The grid points, inclusive of both bounds."""
        return jnp.linspace(self.lo, self.hi, self.n)


@dataclass
class SweepResult(ParamsMixin):
    """A flowsheet evaluated over a grid of parameter values.

    Attributes:
        axes: the parameters that were varied.
        values: output name -> array of shape ``(n_1, ..., n_k)``.
        gradients: output name -> axis key -> array of the same shape,
            the exact derivative at each grid point.
        units: output name -> units, for display.
        baseline: the parameter values the flowsheet started from.
    """

    axes: list[SweepAxis]
    values: dict[str, Any]
    gradients: dict[str, dict[str, Any]] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape, one dimension per axis."""
        return tuple(axis.n for axis in self.axes)

    @property
    def n_points(self) -> int:
        """Total operating points solved --- the product of the axes."""
        n = 1
        for axis in self.axes:
            n *= axis.n
        return n

    def to_dict(self) -> dict:
        """Plain data, ready for JSON and the page."""
        return {
            "axes": [
                {
                    "key": a.key, "label": a.title, "units": a.units,
                    "lo": a.lo, "hi": a.hi, "n": a.n,
                    "values": [float(v) for v in a.values()],
                }
                for a in self.axes
            ],
            "outputs": [
                {
                    "name": name,
                    "units": self.units.get(name, ""),
                    "values": jnp.asarray(grid).tolist(),
                    "gradients": {
                        key: jnp.asarray(g).tolist()
                        for key, g in self.gradients.get(name, {}).items()
                    },
                }
                for name, grid in self.values.items()
            ],
        }


def sweep(
    flowsheet,
    axes: Sequence[SweepAxis],
    outputs: dict[str, Callable],
    *,
    units: dict[str, str] | None = None,
    gradients: bool = True,
    batch: bool = True,
) -> SweepResult:
    """Evaluate a flowsheet across a grid of parameter values.

    Args:
        flowsheet: the flowsheet to sweep.
        axes: parameters to vary. One or two read best on a page; more
            work, but the reader has that many sliders to think about.
        outputs: name -> ``fn(streams) -> scalar``, the quantities to
            record.
        units: name -> units, for display.
        gradients: also record the exact derivative of each output with
            respect to each axis at every grid point.
        batch: evaluate with ``jax.vmap``. Turn off to fall back to a
            plain loop, which is slower but easier to debug.

    Returns:
        A :class:`SweepResult`.

    Raises:
        ValueError: if no axes or no outputs are given.
    """
    axes = list(axes)
    if not axes:
        raise ValueError("a sweep needs at least one axis")
    if not outputs:
        raise ValueError("a sweep needs at least one output")

    keys = [axis.key for axis in axes]
    mesh = jnp.meshgrid(*[axis.values() for axis in axes], indexing="ij")
    flat = [m.reshape(-1) for m in mesh]
    shape = tuple(axis.n for axis in axes)

    values: dict[str, Any] = {}
    grads: dict[str, dict[str, Any]] = {}

    for name, output_fn in outputs.items():
        objective = flowsheet.make_objective_fn(output_fn)

        def at(point, objective=objective):
            return objective({k: point[i] for i, k in enumerate(keys)})

        stacked = jnp.stack(flat, axis=1)          # (n_points, n_axes)
        if batch:
            values[name] = jax.vmap(at)(stacked).reshape(shape)
        else:
            values[name] = jnp.asarray(
                [at(p) for p in stacked]
            ).reshape(shape)

        if gradients:
            def grad_at(point, objective=objective):
                return jax.grad(
                    lambda p: objective({k: p[i] for i, k in enumerate(keys)})
                )(point)

            if batch:
                g = jax.vmap(grad_at)(stacked)
            else:
                g = jnp.stack([grad_at(p) for p in stacked])
            grads[name] = {
                key: g[:, i].reshape(shape) for i, key in enumerate(keys)
            }

    return SweepResult(
        axes=axes, values=values, gradients=grads, units=units or {},
    )
