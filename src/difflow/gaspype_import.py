"""Gaspype thermodynamic data for difflow's :class:`~difflow.GibbsReactor` (#492).

`Gaspype <https://www.gaspype.org/>`_ (DLR, MIT licence) ships NASA-9
ideal-gas polynomials for about a thousand gas-phase species and its own
equilibrium solver, with NumPy or JAX as array backend. It is an
**optional** dependency::

    pip install gaspype            # or:  pip install "difflow[gaspype]"

What this adapter does
----------------------
:class:`GaspypeThermo` turns a list of gaspype species into the thermo object
:class:`~difflow.GibbsReactor` takes (``names``, ``gibbs(T)``,
``enthalpy(T)``, ``ELEMENTS``). difflow then solves the equilibrium with its
element-potential Newton, whose gradients come from implicit differentiation,
so ``jax.jit(jax.grad(...))`` works through the reactor, inside a flowsheet
or an optimisation.

Gaspype's own ``gaspype.equilibrium`` with ``backend="jax"`` differentiates
too, but only eagerly: under ``jax.jit`` its Newton loop becomes a
``lax.while_loop``, which reverse mode cannot differentiate, so
``jax.jit(jax.grad(f))`` raises. Unrolled eagerly it is also slow (seconds
per gradient). Use gaspype directly for vectorised forward sweeps and this
adapter where difflow needs a gradient.

Two ways to evaluate the data (``method``)
------------------------------------------
* ``"nasa9"`` (default): the NASA-9 polynomials, evaluated in ``jax.numpy``
  from the coefficients in gaspype's database. Smooth in T within each
  interval: ``dH/dT`` is the polynomial Cp. Where two intervals meet (most
  species at 1000 K), the fits disagree slightly, and G steps by ~0.01-0.1
  J/mol; a finite difference straddling a join measures that step. A T outside a species' data range uses the nearest
  interval's polynomial (extrapolation) rather than NaN. The coefficients are
  read through gaspype's private database reader (checked against gaspype
  1.1); the data are stored as float32, so H and G carry ~1e-7 relative
  round-off.
* ``"table"``: gaspype's public ``fluid_system`` with the JAX backend, which
  interpolates a 1 K table linearly between ``t_min`` and ``t_max``. Uses
  only gaspype's public API, but ``dH/dT`` and ``dG/dT`` are piecewise
  constant, so derivatives with respect to T jump at every whole kelvin.

``GaspypeThermo`` is also exported as ``difflow.GaspypeThermo``. It is a
dataclass of its constructor arguments, so ``difflow.serialize`` and
``difflow.codegen`` write it as the species list and ``method``.

Usage::

    from difflow import GaspypeThermo, GibbsReactor, GibbsReactorParams, make_stream

    th = GaspypeThermo(["CH4", "H2O", "H2", "CO", "CO2"])
    out, info = GibbsReactor(GibbsReactorParams(thermo=th, T_out=1000.0))(
        make_stream({"CH4": 1.0, "H2O": 3.0}, T=900.0, P=1e5))
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

R_GAS = 8.314462618  # J/mol/K, as in difflow.units.gibbs (gaspype's kB*NA)


def _require_gaspype():
    """Import gaspype, or raise an ImportError that says how to install it."""
    try:
        import gaspype
    except ImportError as exc:  # pragma: no cover - exercised only without gaspype
        raise ImportError(
            "difflow.gaspype_import needs gaspype: pip install gaspype "
            '(or pip install "difflow[gaspype]")'
        ) from exc
    return gaspype


def list_available_species(pattern: str = "*", elements: str | list[str] = ()) -> list[str]:
    """Gas-phase species in gaspype's database.

    Args:
        pattern: gaspype's pattern: ``#`` any count (``'C#H#'`` matches
            ``'CH4'``), ``$`` any element, ``*`` anything.
        elements: Keep only species made of these elements (list or comma
            separated string).

    Returns:
        Species names, as :class:`GaspypeThermo` takes them.

    Example:
        >>> from difflow.gaspype_import import list_available_species
        >>> "CH4" in list_available_species("C#H#")    # doctest: +SKIP
        True
    """
    gp = _require_gaspype()
    return gp.species(pattern, element_names=list(elements) if not isinstance(elements, str)
                      else elements)


def _nasa9_data(names: tuple) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Compositions, interval bounds ``(n, k+1)`` and coefficients ``(n, k, 9)``.

    Species with fewer intervals than the most are padded by repeating their
    last interval, so one array holds them all.
    """
    _require_gaspype()
    try:
        from gaspype._main import _species_db
    except ImportError as exc:  # pragma: no cover - depends on gaspype's internals
        raise ImportError(
            "this gaspype version does not expose its species database; "
            'use GaspypeThermo(..., method="table")'
        ) from exc
    records = []
    for s in names:
        d = _species_db.read(s)
        if d is None:
            raise ValueError(f"species {s!r} is not in gaspype's gas-phase database")
        if d.model != 9:  # pragma: no cover - gaspype stores NASA-9 only
            raise ValueError(f"species {s!r} is not a NASA-9 record")
        records.append(d)
    k = max(len(d.data) for d in records)
    bounds = np.zeros((len(names), k + 1))
    coeffs = np.zeros((len(names), k, 9))
    for i, d in enumerate(records):
        t = list(d.t_range) + [d.t_range[-1]] * (k + 1 - len(d.t_range))
        c = [list(r) for r in d.data] + [list(d.data[-1])] * (k - len(d.data))
        bounds[i] = t
        coeffs[i] = c
    return [dict(d.composition) for d in records], bounds, coeffs


@dataclass
class GaspypeThermo:
    """Ideal-gas standard-state data from gaspype, for :class:`~difflow.GibbsReactor`.

    Attributes:
        species: gaspype gas-phase species names (see
            :func:`list_available_species`); they become the stream keys
            ``F_<name>``.
        method: ``"nasa9"`` (smooth polynomials, default) or ``"table"``
            (gaspype's public 1 K table); see the module docstring.
        t_min: Lower end of the ``"table"`` range (K).
        t_max: Upper end of the ``"table"`` range (K).

    Example:
        >>> from difflow.gaspype_import import GaspypeThermo
        >>> th = GaspypeThermo(["H2", "O2", "H2O"])         # doctest: +SKIP
        >>> th.enthalpy(298.15)                            # doctest: +SKIP
    """

    species: tuple
    method: str = "nasa9"
    t_min: int = 250
    t_max: int = 2000

    def __post_init__(self):
        if isinstance(self.species, str):
            raise TypeError("species must be a list of names (gaspype names may contain commas)")
        self.species = tuple(self.species)
        if not self.species:
            raise ValueError("species is empty")
        if len(set(self.species)) != len(self.species):
            raise ValueError("species are repeated")
        if self.method not in ("nasa9", "table"):
            raise ValueError(f"method must be 'nasa9' or 'table', not {self.method!r}")
        if self.method == "nasa9":
            comps, self._bounds, self._coeffs = _nasa9_data(self.species)
            self._fs = None
        else:
            gp = _require_gaspype()
            try:
                self._fs = gp.fluid_system(list(self.species), self.t_min, self.t_max, backend="jax")
            except Exception as exc:
                raise ValueError(str(exc)) from exc
            comps = self._fs._species_compositions
        els = sorted({e for c in comps for e in c})
        self.ELEMENTS = {e: np.array([c.get(e, 0) for c in comps], dtype=float) for e in els}

    @property
    def names(self) -> tuple:
        """Species names, in array order."""
        return self.species

    def _nasa9(self, T):
        """``(h/RT, s/R)`` of every species at a scalar T, ``(n,)`` each."""
        T = jnp.asarray(T, dtype=float)
        bounds = jnp.asarray(self._bounds)
        # interval i holds T in [bounds[i], bounds[i+1]]; clip to extrapolate
        k = self._coeffs.shape[1]
        idx = jnp.clip(jnp.sum(T >= bounds[:, 1:-1], axis=1), 0, k - 1)
        a = jnp.take_along_axis(jnp.asarray(self._coeffs), idx[:, None, None], axis=1)[:, 0, :]
        lnT = jnp.log(T)
        h_rt = (-a[:, 0] / T ** 2 + a[:, 1] * lnT / T + a[:, 2] + a[:, 3] * T / 2
                + a[:, 4] * T ** 2 / 3 + a[:, 5] * T ** 3 / 4 + a[:, 6] * T ** 4 / 5 + a[:, 7] / T)
        s_r = (-a[:, 0] / (2 * T ** 2) - a[:, 1] / T + a[:, 2] * lnT + a[:, 3] * T
               + a[:, 4] * T ** 2 / 2 + a[:, 5] * T ** 3 / 3 + a[:, 6] * T ** 4 / 4 + a[:, 8])
        return h_rt, s_r

    def enthalpy(self, T):
        """Standard molar enthalpy (J/mol, formation basis), ``(n,)``."""
        if self._fs is not None:
            return self._fs.get_species_h(jnp.asarray(T, dtype=float))
        h_rt, _ = self._nasa9(T)
        return h_rt * R_GAS * jnp.asarray(T, dtype=float)

    def gibbs(self, T):
        """Standard (1 bar) ``H - TS`` (J/mol), ``(n,)``."""
        T = jnp.asarray(T, dtype=float)
        if self._fs is not None:
            return self._fs.get_species_g_rt(T) * R_GAS * T
        h_rt, s_r = self._nasa9(T)
        return (h_rt - s_r) * R_GAS * T
