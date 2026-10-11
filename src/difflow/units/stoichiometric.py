"""Fixed-conversion (stoichiometric) reactor.

The block-flow reactor of a material-balance course and of a simulator's
``RStoic`` block: the conversion of a key reactant is *specified*, and the
extent of reaction follows from it. No kinetics, no equilibrium and no
thermodynamics are involved, so it needs no ``thermo`` and is differentiable
in its conversion (and in the feed) everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, make_stream


def _is_concrete_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


@dataclass(repr=False)
class StoichiometricReactorParams(ParamsMixin):
    """Parameters of a :class:`StoichiometricReactor`.

    Attributes:
        stoich: Stoichiometric coefficients ``{species: nu}`` of one reaction
            (negative for reactants, positive for products), or a list of
            such dicts for several reactions.
        key: Key reactant whose conversion is specified; one per reaction
            (a list when ``stoich`` is a list). Must have a negative
            coefficient in its reaction.
        conversion: Fractional conversion of the key reactant, ``0 <= X <= 1``,
            based on the reactor inlet; one per reaction (a list or array
            when ``stoich`` is a list). May be a JAX tracer.
        T_out: Outlet temperature (K); ``None`` keeps the inlet temperature.
            No energy balance is solved either way.
        pressure_drop: Pressure drop (Pa) across the reactor.
    """

    stoich: dict | list
    key: str | list
    conversion: float | list
    T_out: float | None = None
    pressure_drop: float = 0.0

    def __post_init__(self):
        reactions = self.stoich if isinstance(self.stoich, (list, tuple)) else [self.stoich]
        keys = self.key if isinstance(self.key, (list, tuple)) else [self.key]
        if not reactions or not all(isinstance(r, dict) and r for r in reactions):
            raise ValueError("stoich must be a non-empty {species: nu} dict or a list of them")
        if len(keys) != len(reactions):
            raise ValueError(
                f"one key per reaction: got {len(keys)} key(s) for {len(reactions)} reaction(s)"
            )
        for r, (nu, k) in enumerate(zip(reactions, keys)):
            if k not in nu:
                raise ValueError(f"key {k!r} does not appear in reaction {r}: {nu}")
            if _is_concrete_number(nu[k]) and not float(nu[k]) < 0:
                raise ValueError(
                    f"key {k!r} must be a reactant (negative coefficient) in reaction {r}, "
                    f"got {nu[k]}"
                )
        X = jnp.atleast_1d(jnp.asarray(self.conversion))
        if X.shape[0] != len(reactions):
            raise ValueError(
                f"one conversion per reaction: got {X.shape[0]} for {len(reactions)} reaction(s)"
            )
        if _is_concrete_number(jnp.min(X)) and (float(jnp.min(X)) < 0 or float(jnp.max(X)) > 1):
            raise ValueError(f"conversion must lie in [0, 1], got {self.conversion}")


class StoichiometricReactor:
    """Fixed-conversion reactor: specified conversion of a key reactant.

    ``StoichiometricReactor(params)(inlet) -> (outlet, info)``. For each
    reaction ``r`` with key reactant ``k`` and conversion ``X_r``, the extent
    is ``xi_r = X_r F_k,in / (-nu_k,r)`` and every species leaves at
    ``F_i,out = F_i,in + sum_r nu_i,r xi_r``. Several reactions are taken in
    *parallel*: each conversion is based on the reactor inlet, as in a
    simulator's ``RStoic`` block, so the conversions of reactions sharing a
    key reactant must not add up to more than one.

    Products that are not in the inlet stream are added to the outlet. The
    outlet is at ``T_out`` (default: the inlet temperature) and the inlet
    pressure less ``pressure_drop``; no energy balance is solved, so the unit
    needs no thermodynamic model.

    ``info``: ``extent`` (array, mol/s, one per reaction), ``conversion``
    (array), ``min_outlet_flow`` (mol/s; negative means the specified
    conversions consume more of some species than is fed) and
    ``key_consumed`` (mol/s of each key reactant consumed, all reactions).

    Example:
        >>> from difflow import make_stream, StoichiometricReactor, StoichiometricReactorParams
        >>> rx = StoichiometricReactor(StoichiometricReactorParams(
        ...     stoich={"CO": -1, "H2": -2, "MeOH": 1}, key="CO", conversion=0.25))
        >>> out, info = rx(make_stream({"CO": 100.0, "H2": 200.0}, T=500.0, P=5e6))
        >>> round(float(out["F_MeOH"]), 6), round(float(out["F_H2"]), 6)
        (25.0, 150.0)
    """

    symbol = "Stoichiometric Reactor"
    equations = [
        r"\xi_r = \frac{X_r\, F_{k_r}^{\mathrm{in}}}{-\nu_{k_r,r}}",
        r"F_i^{\mathrm{out}} = F_i^{\mathrm{in}} + \sum_r \nu_{i,r}\, \xi_r",
        r"T^{\mathrm{out}} = T^{\mathrm{in}}\ (\text{or } T_{\mathrm{out}}),\quad P^{\mathrm{out}} = P^{\mathrm{in}} - \Delta P",
    ]
    assumptions = [
        "Conversion of the key reactant is specified; no kinetics or equilibrium.",
        "Several reactions run in parallel, each conversion based on the reactor inlet.",
        "No energy balance: the outlet is at the inlet temperature unless T_out is given.",
    ]
    references = [
        "Felder, Rousseau, Bullard. Elementary Principles of Chemical Processes, 4e, Ch. 4.",
        "Himmelblau, Riggs. Basic Principles and Calculations in Chemical Engineering, 8e, Ch. 4.",
    ]
    parameter_symbols = {"conversion": "X", "key": "k", "stoich": r"\nu", "pressure_drop": r"\Delta P"}
    parameter_units = {"conversion": "-", "T_out": "K", "pressure_drop": "Pa", "stoich": "-"}
    numerical_method = "Closed form (no iteration)."

    def __init__(self, params: StoichiometricReactorParams):
        """Build the reactor.

        Args:
            params: The reaction(s), key reactant(s) and conversion(s).
        """
        self.params = params

    def _reactions(self) -> tuple[list[dict], list[str], Array]:
        p = self.params
        reactions = list(p.stoich) if isinstance(p.stoich, (list, tuple)) else [p.stoich]
        keys = list(p.key) if isinstance(p.key, (list, tuple)) else [p.key]
        X = jnp.atleast_1d(jnp.asarray(p.conversion, dtype=float))
        return reactions, keys, X

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        """React the inlet at the specified conversion(s).

        Args:
            inlet: Feed stream.

        Returns:
            outlet: Reactor effluent.
            info: Extents, conversions and a feasibility diagnostic.
        """
        reactions, keys, X = self._reactions()
        flows = dict(get_flows(inlet))
        for nu in reactions:
            for s in nu:
                flows.setdefault(s, jnp.zeros(()))
        inlet_flows = dict(flows)

        extents = []
        for r, (nu, k) in enumerate(zip(reactions, keys)):
            extents.append(X[r] * inlet_flows[k] / (-jnp.asarray(nu[k], dtype=float)))
        out = {}
        for s, F in inlet_flows.items():
            change = sum(
                (jnp.asarray(nu[s], dtype=float) * xi
                 for nu, xi in zip(reactions, extents) if s in nu),
                jnp.zeros(()),
            )
            out[s] = F + change

        p = self.params
        T = inlet["T"] if p.T_out is None else jnp.asarray(p.T_out, dtype=float)
        P = inlet["P"] - p.pressure_drop
        extent_arr = jnp.stack(extents)
        info = {
            "extent": extent_arr,
            "conversion": X,
            "min_outlet_flow": jnp.min(jnp.stack(list(out.values()))),
            "key_consumed": {
                k: inlet_flows[k] - out[k] for k in dict.fromkeys(keys)
            },
        }
        return make_stream(out, T, P), info
