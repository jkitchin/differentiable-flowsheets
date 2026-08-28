"""Declarative mass-action kinetics.

The reactor units take a Python callable::

    rate_fn(C, T, rate_params) -> rates

which is expressive but not *data*: it cannot be written to a file,
built from a form, or round-tripped through a GUI. That callable is the
only thing standing between difflow's unit operations and a fully
declarative model specification --- of the 78 ``Params`` dataclasses in
the package, 73 are already pure data, and the five that are not are
all reactors, all because of this one field.

:func:`mass_action_kinetics` closes that gap. It takes reactions as
plain dictionaries --- the format :func:`difflow.import_reactions`
already produces from Cantera YAML, and equally what a form or a JSON
file would produce --- and returns everything a reactor needs::

    reactions = import_reactions("mech.yaml")
    kin = mass_action_kinetics(reactions, species_order=["A", "B"])
    params = CSTRParams(V=1.0, **kin.params_kwargs())

The rate law is mass action with a modified Arrhenius coefficient,

.. math::

    k_j(T) = A_j \\, T^{n_j} \\exp\\!\\left(-E_{a,j} / R T\\right),
    \\qquad
    r_j = k_j \\prod_i C_i^{\\alpha_{ji}}

with reaction orders taken from the reactant stoichiometry unless given
explicitly. A reversible reaction subtracts the reverse term scaled by
its equilibrium constant,

.. math::

    r_j = k_j \\left( \\prod_i C_i^{\\alpha_{ji}}
          - \\frac{1}{K_{eq,j}} \\prod_i C_i^{\\beta_{ji}} \\right).

Units are difflow's throughout: concentrations mol/m^3, temperature K,
activation energy J/mol, rates mol/m^3/s. Cantera files declare their
own units in a ``units:`` block; a mechanism written in cm^3 or
kcal/mol will import numerically unchanged and be silently wrong, so
check that block before trusting a rate constant.

What this deliberately does *not* do: three-body, falloff and any other
pressure-dependent reaction type is rejected rather than approximated,
and a reversible reaction without an equilibrium constant raises rather
than quietly dropping its reverse term. Both are cases where guessing
would produce a plausible number that is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_log
from difflow.params_mixin import ParamsMixin

#: gas constant, J/(mol K)
R_GAS = 8.314462618

#: reaction types whose kinetics are plain mass action
SUPPORTED_TYPES = frozenset({"elementary", "irreversible", ""})

#: Concentration floor inside the log-space product. Small enough that a
#: reactant at zero contributes an underflowed rate rather than a
#: spurious one (1e-300 to the first power, exactly 0 beyond), while
#: never clipping a concentration any real problem could produce.
#:
#: The alternative, ``prod(C ** order)``, gives exact zeros but a *nan*
#: derivative at C = 0, and one nan poisons the whole reactor solve. A
#: floor 300 orders of magnitude below anything physical is the cheaper
#: error: away from zero both forms agree exactly, and at zero this one
#: reports a derivative of 0 where the true first-order value is k.
CONC_FLOOR = 1e-300

#: how to treat a reaction marked reversible
REVERSE_MODES = ("error", "forward_only", "equilibrium")


class KineticsSpecError(ValueError):
    """A reaction specification cannot be turned into a rate law.

    Raised for unsupported reaction types, species that are not in the
    species order, and reversible reactions whose reverse term is not
    determined. The message names the offending reactions.
    """


@dataclass
class ReactionSet(ParamsMixin):
    """A compiled set of mass-action reactions.

    Everything a reactor needs, derived from data rather than code.

    Attributes:
        species_order: species names, matching the rows of ``stoich``.
        stoich: net stoichiometry, shape ``(n_species, n_reactions)``,
            negative for reactants and positive for products.
        rate_fn: ``rate_fn(C, T, rate_params) -> (n_reactions,)``, the
            callable the reactor units expect.
        rate_params: the arrays ``rate_fn`` closes over --- ``A``,
            ``n``, ``Ea``, ``K_eq``, and the forward and reverse order
            matrices. A pytree, so it differentiates and jits.
        equations: one LaTeX rate expression per reaction, following the
            ``equations`` convention the unit operations use.
        reactions: the source specification, kept so the set can be
            written back out.
        reverse: the reversibility mode this set was built with.
    """

    species_order: list[str]
    stoich: Array
    rate_fn: Callable
    rate_params: dict
    equations: list[str] = field(default_factory=list)
    reactions: list[dict] = field(default_factory=list)
    reverse: str = "error"

    @property
    def n_species(self) -> int:
        return len(self.species_order)

    @property
    def n_reactions(self) -> int:
        return int(self.stoich.shape[1])

    def params_kwargs(self) -> dict:
        """The reactor ``Params`` fields this set supplies.

        Splat into any reactor that takes a rate law::

            CSTRParams(V=1.0, **kin.params_kwargs())
            PFRParams(V=1.0, n_steps=50, **kin.params_kwargs())
        """
        return {
            "rate_fn": self.rate_fn,
            "stoich": self.stoich,
            "rate_params": self.rate_params,
            "species_order": list(self.species_order),
        }

    def rates(self, concentrations: dict[str, Any], T: float) -> Array:
        """Evaluate the rate law directly, for inspection or testing."""
        return self.rate_fn(concentrations, T, self.rate_params)

    def summary(self) -> str:
        """Readable table of the reactions and their coefficients."""
        p = self.rate_params
        lines = [
            f"{self.n_reactions} reaction(s) over {self.n_species} species "
            f"({', '.join(self.species_order)}); reverse mode: {self.reverse}",
            "",
            f"{'equation':<34} {'A':>11} {'n':>6} {'Ea (kJ/mol)':>12} {'K_eq':>10}",
            "-" * 78,
        ]
        for j, rxn in enumerate(self.reactions):
            k_eq = float(p["K_eq"][j])
            k_eq_s = "-" if not jnp.isfinite(k_eq) else f"{k_eq:10.4g}"
            lines.append(
                f"{rxn.get('equation', f'reaction {j}'):<34} "
                f"{float(p['A'][j]):11.4g} {float(p['n'][j]):6.2f} "
                f"{float(p['Ea'][j]) / 1000.0:12.2f} {k_eq_s:>10}"
            )
        return "\n".join(lines)


def _species_in(reactions: Sequence[dict]) -> list[str]:
    """Sorted union of every species appearing in the reactions."""
    seen: set[str] = set()
    for rxn in reactions:
        seen.update(rxn.get("reactants", {}))
        seen.update(rxn.get("products", {}))
    return sorted(seen)


def _latex(rxn: dict, reverse: bool) -> str:
    """LaTeX rate expression for one reaction."""
    def side(d):
        return " ".join(
            (f"C_{{{s}}}" if abs(c - 1.0) < 1e-12 else f"C_{{{s}}}^{{{c:g}}}")
            for s, c in d.items()
        ) or "1"

    forward = side(rxn.get("reactants", {}))
    if not reverse:
        return f"r = k(T) \\, {forward}"
    return (
        f"r = k(T) \\left( {forward} - "
        f"\\frac{{1}}{{K_{{eq}}}} {side(rxn.get('products', {}))} \\right)"
    )


def mass_action_kinetics(
    reactions: Sequence[dict],
    species_order: Sequence[str] | None = None,
    *,
    reverse: str = "error",
    orders: Sequence[dict] | None = None,
) -> ReactionSet:
    """Build a rate law and stoichiometry from reaction dictionaries.

    Args:
        reactions: one dict per reaction, with keys ``reactants`` and
            ``products`` (``{species: coefficient}``) and
            ``rate_params`` (``{"A": ..., "Ea": ..., "n": ...}``, where
            ``Ea`` and ``n`` default to 0). Optional keys: ``equation``
            (for reporting), ``reversible``, ``type``, and ``K_eq``.
            This is exactly what :func:`difflow.import_reactions`
            returns, and equally what a form or JSON file can produce.
        species_order: species names fixing the rows of ``stoich``.
            Defaults to the sorted union of every species mentioned;
            pass it explicitly to match a stream's species order.
        reverse: how to treat reactions marked reversible.
            ``"error"`` (default) refuses them, because the reverse
            rate is not determined by forward Arrhenius parameters
            alone. ``"forward_only"`` drops the reverse term --- a real
            approximation, recorded on the result. ``"equilibrium"``
            uses each reaction's ``K_eq``, which must then be present.
        orders: optional per-reaction ``{species: order}`` overriding
            the reaction orders, which otherwise come from the reactant
            stoichiometry. Use for empirical rate laws where the order
            is not the coefficient. ``None`` for a reaction keeps its
            stoichiometric orders; an empty dict makes it zeroth order
            in everything.

    Returns:
        A :class:`ReactionSet`.

    Raises:
        KineticsSpecError: for an unsupported reaction type, a species
            outside ``species_order``, a reversible reaction under
            ``reverse="error"``, or a missing ``K_eq`` under
            ``reverse="equilibrium"``.

    Example:
        >>> reactions = [{
        ...     "equation": "A -> B",
        ...     "reactants": {"A": 1.0}, "products": {"B": 1.0},
        ...     "rate_params": {"A": 1.0e6, "Ea": 50_000.0, "n": 0.0},
        ... }]
        >>> kin = mass_action_kinetics(reactions, ["A", "B"])
        >>> params = CSTRParams(V=1.0, **kin.params_kwargs())
    """
    if reverse not in REVERSE_MODES:
        raise ValueError(
            f"reverse={reverse!r} is not one of {REVERSE_MODES}"
        )
    reactions = list(reactions)
    if not reactions:
        raise KineticsSpecError("no reactions given")

    names = (
        list(species_order) if species_order is not None
        else _species_in(reactions)
    )
    index = {s: i for i, s in enumerate(names)}
    n_s, n_r = len(names), len(reactions)

    # ---- validate before building anything -------------------------
    bad_type = [
        (j, rxn.get("type"))
        for j, rxn in enumerate(reactions)
        if str(rxn.get("type", "elementary")).lower() not in SUPPORTED_TYPES
    ]
    if bad_type:
        detail = ", ".join(f"reaction {j} ({t!r})" for j, t in bad_type)
        raise KineticsSpecError(
            f"unsupported reaction type(s): {detail}. Mass action covers "
            "elementary reactions only; three-body, falloff and other "
            "pressure-dependent forms need their own rate law and are "
            "refused rather than approximated."
        )

    # ``equation`` is a label, not a source of stoichiometry, so a
    # reaction giving neither side would build an all-zero column: a
    # rate law that runs, consumes nothing and produces nothing. That is
    # the silently-different model this module exists to prevent.
    empty = [
        j for j, rxn in enumerate(reactions)
        if not rxn.get("reactants") and not rxn.get("products")
    ]
    if empty:
        listed = ", ".join(
            f"{j} ({reactions[j].get('equation', '?')!r})" for j in empty
        )
        raise KineticsSpecError(
            f"reaction(s) {listed} name no reactants and no products, so "
            "they would react nothing. Stoichiometry comes from the "
            "'reactants' and 'products' keys; 'equation' is only a label "
            "and is not parsed."
        )

    unknown = sorted({
        s
        for rxn in reactions
        for s in list(rxn.get("reactants", {})) + list(rxn.get("products", {}))
        if s not in index
    })
    if unknown:
        raise KineticsSpecError(
            f"species not in species_order: {', '.join(unknown)}. "
            f"species_order is {names}."
        )

    reversible = [
        j for j, rxn in enumerate(reactions) if rxn.get("reversible", False)
    ]
    if reversible and reverse == "error":
        listed = ", ".join(
            f"{j} ({reactions[j].get('equation', '?')})" for j in reversible
        )
        raise KineticsSpecError(
            f"reaction(s) {listed} are marked reversible, but forward "
            "Arrhenius parameters alone do not determine the reverse "
            "rate. Pass reverse='equilibrium' and give each one a "
            "'K_eq', or reverse='forward_only' to drop the reverse term "
            "as an explicit approximation."
        )
    if reverse == "equilibrium":
        missing = [j for j in reversible if reactions[j].get("K_eq") is None]
        if missing:
            raise KineticsSpecError(
                f"reverse='equilibrium' needs a 'K_eq' on every reversible "
                f"reaction; missing on reaction(s) {missing}."
            )

    # ---- build the arrays ------------------------------------------
    stoich = jnp.zeros((n_s, n_r), dtype=jnp.float64)
    order_f = jnp.zeros((n_r, n_s), dtype=jnp.float64)
    order_r = jnp.zeros((n_r, n_s), dtype=jnp.float64)
    a_vals, n_vals, ea_vals, keq_vals = [], [], [], []

    for j, rxn in enumerate(reactions):
        reactants = rxn.get("reactants", {})
        products = rxn.get("products", {})
        for s, c in reactants.items():
            stoich = stoich.at[index[s], j].add(-float(c))
            order_f = order_f.at[j, index[s]].add(float(c))
        for s, c in products.items():
            stoich = stoich.at[index[s], j].add(float(c))
            order_r = order_r.at[j, index[s]].add(float(c))

        # None means "no override, use the stoichiometry"; an empty dict
        # is an override to zeroth order in everything, so test against
        # None rather than truthiness
        if orders is not None and orders[j] is not None:
            # an explicit empirical order replaces the whole forward row
            order_f = order_f.at[j, :].set(0.0)
            for s, o in orders[j].items():
                if s not in index:
                    raise KineticsSpecError(
                        f"order given for {s!r}, which is not in species_order"
                    )
                order_f = order_f.at[j, index[s]].set(float(o))

        # Arrhenius coefficients may be traced: the reaction *structure*
        # is static, but the numbers can come from an optimizer, so
        # building a set inside jit/grad works.
        rp = rxn.get("rate_params", {}) or {}
        a_vals.append(jnp.asarray(rp.get("A", 0.0), dtype=jnp.float64))
        n_vals.append(jnp.asarray(rp.get("n", 0.0), dtype=jnp.float64))
        ea_vals.append(jnp.asarray(rp.get("Ea", 0.0), dtype=jnp.float64))

        is_rev = j in reversible and reverse == "equilibrium"
        keq_vals.append(
            jnp.asarray(rxn["K_eq"], dtype=jnp.float64) if is_rev
            else jnp.asarray(jnp.inf, dtype=jnp.float64)
        )

    rate_params = {
        "A": jnp.asarray(a_vals, dtype=jnp.float64),
        "n": jnp.asarray(n_vals, dtype=jnp.float64),
        "Ea": jnp.asarray(ea_vals, dtype=jnp.float64),
        # inf disables the reverse term without a branch: 1/inf = 0
        "K_eq": jnp.asarray(keq_vals, dtype=jnp.float64),
        "order_f": order_f,
        "order_r": order_r,
    }

    def rate_fn(C, T, p):
        """Mass-action rates, mol/m^3/s. Traceable and differentiable."""
        c = jnp.stack([jnp.asarray(C[s], dtype=jnp.float64) for s in names])
        # Products in log space: exp(order . log c) stays differentiable
        # at c = 0 and gives 1 for a zero order, where c ** 0 would be
        # the indeterminate 0 ** 0. See CONC_FLOOR for the trade-off.
        log_c = safe_log(jnp.maximum(c, 0.0), CONC_FLOOR)
        k = p["A"] * jnp.power(T, p["n"]) * jnp.exp(-p["Ea"] / (R_GAS * T))
        forward = jnp.exp(p["order_f"] @ log_c)
        backward = jnp.exp(p["order_r"] @ log_c) / p["K_eq"]
        return k * (forward - backward)

    # Record how this rate law was built, so it can be written to a
    # file. The callable itself is a closure and cannot be serialized;
    # the specification that produced it can, and rebuilding from it
    # gives back the identical function. See difflow.serialize.
    rate_fn.__difflow_spec__ = {
        "factory": "mass_action_kinetics",
        "attr": "rate_fn",
        "kwargs": {
            "reactions": reactions,
            "species_order": list(names),
            "reverse": reverse,
            "orders": list(orders) if orders is not None else None,
        },
    }

    equations = [
        _latex(rxn, reverse == "equilibrium" and j in reversible)
        for j, rxn in enumerate(reactions)
    ]

    return ReactionSet(
        species_order=names,
        stoich=stoich,
        rate_fn=rate_fn,
        rate_params=rate_params,
        equations=equations,
        reactions=reactions,
        reverse=reverse,
    )
