"""Mass-action equilibrium closure for REE solvent extraction (#196).

What this replaces
------------------
The correlation layer evaluates ``log10(D)`` at a *specified* pH and applies
multiplicative corrections for loading. Three limitations follow from having
no closure, and none of them can be fixed inside a correlation (#196):

1. **pH is a parameter, not a state.** Every extracted trivalent ion releases
   three protons, so a real cascade's pH profile is set by the extraction
   itself. Here the pH profile is an *output* of the proton balance.
2. **Competitive loading is a correction rather than an outcome.** The
   elements share one finite extractant inventory. Here that is a single free
   extractant balance, not a ``(1 - theta)^3`` factor (#189, #190, #191).
3. **Extractant selection is not physically grounded.** A fitted ``D`` cannot
   respond to loading or medium; a mass-action constant does.

The model
---------
The reaction network is data (:mod:`difflow_ree.equilibrium.network`): a
component basis, a set of formed species with integer stoichiometry, one
``log10 K`` per species and a phase per species. The unknowns per stage are
the natural logs of the free component concentrations -- free ``[H+]``, free
extractant, free anion, the counter-ion and the aqueous concentration of each
rare earth. The equations are the component balances; the mass-action
expressions are substituted rather than posed as extra rows, so the system is
square and mass action holds identically at *every* Newton iterate, converged
or not.

Aqueous charge balance is then a *consequence* of the component balances
whenever the entering totals are electroneutral, not an independent equation.
It is always reported as :func:`charge_imbalance` so an inconsistent feed is
visible, and it can be used *in place of* the anion balance with
``anion_closure="charge"`` when the anion total is not independently known.
The default is ``"total"``, because that is the closure under which the anion
is conserved exactly.

Four design decisions, and why
------------------------------
**Solve at section scope, not per stage.** The whole counter-current section
is one residual ``r(z; theta, u) = 0`` handed to
:func:`difflow.eo_solver.solve_residual_system`, which is an
``optimistix.root_find`` and therefore gives implicit differentiation for
free. Against a sequential sweep of per-stage Newton solves this wins four
ways: the reverse-mode tape is constant size rather than proportional to
stages times iterations; the section Jacobian falls out and is the object the
linearization, back-off and estimation layers want; the geometric-decay
conditioning of a long cascade becomes a residual-scaling question, handled
once in :func:`section_scales`; and a recycle tear stops being a separate
mechanism -- it is another row of ``r``.

**Solve in log concentration.** Positivity is automatic (no clipping, so no
dead gradient), the ten-plus orders of magnitude a real cascade spans stay
conditioned, and the mass-action substitution
``ln[S_j] = ln K_j + sum_c nu_jc u_c`` is *linear* in the unknowns, which is
why Newton behaves at all.

**Initialize by continuation from the correlation.** Mass-action systems are
exponentially nonlinear and lose Newton from a poor start. The existing L1
correlation is solved first and its Kremser profile becomes ``u0``
(:func:`correlation_initial_guess`), which is what gives the correlation path a
continuing purpose. That start is then walked into Newton's basin by
:func:`_globalize` -- damped Newton, a trust region, damped Newton -- and
``n_continuation_steps > 1`` will additionally ramp the rare-earth feed from
dilute to full strength if a section needs it. All of that runs under
``stop_gradient``; it moves the starting point and never the answer.

**Return soft failures.** One cannot raise from inside ``vmap`` or ``scan``,
so :func:`solve_section` returns a :class:`MassActionSolution` carrying the
solution *together with* a residual norm and a boolean feasibility flag. No
Python branch is ever taken on a traced value.

Tolerances
----------
``inner_tol`` defaults to ``1e-12`` on the *scaled* (dimensionless) component
balances, and feasibility is declared at ``1e-8``. Outer flowsheet tolerances
in difflow are 1e-6 to 1e-8, so the inner solve is four to six orders tighter.
Keep it that way: a loosely converged inner solve gives an implicit-function
gradient that is exact for the solution manifold but inconsistent with the
number the code actually returned, and the finite-difference disagreement that
results is very hard to diagnose afterwards.

Conservation
------------
Component conservation is *structural*, not asymptotic. Stages are indexed
from 0, where the aqueous feed enters and the loaded organic leaves, to
``N - 1``, where the fresh solvent enters and the raffinate leaves. The organic
outlet is read from the converged stage-0 organic phase and the aqueous outlet
is then formed as ``(everything in) - (organic out)``, componentwise on the
tableau. Every component therefore balances to floating-point round-off
regardless of how well the equilibrium converged, and how well it converged is
reported separately as ``residual_norm``. ``info["equilibrium_closure"]`` gives
the (tolerance-sized) difference between that aqueous outlet and the one the
converged stage-``N-1`` aqueous phase predicts, so the choice is visible rather
than hidden.

What is deliberately not modelled
---------------------------------
Water dissociation and rare-earth hydrolysis (no ``OH-`` species), aqueous
complexation with the anion, non-idealities in either phase (the constants are
conditional constants at the medium's ionic strength, the same convention the
correlations use, see #194), third-phase formation, and any temperature
dependence of ``log10 K`` beyond what the calibration point carries. Each of
those is a row in ``reaction_networks.yaml`` away, which is the point of
carrying the network as data. For a treatment that does pose both phases with
activity models, see Iloeje et al., *Environ. Sci. Technol.* 53 (2019) 8926,
doi:10.1021/acs.est.9b01718, which formulates rare-earth extraction as Gibbs
energy minimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array, lax

from difflow.eo_solver import solve_residual_system
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, make_stream
from difflow_ree.database import get_extractant
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.network import (
    ReactionNetwork,
    build_network,
    get_network_template,
    log_K_from_correlation,
    network_for_extractant,
)
from difflow_ree.equilibrium.schema import REEStreamSchema


#: Relative floor applied to a component total that must be positive, taken
#: against the largest total in the section. A component with no throughput at
#: all (an unused counter-ion, an element absent from the feed) would otherwise
#: give ``ln(0)`` and a singular Newton row. 1e-24 is far below anything
#: physical and, being applied in log space, costs nothing in conditioning.
_TOTAL_FLOOR_REL = 1e-24

#: Roles whose component total is required to be strictly positive, and which
#: are therefore floored. The proton is *not* in this list: a loaded organic
#: phase carries a negative H component by construction (see
#: :mod:`difflow_ree.equilibrium.network`), so flooring it would be wrong.
_POSITIVE_ROLES = ("rare_earth", "counter_ion", "anion", "extractant")

#: Anion closures. ``"total"`` uses the anion component balance, under which
#: the anion is conserved exactly. ``"charge"`` replaces that row with aqueous
#: electroneutrality, which is the right choice when the anion total is not
#: independently known -- at the cost that the anion is then no longer a
#: conserved quantity of the model.
ANION_CLOSURES = ("total", "charge")


# =============================================================================
# Soft-failure solution container
# =============================================================================

@dataclass(frozen=True, eq=False)
class MassActionSolution:
    """Converged (or not) section state, with its own failure report.

    Registered as a JAX pytree, so it survives ``jit``, ``vmap`` and ``grad``
    and can be returned from inside them. That is the whole point: one cannot
    raise from inside ``vmap``, so failure is a *value* here (#196).

    Attributes:
        u: ``(n_stages, n_components)`` natural log of the free component
            concentrations. Stage 0 is where the aqueous feed enters and the
            loaded organic leaves.
        residual_norm: Infinity norm of the scaled component balances. Scaled
            means dimensionless: each balance is divided by that component's
            own throughput.
        feasible: Boolean scalar. True when ``u`` is finite and
            ``residual_norm`` is below the feasibility tolerance. Consume it
            with ``jnp.where``, never with ``if``.
        ln_K: ``(n_species,)`` natural-log formation constants the solution
            was computed with. Carried on the solution rather than read back
            off the network, because a caller may have overridden them (that
            is how you differentiate with respect to equilibrium constants);
            reading the network's own values instead would silently return
            species concentrations that do not satisfy the solved balances.

    Example:
        >>> sol.feasible  # doctest: +SKIP
        Array(True, dtype=bool)
    """
    u: Array
    residual_norm: Array
    feasible: Array
    ln_K: Array

    def pH(self, network: ReactionNetwork) -> Array:
        """Per-stage pH profile, on the concentration scale.

        This is an **output** of the model. It responds to the three protons
        released per trivalent ion because the proton balance is one of the
        equations solved, not a parameter supplied.

        Args:
            network: The network the solution was computed on.

        Returns:
            ``(n_stages,)`` array of ``-log10([H+])``.
        """
        return -self.u[:, network.proton_index] / float(np.log(10.0))

    def concentrations(self, network: ReactionNetwork) -> dict[str, Array]:
        """Free component and formed species concentrations, per stage.

        Args:
            network: The network the solution was computed on.

        Returns:
            Mapping of component/species name to a ``(n_stages,)`` array of
            molar concentrations (M) in that species' own phase.
        """
        c_comp = jnp.exp(self.u)
        ln_s = self.ln_K[None, :] + self.u @ jnp.asarray(network.nu).T
        c_spec = jnp.exp(ln_s)
        out = {n: c_comp[:, i] for i, n in enumerate(network.component_names)}
        out.update(
            {n: c_spec[:, j] for j, n in enumerate(network.species_names)}
        )
        return out


jax.tree_util.register_pytree_node(
    MassActionSolution,
    lambda s: ((s.u, s.residual_norm, s.feasible, s.ln_K), None),
    lambda _, children: MassActionSolution(*children),
)


# =============================================================================
# Component totals from streams
# =============================================================================

def aqueous_component_totals(
    network: ReactionNetwork,
    schema: REEStreamSchema,
    stream: Stream,
    base_addition: Array | float = 0.0,
) -> Array:
    """Component totals (mol/s) carried by an aqueous stream.

    The shipped networks declare no aqueous species beyond the free
    components, so each aqueous key maps straight onto its component column.

    Args:
        network: Expanded reaction network.
        schema: Stream schema naming the acid, counter-ion and anion keys.
        stream: Aqueous stream.
        base_addition: Molar flow (mol/s) of a strong monoacidic base MOH
            dosed into this stream. It moves one mole from the proton total
            to the counter-ion total per mole added, which leaves
            electroneutrality untouched. This is the closed model's *input*
            degree of freedom in place of the correlation's specified pH; see
            :func:`base_addition_for_pH`.

    Returns:
        ``(n_components,)`` array of component molar flows.

    Raises:
        ValueError: If base is being added to a network that declares no
            counter-ion component, so the cation it brings in has nowhere to
            be conserved. Only a *concrete* non-zero dosing can be caught; a
            tracer is not inspectable.
    """
    flows = get_flows(stream)
    totals = [jnp.asarray(0.0, dtype=jnp.float64)] * network.n_components

    for i, el in enumerate(network.elements):
        totals[network.element_component_index[i]] = jnp.asarray(
            flows.get(el, 0.0), dtype=jnp.float64
        )

    b = jnp.asarray(base_addition, dtype=jnp.float64)
    acid = jnp.asarray(flows.get(schema.acid, 0.0), dtype=jnp.float64)
    # Co-extracted acid declared on an aqueous stream is unusual but is
    # counted rather than dropped, so nothing disappears silently.
    acid = acid + jnp.asarray(flows.get(schema.organic_acid, 0.0),
                              dtype=jnp.float64)
    totals[network.proton_index] = acid - b

    if network.counter_ion_index is not None:
        m_key = schema.counter_ion
        m = jnp.asarray(
            flows.get(m_key, 0.0) if m_key else 0.0, dtype=jnp.float64
        )
        totals[network.counter_ion_index] = m + b
    else:
        _reject_base_without_counter_ion(b)

    totals[network.anion_index] = jnp.asarray(
        flows.get(schema.anion, 0.0), dtype=jnp.float64
    )
    return jnp.stack(totals)


def _reject_base_without_counter_ion(base_addition: Array) -> None:
    """Refuse base addition when the network has no counter-ion component.

    Args:
        base_addition: The dosed base flow; only a *concrete* non-zero value
            is rejected, because a tracer cannot be inspected.

    Raises:
        ValueError: If a concrete non-zero base addition was requested.
    """
    try:
        value = float(base_addition)
    except Exception:  # TracerArrayConversionError / ConcretizationTypeError
        return
    if value != 0.0:
        raise ValueError(
            "base_addition is non-zero but the reaction network declares no "
            "counter-ion component, so the cation the base brings in has "
            "nowhere to be conserved. Use a network with a 'counter_ion' "
            "component (every shipped network has one) (#196)."
        )


def organic_component_totals(
    network: ReactionNetwork,
    schema: REEStreamSchema,
    stream: Stream,
    extra_species_flows: Mapping[str, Array | float] | None = None,
) -> Array:
    """Component totals (mol/s) carried by an organic stream.

    The organic stream declares its **total** extractant on a monomer basis
    and its loaded rare earths by element symbol. The loaded complexes are
    read through the tableau, so a recycled loaded solvent correctly brings a
    *negative* proton component into the section -- three per trivalent ion,
    the protons it took up during stripping.

    Args:
        network: Expanded reaction network.
        schema: Stream schema naming the extractant and diluent keys.
        stream: Organic stream.
        extra_species_flows: Molar flows of non-per-element organic species,
            keyed by species name. Needed only for networks with such species
            (saponified extractant, #197); the shipped cation-exchange network
            has none.

    Returns:
        ``(n_components,)`` array of component molar flows.
    """
    flows = get_flows(stream)
    nu = jnp.asarray(network.nu, dtype=jnp.float64)

    amounts = jnp.zeros(network.n_species, dtype=jnp.float64)
    for i, el in enumerate(network.elements):
        j = network.element_species_index[i]
        amounts = amounts.at[j].set(
            jnp.asarray(flows.get(el, 0.0), dtype=jnp.float64)
        )
    if extra_species_flows:
        names = network.species_names
        for name, value in extra_species_flows.items():
            amounts = amounts.at[names.index(name)].set(
                jnp.asarray(value, dtype=jnp.float64)
            )

    totals = amounts @ nu

    # The extractant column is the one total the stream states directly: the
    # complex contributions computed above are already inside it.
    f_ext = jnp.asarray(flows.get(schema.extractant, 0.0), dtype=jnp.float64)
    totals = totals.at[network.extractant_index].set(
        f_ext / network.monomers_per_component
    )
    # Co-extracted acid has no organic species in the shipped networks, so it
    # is counted into the proton total (where it behaves as free acid) rather
    # than dropped.
    totals = totals.at[network.proton_index].add(
        jnp.asarray(flows.get(schema.organic_acid, 0.0), dtype=jnp.float64)
    )
    return totals


def section_scales(
    network: ReactionNetwork,
    feed_totals: Array,
    solvent_totals: Array,
) -> Array:
    """Per-component residual scales.

    Every component balance is divided by that component's own throughput, so
    the residual vector is dimensionless and the Newton system's rows are all
    O(1) regardless of whether the component is 5 mol/s of chloride or 1e-8
    mol/s of a trace lanthanum. This is the "geometric-decay conditioning
    becomes a residual-scaling question" of #196, and it is done once here
    rather than tuned per cascade.

    The magnitudes are added, not the signed totals: a loaded solvent's
    proton component is negative and would otherwise cancel against the feed
    acid and leave a meaningless scale.

    Cancellation is not only a solvent effect, though. Dosing base exactly
    neutralizes the feed acid at one particular rate, and there the proton
    total is *identically zero* while the proton row is anything but
    unimportant -- the section still exchanges three protons per trivalent
    ion. Scaling that row by its own (zero) total makes the relative tolerance
    unreachable. So the scale also counts the protons, extractant and anion
    that the rare earths present could put into play,
    ``sum_i |nu_ic| * (RE_i total)``, which is the honest magnitude of what
    each balance has to resolve.

    Args:
        network: Expanded reaction network.
        feed_totals: Aqueous feed component totals (mol/s).
        solvent_totals: Organic solvent component totals (mol/s).

    Returns:
        ``(n_components,)`` array of strictly positive scales.
    """
    cols = np.asarray(network.element_component_index)
    rows = np.asarray(network.element_species_index)
    re_totals = feed_totals[jnp.asarray(cols)] + solvent_totals[jnp.asarray(cols)]
    exchangeable = jnp.abs(jnp.asarray(network.nu[rows])).T @ jnp.abs(re_totals)
    raw = jnp.abs(feed_totals) + jnp.abs(solvent_totals) + exchangeable
    floor = _TOTAL_FLOOR_REL * jnp.max(raw)
    return jnp.maximum(raw, floor)


def floor_totals(
    network: ReactionNetwork,
    totals: Array,
    reference: Array,
) -> Array:
    """Floor the totals of components that must be strictly positive.

    Args:
        network: Expanded reaction network.
        totals: Component totals to floor (mol/s).
        reference: Scale reference, as returned by :func:`section_scales`.

    Returns:
        The totals with :data:`_POSITIVE_ROLES` components floored at
        ``_TOTAL_FLOOR_REL`` times the largest scale.
    """
    floor = _TOTAL_FLOOR_REL * jnp.max(reference)
    mask = np.array(
        [c.role in _POSITIVE_ROLES for c in network.components], dtype=bool
    )
    return jnp.where(jnp.asarray(mask), jnp.maximum(totals, floor), totals)


# =============================================================================
# The section residual
# =============================================================================

def make_section_residual(
    network: ReactionNetwork,
    n_stages: int,
    anion_closure: str = "total",
) -> tuple[Callable[[Array, dict], Array], Callable[[Array, dict], tuple[Array, Array]]]:
    """Build the counter-current section residual ``r(z; args) = 0``.

    Stage 0 is where the aqueous feed enters and the loaded organic leaves;
    stage ``N-1`` is where the fresh solvent enters and the raffinate leaves.
    Aqueous flow runs ``0 -> N-1``, organic flow runs ``N-1 -> 0``.

    For component ``c`` at stage ``k`` the balance is

    .. math::

        A_{c,k} + O_{c,k} - \\bigl(A_{c,k-1} + O_{c,k+1}\\bigr) = 0

    with ``A`` and ``O`` the component molar flows leaving a stage in the
    aqueous and organic phase, both evaluated from the *same* unknowns through
    the tableau. That is ``n_stages * n_components`` equations in exactly as
    many unknowns.

    Args:
        network: Expanded reaction network. Static: it is closed over, never
            traced.
        n_stages: Number of equilibrium stages. Static (it sets the shape).
        anion_closure: ``"total"`` (default) or ``"charge"``; see
            :data:`ANION_CLOSURES`.

    Returns:
        ``(residual_fn, phase_flows_fn)``.

        ``residual_fn(u_flat, args) -> (n_stages*n_components,)`` is what goes
        to the solver. ``phase_flows_fn(u, args) -> (aq, org)`` returns the
        per-stage component molar flows in each phase and is what the outlet
        streams are built from.

        ``args`` is a dict with keys ``ln_K``, ``Q_aq``, ``Q_org``,
        ``feed_totals``, ``solvent_totals`` and ``scale`` -- all traced, so a
        cascade is differentiable with respect to any of them, equilibrium
        constants included.

    Raises:
        ValueError: If ``n_stages < 1`` or ``anion_closure`` is unknown.
    """
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages} (#196).")
    if anion_closure not in ANION_CLOSURES:
        raise ValueError(
            f"anion_closure must be one of {list(ANION_CLOSURES)}, got "
            f"{anion_closure!r} (#196)."
        )

    n_comp = network.n_components
    nu = jnp.asarray(network.nu, dtype=jnp.float64)
    spec_aq = jnp.asarray(network.species_is_aqueous, dtype=jnp.float64)
    spec_org = 1.0 - spec_aq
    comp_aq = jnp.asarray(network.component_is_aqueous, dtype=jnp.float64)
    comp_org = 1.0 - comp_aq
    z_comp = jnp.asarray(network.component_charges, dtype=jnp.float64)
    z_spec = jnp.asarray(network.species_charges, dtype=jnp.float64)
    anion_index = network.anion_index

    def phase_flows(u: Array, args: dict) -> tuple[Array, Array]:
        """Component molar flows leaving each stage in each phase."""
        c_comp = jnp.exp(u)                                   # (N, C)
        # Mass action, in log space and therefore linear in the unknowns.
        c_spec = jnp.exp(args["ln_K"][None, :] + u @ nu.T)    # (N, S)
        aq = args["Q_aq"] * (c_comp * comp_aq + (c_spec * spec_aq) @ nu)
        org = args["Q_org"] * (c_comp * comp_org + (c_spec * spec_org) @ nu)
        return aq, org

    def aqueous_charge(u: Array, args: dict) -> Array:
        """Net aqueous charge concentration per stage (M)."""
        c_comp = jnp.exp(u)
        c_spec = jnp.exp(args["ln_K"][None, :] + u @ nu.T)
        return (c_comp * comp_aq) @ z_comp + (c_spec * spec_aq) @ z_spec

    def residual(u_flat: Array, args: dict) -> Array:
        u = u_flat.reshape(n_stages, n_comp)
        aq, org = phase_flows(u, args)
        aq_prev = jnp.concatenate(
            [args["feed_totals"][None, :], aq[:-1]], axis=0
        )
        org_next = jnp.concatenate(
            [org[1:], args["solvent_totals"][None, :]], axis=0
        )
        r = (aq + org) - (aq_prev + org_next)
        r = r / args["scale"][None, :]
        if anion_closure == "charge":
            # Electroneutrality in place of the anion balance. Scaled by the
            # anion's own concentration scale so the row stays O(1).
            c_scale = args["scale"][anion_index] / args["Q_aq"]
            r = r.at[:, anion_index].set(aqueous_charge(u, args) / c_scale)
        return r.reshape(-1)

    return residual, phase_flows


def charge_imbalance(
    network: ReactionNetwork,
    solution: MassActionSolution,
) -> Array:
    """Net aqueous charge concentration per stage (M): should be zero.

    With a complete component basis and electroneutral entering totals, the
    aqueous charge balance is implied by the component balances rather than
    independent of them. It is therefore reported as a diagnostic instead of
    being solved redundantly: a non-zero value means the *feed* was not
    electroneutral, which is a statement about the input, not about the
    solver.

    Args:
        network: Expanded reaction network.
        solution: A solved section.

    Returns:
        ``(n_stages,)`` array of ``sum_j z_j [S_j]`` over aqueous species.
    """
    c = solution.concentrations(network)
    total = jnp.zeros(solution.u.shape[0], dtype=jnp.float64)
    for i, comp in enumerate(network.components):
        if comp.phase == "aqueous":
            total = total + comp.charge * c[comp.name]
    for j, spec in enumerate(network.species):
        if spec.phase == "aqueous":
            total = total + spec.charge * c[spec.name]
    return total


# =============================================================================
# Initialization by continuation from the correlation
# =============================================================================

def correlation_initial_guess(
    network: ReactionNetwork,
    n_stages: int,
    feed_totals: Array,
    solvent_totals: Array,
    Q_aq: Array | float,
    Q_org: Array | float,
    D_values: Array,
    scale: Array,
) -> Array:
    """Starting point for the section solve, taken from the L1 correlation.

    Mass-action systems lose Newton from a poor start, so the closed model is
    started from the answer the correlation gives. The aqueous rare-earth
    profile is a Kremser fraction-remaining interpolated geometrically across
    the stages, the free extractant is the total less what that profile loads,
    and the proton, counter-ion and anion start at their feed concentrations.
    It does not have to be accurate -- it has to be the right orders of
    magnitude, which in log space is all Newton needs.

    Args:
        network: Expanded reaction network.
        n_stages: Number of stages.
        feed_totals: Aqueous feed component totals (mol/s), already floored.
        solvent_totals: Organic solvent component totals (mol/s), floored.
        Q_aq: Aqueous volumetric flow (L/s).
        Q_org: Organic volumetric flow (L/s).
        D_values: ``(n_elements,)`` distribution ratios from the correlation.
        scale: Per-component scales from :func:`section_scales`, used as the
            fallback magnitude for components whose signed total is zero or
            negative.

    Returns:
        ``(n_stages, n_components)`` array of natural log concentrations.
    """
    Q_aq = jnp.asarray(Q_aq, dtype=jnp.float64)
    Q_org = jnp.asarray(Q_org, dtype=jnp.float64)
    n_el = len(network.elements)
    cols = jnp.asarray(network.element_component_index)

    # Kremser fraction remaining, then a geometric profile across the section.
    E = jnp.asarray(D_values, dtype=jnp.float64) * Q_org / Q_aq
    E = jnp.clip(E, 1e-12, 1e12)
    near_one = jnp.abs(E - 1.0) < 1e-6
    E_safe = jnp.where(near_one, 2.0, E)
    frac = jnp.where(
        near_one,
        1.0 / (n_stages + 1.0),
        (E_safe - 1.0) / (jnp.power(E_safe, n_stages + 1.0) - 1.0),
    )
    frac = jnp.clip(frac, 1e-14, 1.0)
    steps = (jnp.arange(n_stages, dtype=jnp.float64) + 1.0) / n_stages
    # (n_stages, n_elements)
    profile = jnp.power(frac[None, :], steps[:, None])

    x_feed = feed_totals[cols] / Q_aq                     # (n_elements,)
    x = jnp.maximum(x_feed[None, :] * profile, 1e-300)

    # Free extractant: total less what the profile loads onto it.
    q = jnp.asarray(
        [network.nu[network.element_species_index[i], network.extractant_index]
         for i in range(n_el)],
        dtype=jnp.float64,
    )
    c_ext_total = solvent_totals[network.extractant_index] / Q_org
    loaded = jnp.sum(q[None, :] * jnp.asarray(D_values)[None, :] * x, axis=1)
    # Soft saturation rather than `max(total - loaded, floor)`. The correlation
    # is evaluated at the *calibration* pH, so on a poorly buffered feed it
    # predicts far more loading than the section can hold and a hard floor put
    # the start four orders of magnitude below the answer, which is a long way
    # to travel with a step limiter (#196).
    c_ext_free = c_ext_total / (1.0 + loaded / c_ext_total)

    u = jnp.zeros((n_stages, network.n_components), dtype=jnp.float64)
    u = u.at[:, cols].set(jnp.log(x))
    u = u.at[:, network.extractant_index].set(jnp.log(c_ext_free))

    for idx in (network.proton_index, network.anion_index,
                network.counter_ion_index):
        if idx is None:
            continue
        # The proton total can legitimately be NEGATIVE -- a loaded solvent
        # brings a negative H component, and base dosing subtracts from it --
        # so the total is not usable as a starting concentration on its own.
        # Fall back to a hundredth of the component's throughput, which is a
        # free [H+] a couple of pH units above the fully acidic case and is
        # what the section is heading towards anyway (#196). Taking log(0)
        # here instead put the start at exp(-690), i.e. every species
        # identically zero and a singular Newton row.
        c0 = jnp.maximum(
            jnp.maximum(feed_totals[idx], 1e-2 * scale[idx]) / Q_aq, 1e-300
        )
        u = u.at[:, idx].set(jnp.log(c0))
    return u


#: Tolerance the globalization phase aims for. It is *tighter* than the final
#: solve's, because the globalization is what actually does the converging: the
#: ``optimistix`` Newton that follows exists to supply the implicit derivative
#: at a point that is already a root, and an undamped Newton started anywhere
#: else will happily walk off into ``exp`` overflow.
_GLOBALIZE_TOL = 1e-14

#: Largest step, in log concentration, that the damped phase will take along
#: the Newton direction. ``exp(2) ~ 7.4``: no species moves by much more than
#: an order of magnitude per iteration, which is what keeps ``exp`` finite when
#: the raw Newton step is 1e3 or more (and it routinely is, from the
#: correlation start).
_MAX_LOG_STEP = 2.0

#: Backtracking factors tried along the (already limited) Newton direction,
#: largest first.
_BACKTRACK = (1.0, 0.5, 0.25, 0.1, 0.03, 0.01)

#: Trust-region iterations per damped-Newton iteration in :func:`_globalize`.
_TRUST_REGION_RATIO = 5


def _damped_newton(
    residual_fn: Callable[[Array, dict], Array],
    z: Array,
    args: dict,
    n_steps: int,
) -> Array:
    """Newton with a step-length limit and backtracking on the max residual.

    The direction is the *full, unmodified* Newton direction; only its length
    is limited, to :data:`_MAX_LOG_STEP` in the infinity norm, and then
    backtracked until the largest scaled balance residual actually falls.
    Keeping the direction matters -- clipping each ``d ln c`` elementwise
    instead gives a direction that is not Newton's and, on a counter-current
    section, not a descent direction either, and a monotone line search then
    rejects every step and stalls outright.

    Args:
        residual_fn: The section residual.
        z: Starting point, flat.
        args: Residual arguments.
        n_steps: Iterations (static).

    Returns:
        An improved point, same shape as ``z``. Never worse: an iteration that
        cannot improve the residual returns its input unchanged.
    """
    factors = jnp.asarray(_BACKTRACK, dtype=jnp.float64)

    def merit(zz):
        return jnp.max(jnp.abs(residual_fn(zz, args)))

    def body(_, z):
        r = residual_fn(z, args)
        J = jax.jacfwd(residual_fn)(z, args)
        dz = jnp.linalg.solve(J, -r)
        dz = jnp.nan_to_num(dz, nan=0.0, posinf=0.0, neginf=0.0)
        dz = dz * jnp.minimum(
            1.0, _MAX_LOG_STEP / jnp.maximum(jnp.max(jnp.abs(dz)), 1e-300)
        )
        candidates = z[None, :] + factors[:, None] * dz[None, :]
        merits = jax.vmap(merit)(candidates)
        merits = jnp.where(jnp.isfinite(merits), merits, jnp.inf)
        best = jnp.argmin(merits)
        return jnp.where(merits[best] < jnp.max(jnp.abs(r)), candidates[best], z)

    return lax.fori_loop(0, n_steps, body, z)


def _trust_region(
    residual_fn: Callable[[Array, dict], Array],
    z: Array,
    args: dict,
    n_steps: int,
) -> Array:
    """Levenberg-Marquardt on the least-squares merit.

    Complements :func:`_damped_newton`: where a monotone damped Newton stalls
    because the linearization is a poor model of the exponentials, a trust
    region shrinks until it is not, and mixes in the gradient direction.

    Args:
        residual_fn: The section residual.
        z: Starting point, flat.
        args: Residual arguments.
        n_steps: Maximum trust-region iterations.

    Returns:
        An improved point, same shape as ``z``. May be non-finite; the caller
        filters that.
    """
    sol = optx.least_squares(
        residual_fn,
        optx.LevenbergMarquardt(rtol=_GLOBALIZE_TOL, atol=_GLOBALIZE_TOL),
        z,
        args=args,
        max_steps=n_steps,
        throw=False,
    )
    return sol.value


def _globalize(
    residual_fn: Callable[[Array, dict], Array],
    z: Array,
    args: dict,
    n_steps: int,
) -> Array:
    """Bring the section state into Newton's basin from the correlation start.

    Log-concentration Newton is *locally* excellent -- the mass-action
    substitution is linear in the unknowns, so once the balances are nearly
    satisfied convergence is quadratic. It is *globally* fragile for exactly
    the same reason: the balances are sums of exponentials, an undamped step
    from the correlation start is routinely 1e3 or more in log space, and
    ``exp`` of that is ``inf`` and then ``NaN``.

    Neither standard remedy is sufficient on its own here, and it is worth
    recording which way each one fails:

    - **Damped Newton alone** (:func:`_damped_newton`) is monotone and cannot
      diverge, but it stalls where the linear model of a sum of exponentials
      is poor -- notably on a section dosed close to full neutralization,
      where the proton total passes through zero.
    - **Levenberg-Marquardt alone** (:func:`_trust_region`) handles that, but
      it minimizes a least-squares merit, and on a ten-element cascade it
      converges to a spurious local minimum with an element driven to
      ``ln c = -27`` and the proton balance out by 3%. A least-squares
      minimizer is not a root finder.

    Running one after the other fixes both: Newton first (cheap, and usually
    enough), the trust region to escape a stall, then Newton again to convert
    the trust region's stopping point -- which is a small *step*, not
    necessarily a small residual -- into an actual root at machine precision.
    Each phase decreases its own merit or leaves its input alone, a non-finite
    result is discarded, and each is a no-op if the previous one already
    converged. (The trust region minimizes the sum of squares while the damped
    Newton watches the maximum, so the chain is not strictly monotone in
    either measure; the final Newton phase is what guarantees the *residual*
    is small at the handover.)

    The whole thing runs under ``stop_gradient`` in :func:`solve_section`: it
    only ever moves the starting point. The answer and its derivative come
    from the ``optimistix`` root find that follows (#196).

    Args:
        residual_fn: The section residual.
        z: Starting point, flat.
        args: Residual arguments.
        n_steps: Damped-Newton iterations per phase; the trust region gets
            :data:`_TRUST_REGION_RATIO` times as many, being much cheaper per
            step.

    Returns:
        An improved starting point, same shape as ``z``.
    """
    def keep_finite(new, old):
        return jnp.where(jnp.all(jnp.isfinite(new)), new, old)

    z1 = keep_finite(_damped_newton(residual_fn, z, args, n_steps), z)
    z2 = keep_finite(
        _trust_region(residual_fn, z1, args, _TRUST_REGION_RATIO * n_steps), z1
    )
    return keep_finite(_damped_newton(residual_fn, z2, args, n_steps), z2)


# =============================================================================
# The section solve
# =============================================================================

def solve_section(
    network: ReactionNetwork,
    n_stages: int,
    feed_totals: Array,
    solvent_totals: Array,
    Q_aq: Array | float,
    Q_org: Array | float,
    D_values: Array,
    log10_K: Array | None = None,
    u0: Array | None = None,
    inner_tol: float = 1e-12,
    feasible_tol: float = 1e-8,
    max_steps: int = 200,
    n_globalize_steps: int = 100,
    n_continuation_steps: int = 1,
    anion_closure: str = "total",
) -> MassActionSolution:
    """Solve a whole counter-current section simultaneously.

    Args:
        network: Expanded reaction network.
        n_stages: Number of equilibrium stages (static).
        feed_totals: Aqueous feed component totals (mol/s).
        solvent_totals: Organic solvent component totals (mol/s).
        Q_aq: Aqueous volumetric flow (L/s).
        Q_org: Organic volumetric flow (L/s).
        D_values: ``(n_elements,)`` correlation distribution ratios, used only
            to build the starting point.
        log10_K: Formation constants overriding ``network.log10_K``. Pass this
            (rather than rebuilding the network) to differentiate with respect
            to the constants.
        u0: Explicit starting point, ``(n_stages, n_components)`` log
            concentrations. None uses :func:`correlation_initial_guess`.
        inner_tol: Newton tolerance on the scaled residual. Four to six orders
            below any outer flowsheet tolerance, deliberately; see the module
            docstring.
        feasible_tol: Residual norm below which the solve is called feasible.
        max_steps: Maximum Newton steps in the final (differentiated) solve.
        n_globalize_steps: Globalization iterations per phase, run under
            ``stop_gradient`` before the final solve to make the start good
            enough for undamped Newton; see :func:`_globalize`. Set to 0 only
            if you are supplying an already converged ``u0`` yourself.
        n_continuation_steps: Number of continuation steps ramping the
            rare-earth feed totals from ``1/n`` of full strength to full
            strength. 1 (the default) solves directly from the correlation
            start, which the trust-region globalization handles across the
            whole range tested. Raise it if you meet a section it does not:
            the intermediate solves are wrapped in ``stop_gradient``, so they
            shape the path and never the answer or its derivative.
        anion_closure: ``"total"`` or ``"charge"``; see
            :data:`ANION_CLOSURES`.

    Returns:
        A :class:`MassActionSolution`. Nothing is raised on failure -- check
        ``feasible``.

    Example:
        >>> sol = solve_section(net, 4, Tf, Ts, 1.0, 1.0, D)  # doctest: +SKIP
        >>> bool(sol.feasible)                                # doctest: +SKIP
        True
    """
    residual_fn, _ = make_section_residual(network, n_stages, anion_closure)

    scale = section_scales(network, feed_totals, solvent_totals)
    feed_totals = floor_totals(network, feed_totals, scale)
    solvent_totals = floor_totals(network, solvent_totals, scale)

    ln_K = (
        network.ln_K() if log10_K is None
        else jnp.asarray(log10_K, dtype=jnp.float64) * float(np.log(10.0))
    )
    args = {
        "ln_K": ln_K,
        "Q_aq": jnp.asarray(Q_aq, dtype=jnp.float64),
        "Q_org": jnp.asarray(Q_org, dtype=jnp.float64),
        "feed_totals": feed_totals,
        "solvent_totals": solvent_totals,
        "scale": scale,
    }

    if u0 is None:
        u0 = correlation_initial_guess(
            network, n_stages, feed_totals, solvent_totals, Q_aq, Q_org,
            D_values, scale,
        )
    z = lax.stop_gradient(jnp.asarray(u0, dtype=jnp.float64).reshape(-1))

    # Continuation on the rare-earth feed strength. Every intermediate solve
    # is under stop_gradient, so it can only move the starting point, never
    # the converged answer or its derivative.
    if n_continuation_steps > 1:
        cols = jnp.asarray(network.element_component_index)
        for step in range(1, n_continuation_steps):
            lam = step / n_continuation_steps
            ramped = dict(args)
            ramped["feed_totals"] = feed_totals.at[cols].multiply(lam)
            z_try = _globalize(residual_fn, z, ramped, n_globalize_steps)
            # A failed intermediate step must not poison the start.
            ok = jnp.all(jnp.isfinite(z_try))
            z = lax.stop_gradient(jnp.where(ok, z_try, z))

    if n_globalize_steps > 0:
        z_try = _globalize(residual_fn, z, args, n_globalize_steps)
        z = lax.stop_gradient(
            jnp.where(jnp.all(jnp.isfinite(z_try)), z_try, z)
        )

    z, residual_norm, feasible = solve_residual_system(
        residual_fn, z, args, rtol=inner_tol, atol=inner_tol,
        max_steps=max_steps, feasible_tol=feasible_tol,
    )
    return MassActionSolution(
        u=z.reshape(n_stages, network.n_components),
        residual_norm=residual_norm,
        feasible=feasible,
        ln_K=ln_K,
    )


def solve_stage(
    network: ReactionNetwork,
    feed_totals: Array,
    solvent_totals: Array,
    Q_aq: Array | float,
    Q_org: Array | float,
    D_values: Array,
    **kwargs: Any,
) -> MassActionSolution:
    """Solve a single equilibrium stage.

    A stage is a section of one, so this is :func:`solve_section` with
    ``n_stages=1``. It exists because the stage operator is the natural unit
    for a mixer-settler and for tests.

    Args:
        network: Expanded reaction network.
        feed_totals: Aqueous inlet component totals (mol/s).
        solvent_totals: Organic inlet component totals (mol/s).
        Q_aq: Aqueous volumetric flow (L/s).
        Q_org: Organic volumetric flow (L/s).
        D_values: Correlation distribution ratios for the starting point.
        **kwargs: Forwarded to :func:`solve_section`.

    Returns:
        A :class:`MassActionSolution` with ``u`` of shape
        ``(1, n_components)``.
    """
    return solve_section(
        network, 1, feed_totals, solvent_totals, Q_aq, Q_org, D_values,
        **kwargs,
    )


# =============================================================================
# Parameters and the section unit operation
# =============================================================================

@dataclass(repr=False)
class MassActionParams(ParamsMixin):
    """Parameters for a mass-action equilibrium extraction section (#196).

    Attributes:
        n_stages: Number of equilibrium stages. Static (it sets the problem
            size), so unlike the correlation path's Kremser ``n_stages`` this
            one cannot be differentiated with respect to.
        extractant: Extractant name, e.g. ``"D2EHPA"``.
        elements: REE symbols to track.
        diluent: Organic diluent species key.
        counter_ion: Aqueous counter-ion key, or None.
        anion: Aqueous anion key.
        extractant_conc: Total extractant concentration (M, monomer basis).
            Used for the calibration and for the starting point.
        aqueous_volumetric_flow: Aqueous volumetric flow (L/s). Concentrations
            are molar, so the closed model needs volumes where the
            correlation needed only a flow ratio. There is no defensible way
            to guess it, so it is required.
        organic_volumetric_flow: Organic volumetric flow (L/s).
        calibration_pH: pH at which ``log10 K`` is calibrated from the L1
            correlation. **This is not an operating specification.** In the
            closed model the pH is an output; this number only says where the
            two levels are made to agree. Set it near the pH the section
            actually runs at.
        log10_K: Measured formation constants keyed by element symbol.
            Supplying these bypasses the calibration, and is what a user with
            real data should do.
        network: Reaction network template name, a key of
            ``data/reaction_networks.yaml``. None picks the template matching
            the extractant record's mechanism and basis.
        anion_conc: Reference free anion concentration (M) used when
            calibrating ``log10 K`` from the correlation. It only matters for
            a network whose complex contains the anion -- solvating and anion
            exchange -- where the anion is the driving variable; for cation
            exchange its coefficient is zero and it cancels exactly. None
            (default) uses the extractant record's ``reference_nitrate`` when
            it has one, else 1.0 M.
        inner_tol: Newton tolerance on the scaled residual (default 1e-12).
        feasible_tol: Residual norm below which a solve is feasible
            (default 1e-8).
        max_steps: Maximum Newton steps in the final, differentiated solve.
        n_globalize_steps: Globalization iterations per phase, run before the
            final solve; see :func:`_globalize`.
        n_continuation_steps: Continuation steps on the rare-earth feed; see
            :func:`solve_section`.
        anion_closure: ``"total"`` or ``"charge"``.
        base_addition: Molar flow (mol/s) of strong monoacidic base dosed into
            the aqueous feed. This is the closed model's input degree of
            freedom in place of the correlation's specified pH. #197 will add
            saponification degree as the organic-side equivalent; the
            counter-ion is already a conserved component so that will not
            change this interface.

    Example:
        >>> p = MassActionParams(
        ...     n_stages=4, extractant="D2EHPA", elements=("Nd", "Dy"),
        ...     aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
        ... )
        >>> p["n_stages"]
        4
    """
    n_stages: int
    extractant: str
    elements: tuple[str, ...]
    aqueous_volumetric_flow: float
    organic_volumetric_flow: float
    diluent: str = "kerosene"
    counter_ion: str | None = "Na"
    anion: str = "Cl"
    extractant_conc: float = 0.5
    calibration_pH: float = 3.0
    log10_K: Mapping[str, float] | None = None
    network: str | None = None
    anion_conc: float | None = None
    inner_tol: float = 1e-12
    feasible_tol: float = 1e-8
    max_steps: int = 200
    n_globalize_steps: int = 100
    n_continuation_steps: int = 1
    anion_closure: str = "total"
    base_addition: float | Array = 0.0

    def __post_init__(self) -> None:
        """Validate the section parameters.

        Raises:
            ValueError: On an unknown extractant or element, a non-positive
                volumetric flow, fewer than one stage, or an unknown anion
                closure.
        """
        from difflow_ree.database import get_extractant_database, get_ree_database

        if self.extractant not in get_extractant_database().list_extractants():
            raise ValueError(
                f"Unknown extractant: {self.extractant!r}. Available: "
                f"{get_extractant_database().list_extractants()}"
            )
        valid = get_ree_database().list_elements()
        for el in self.elements:
            if el not in valid:
                raise ValueError(
                    f"Unknown REE element: {el!r}. Valid elements: {valid}"
                )
        if int(self.n_stages) < 1:
            raise ValueError(f"n_stages must be >= 1, got {self.n_stages}")
        for name in ("aqueous_volumetric_flow", "organic_volumetric_flow"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(
                    f"{name} must be > 0, got {value}. The closed model works "
                    f"in concentrations, so it needs the phase volumes the "
                    f"correlation could do without (#196)."
                )
        if self.extractant_conc <= 0:
            raise ValueError(
                f"extractant_conc must be > 0, got {self.extractant_conc}"
            )
        if self.anion_closure not in ANION_CLOSURES:
            raise ValueError(
                f"anion_closure must be one of {list(ANION_CLOSURES)}, got "
                f"{self.anion_closure!r}"
            )


class MassActionSection:
    """Counter-current extraction section closed by mass action (#196).

    The interface is the one ``REEExtractor`` already has --
    ``(feed, solvent) -> (raffinate, extract, info)`` -- so cascade code is
    level-agnostic. What differs, and cannot be hidden, is that ``pH`` is an
    *output* here: ``info["pH_profile"]`` is solved for, and the input that
    replaces it is ``base_addition``.

    Attributes:
        params: The :class:`MassActionParams`.
        schema: The :class:`~difflow_ree.equilibrium.schema.REEStreamSchema`
            this section reads and writes.
        network: The expanded
            :class:`~difflow_ree.equilibrium.network.ReactionNetwork`.

    Example:
        >>> params = MassActionParams(
        ...     n_stages=3, extractant="D2EHPA", elements=("Nd", "Dy"),
        ...     aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
        ... )
        >>> section = MassActionSection(params)
        >>> feed = section.schema.make_aqueous(
        ...     {"Nd": 0.02, "Dy": 0.02}, acid=0.01, water=55.0
        ... )
        >>> solvent = section.schema.make_organic(0.5, diluent_flow=4.0)
        >>> raff, ext, info = section(feed, solvent)
        >>> bool(info["feasible"])
        True
    """

    symbol = "REE Mass-Action Section"
    equations = [
        r"\mathrm{RE}^{3+} + 3\,\overline{(\mathrm{HA})_2} \rightleftharpoons \overline{\mathrm{RE}(\mathrm{HA}_2)_3} + 3\,\mathrm{H}^+",
        r"\log_{10}[S_j] = \log_{10}K_j + \sum_c \nu_{jc}\log_{10}[C_c]",
        r"A_{c,k} + O_{c,k} - \left(A_{c,k-1} + O_{c,k+1}\right) = 0",
    ]
    assumptions = [
        "Equilibrium stages; constant volumetric phase flows across the section.",
        "Conditional equilibrium constants at the medium's ionic strength; no activity model.",
        "No water dissociation, hydrolysis or aqueous complexation unless declared in the network.",
        "Phases immiscible; no third phase.",
    ]
    references = [
        "Rydberg, J., Musikas, C., Choppin, G.R. Principles and Practices of Solvent Extraction, Marcel Dekker, 1992.",
        "Iloeje, C.O. et al. Environ. Sci. Technol. 53, 8926 (2019). doi:10.1021/acs.est.9b01718",
    ]
    numerical_method = (
        "Section-scope Newton on the component balances in log concentration, "
        "via optimistix (implicit differentiation), started from the L1 "
        "correlation."
    )

    def __init__(self, params: MassActionParams):
        """Build the schema and calibrate the reaction network.

        Args:
            params: Section parameters.
        """
        self.params = params
        self.schema = REEStreamSchema(
            elements=tuple(params.elements),
            extractant=params.extractant,
            diluent=params.diluent,
            counter_ion=params.counter_ion,
            anion=params.anion,
        )
        template = get_network_template(
            params.network or network_for_extractant(params.extractant)
        )
        ext = get_extractant(params.extractant)
        anion_conc = params.anion_conc
        if anion_conc is None:
            anion_conc = (
                ext.reference_nitrate if ext.reference_nitrate else 1.0
            )
        # A solvating extractant's D is driven by the salting anion rather
        # than by pH, so the correlation it is calibrated against needs that
        # concentration too (#195).
        dist_kwargs = (
            {"nitrate_conc": anion_conc} if ext.requires_nitrate else {}
        )
        log10_K = (
            dict(params.log10_K) if params.log10_K
            else log_K_from_correlation(
                template,
                params.elements,
                params.extractant,
                calibration_pH=params.calibration_pH,
                extractant_conc=params.extractant_conc,
                anion_conc=anion_conc,
                **dist_kwargs,
            )
        )
        self.network = build_network(template, params.elements, log10_K)

        # The anion's formal charge is stated in two places -- on the schema
        # and on the network row -- and a disagreement would silently break
        # the aqueous charge balance and the electroneutral feed closure.
        network_charge = self.network.components[self.network.anion_index].charge
        if network_charge != self.schema.anion_charge:
            raise ValueError(
                f"Anion {params.anion!r} has charge "
                f"{self.schema.anion_charge:+d} but reaction network "
                f"{template.name!r} declares its anion component with charge "
                f"{network_charge:+d}. The shipped networks are written for a "
                f"monovalent anion; a divalent anion such as sulfate needs its "
                f"own network row with the charge (and, for a solvating or "
                f"anion-exchange complex, the stoichiometry) corrected (#196)."
            )

        self._distribution = REEDistribution(
            extractant=params.extractant,
            elements=tuple(params.elements),
            concentration=params.extractant_conc,
            **dist_kwargs,
        )

    # -- helpers ---------------------------------------------------------

    def correlation_D(
        self,
        pH: Array | float | None = None,
        T: Array | float = 298.15,
    ) -> Array:
        """Distribution ratios from the L1 correlation, in element order.

        Used for the starting point, and available so the two levels can be
        compared directly.

        Args:
            pH: pH for the correlation; None uses ``calibration_pH``.
            T: Temperature (K).

        Returns:
            ``(n_elements,)`` array of ``D``.
        """
        if pH is None:
            pH = self.params.calibration_pH
        return jnp.stack(
            [self._distribution.get_D(el, pH, T) for el in self.params.elements]
        )

    def component_totals(
        self,
        feed: Stream,
        solvent: Stream,
        base_addition: Array | float | None = None,
    ) -> tuple[Array, Array]:
        """Component totals entering the section from each phase.

        Args:
            feed: Aqueous feed stream.
            solvent: Organic solvent stream.
            base_addition: Overrides ``params.base_addition``.

        Returns:
            ``(feed_totals, solvent_totals)``, each ``(n_components,)`` mol/s.
        """
        if base_addition is None:
            base_addition = self.params.base_addition
        return (
            aqueous_component_totals(
                self.network, self.schema, feed, base_addition
            ),
            organic_component_totals(self.network, self.schema, solvent),
        )

    def solve(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float = 298.15,
        base_addition: Array | float | None = None,
        u0: Array | None = None,
    ) -> MassActionSolution:
        """Solve the section and return the raw log-concentration state.

        Args:
            feed: Aqueous feed stream.
            solvent: Organic solvent stream.
            T: Temperature (K), used only for the correlation start.
            base_addition: Strong-base dosing into the aqueous feed (mol/s);
                None uses ``params.base_addition``.
            u0: Explicit starting point.

        Returns:
            The :class:`MassActionSolution`.
        """
        p = self.params
        feed_totals, solvent_totals = self.component_totals(
            feed, solvent, base_addition
        )
        return solve_section(
            self.network,
            int(p.n_stages),
            feed_totals,
            solvent_totals,
            p.aqueous_volumetric_flow,
            p.organic_volumetric_flow,
            self.correlation_D(T=T),
            u0=u0,
            inner_tol=p.inner_tol,
            feasible_tol=p.feasible_tol,
            max_steps=p.max_steps,
            n_globalize_steps=p.n_globalize_steps,
            n_continuation_steps=p.n_continuation_steps,
            anion_closure=p.anion_closure,
        )

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float = 298.15,
        base_addition: Array | float | None = None,
        u0: Array | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Run the section.

        Args:
            feed: Aqueous feed stream on this section's schema.
            solvent: Organic solvent stream on this section's schema.
            T: Temperature (K).
            base_addition: Strong-base dosing into the aqueous feed (mol/s).
            u0: Explicit starting point for the solve.

        Returns:
            ``(raffinate, extract, info)``.

            ``info`` carries ``pH_profile`` (an **output**, one value per
            stage, responding to the three protons released per trivalent
            ion), ``pH`` (the raffinate stage), ``residual_norm``,
            ``feasible``, ``solution``, ``D`` (per element, from the closed
            model, not the correlation), ``theta`` (organic loading fraction
            per stage), ``free_extractant`` (M), ``charge_imbalance``,
            ``equilibrium_closure`` and ``component_totals_in``.

        Note:
            Component conservation is structural: the organic outlet is read
            from stage 0 and the aqueous outlet is formed by difference, so
            every component balances to round-off no matter how the
            equilibrium converged. ``info["equilibrium_closure"]`` reports the
            tolerance-sized gap against the aqueous phase the solve predicts.
        """
        p = self.params
        net = self.network
        schema = self.schema
        n_stages = int(p.n_stages)

        feed_totals, solvent_totals = self.component_totals(
            feed, solvent, base_addition
        )
        solution = self.solve(feed, solvent, T, base_addition, u0)

        _, phase_flows = make_section_residual(net, n_stages, p.anion_closure)
        args = {
            "ln_K": solution.ln_K,
            "Q_aq": jnp.asarray(p.aqueous_volumetric_flow, dtype=jnp.float64),
            "Q_org": jnp.asarray(p.organic_volumetric_flow, dtype=jnp.float64),
        }
        aq, org = phase_flows(solution.u, args)

        total_in = feed_totals + solvent_totals
        organic_out = org[0]
        aqueous_out = total_in - organic_out  # exact by construction

        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)
        c = solution.concentrations(net)

        raffinate_flows: dict[str, Array | float] = dict(feed_flows)
        extract_flows: dict[str, Array | float] = dict(solvent_flows)

        for i, el in enumerate(net.elements):
            col = net.element_component_index[i]
            j = net.element_species_index[i]
            extract_flows[el] = args["Q_org"] * c[net.species_names[j]][0]
            raffinate_flows[el] = aqueous_out[col]
        raffinate_flows[schema.acid] = aqueous_out[net.proton_index]
        raffinate_flows[schema.anion] = aqueous_out[net.anion_index]
        if schema.counter_ion is not None and net.counter_ion_index is not None:
            raffinate_flows[schema.counter_ion] = aqueous_out[
                net.counter_ion_index
            ]
        # Co-extracted acid was counted into the proton total, and the shipped
        # networks have no organic species to hold it, so it leaves aqueous.
        raffinate_flows[schema.organic_acid] = jnp.asarray(0.0)
        extract_flows[schema.organic_acid] = jnp.asarray(0.0)
        # The extractant is conserved in the organic phase; take it from the
        # inlet so it is exact rather than tolerance-limited.
        extract_flows[schema.extractant] = jnp.asarray(
            solvent_flows.get(schema.extractant, 0.0), dtype=jnp.float64
        )
        for key in (schema.acid, schema.anion, schema.water):
            extract_flows.pop(key, None)
        if schema.counter_ion:
            extract_flows.pop(schema.counter_ion, None)
        raffinate_flows.pop(schema.extractant, None)
        raffinate_flows.pop(schema.diluent, None)
        raffinate_flows.pop(schema.organic_water, None)

        P = feed["P"]
        raffinate = make_stream(raffinate_flows, T, P)
        extract = make_stream(extract_flows, T, P)

        ln10 = float(np.log(10.0))
        pH_profile = -solution.u[:, net.proton_index] / ln10
        free_ext = jnp.exp(solution.u[:, net.extractant_index])
        loaded = jnp.stack(
            [c[net.species_names[net.element_species_index[i]]]
             for i in range(len(net.elements))],
            axis=1,
        )
        q = jnp.asarray(
            [net.nu[net.element_species_index[i], net.extractant_index]
             for i in range(len(net.elements))],
            dtype=jnp.float64,
        )
        c_ext_total = solvent_totals[net.extractant_index] / args["Q_org"]
        theta = jnp.sum(q[None, :] * loaded, axis=1) / c_ext_total

        D_closed = jnp.stack(
            [loaded[0, i]
             / jnp.exp(solution.u[0, net.element_component_index[i]])
             for i in range(len(net.elements))]
        )

        info = {
            "n_stages": n_stages,
            "T": jnp.asarray(T),
            # pH is an OUTPUT of this model, not an input (#196).
            "pH_profile": pH_profile,
            "pH": pH_profile[-1],
            "residual_norm": solution.residual_norm,
            "feasible": solution.feasible,
            "solution": solution,
            "D": {el: D_closed[i] for i, el in enumerate(net.elements)},
            "theta": theta,
            "free_extractant": free_ext,
            "charge_imbalance": charge_imbalance(net, solution),
            # How far the balance-closed aqueous outlet sits from the one the
            # converged stage-(N-1) aqueous phase predicts: tolerance-sized,
            # and reported so the structural closure is visible.
            "equilibrium_closure": jnp.max(jnp.abs(aqueous_out - aq[-1])),
            "component_totals_in": total_in,
            "component_names": net.component_names,
            "base_addition": jnp.asarray(
                p.base_addition if base_addition is None else base_addition,
                dtype=jnp.float64,
            ),
        }
        return raffinate, extract, info


# =============================================================================
# The degrees-of-freedom bridge between the two levels
# =============================================================================

def base_addition_bounds(
    section: "MassActionSection",
    feed: Stream,
    solvent: Stream,
) -> tuple[Array, Array]:
    """Range of base dosing over which the section still has a solution.

    Base addition is not unbounded. Summing the proton balance over the whole
    section telescopes to

        ``T_H(b) = Q_aq [H+]_raffinate - p * (rare earth extracted)``

    so a solution with a positive free proton concentration exists only while
    ``T_H(b)`` stays above ``-p * (what can actually be extracted)``, and what
    can be extracted is capped by the smaller of the rare earth present and the
    extractant inventory. Past that point every proton in the section has been
    neutralized and the model has no root at all -- which is a statement about
    the chemistry, not a solver failure.

    Below, ``b`` may only go as negative as the counter-ion the feed already
    carries: negative dosing here means *removing* base, and the counter-ion
    total cannot go negative. With no counter-ion in the feed the lower bound
    is therefore zero, and a pH target below the un-dosed pH is unreachable.

    Args:
        section: A configured :class:`MassActionSection`.
        feed: Aqueous feed stream.
        solvent: Organic solvent stream.

    Returns:
        ``(b_lo, b_hi)`` in mol/s. ``b_hi`` carries a 1% margin off the true
        singularity so the solve stays finite at the bound.
    """
    net = section.network
    feed_totals, solvent_totals = section.component_totals(feed, solvent, 0.0)

    n_el = len(net.elements)
    protons = jnp.asarray(
        [abs(net.nu[net.element_species_index[i], net.proton_index])
         for i in range(n_el)], dtype=jnp.float64,
    )
    q = jnp.asarray(
        [net.nu[net.element_species_index[i], net.extractant_index]
         for i in range(n_el)], dtype=jnp.float64,
    )
    cols = jnp.asarray(net.element_component_index)
    re_total = feed_totals[cols] + solvent_totals[cols]
    capacity = solvent_totals[net.extractant_index] / jnp.max(
        jnp.maximum(q, 1e-30)
    )
    extractable = jnp.minimum(jnp.sum(re_total), capacity)
    releasable = jnp.max(protons) * extractable

    T_H0 = feed_totals[net.proton_index] + solvent_totals[net.proton_index]
    b_hi = 0.99 * (T_H0 + releasable)
    if net.counter_ion_index is None:
        b_lo = jnp.asarray(0.0, dtype=jnp.float64)
    else:
        b_lo = -(
            feed_totals[net.counter_ion_index]
            + solvent_totals[net.counter_ion_index]
        )
    return b_lo, jnp.maximum(b_hi, b_lo)


def base_addition_for_pH(
    section: "MassActionSection",
    feed: Stream,
    solvent: Stream,
    target_pH: Array | float,
    stage: int = -1,
    T: Array | float = 298.15,
    bracket: tuple[Array | float, Array | float] | None = None,
    n_bisection_steps: int = 45,
    tol: float = 1e-10,
    max_steps: int = 20,
) -> tuple[Array, Array]:
    """Invert the closed model for the base rate that hits a specified pH.

    This is the explicit inverse problem #196 asks for, and the reason it has
    to exist. The two levels do not share degrees of freedom: pH is an
    **input** to the correlation and an **output** of the closed model, whose
    corresponding input is base addition (or, once #197 lands, saponification
    degree). A design specified at one level is therefore not directly
    expressible at the other, and quietly reusing the same number behind a
    shared interface is how a "level-agnostic" cascade changes meaning without
    saying so. Solve this instead:

        given ``pH*`` at a stage, find the base rate ``b`` with
        ``pH_stage(b) = pH*``.

    Method. The map is monotone -- more base, less free acid, higher pH -- but
    it is also close to exponential in ``b`` and it *ends*: past
    :func:`base_addition_bounds` there is no root at all. So the bracket is
    computed from the proton balance first and bisected inside it (that part
    runs under ``stop_gradient`` and only locates the root), and the answer
    then comes from a single *augmented* root find: the section's own
    component balances with one extra unknown, the base rate, and one extra
    row, "the pH at this stage equals the target". Posing the specification as
    another row of ``r`` rather than as an outer loop around an inner solve is
    what makes the derivative fall out of one ``optimistix`` implicit
    differentiation -- ``d b / d pH*`` and the sensitivity of the required
    dosing to the feed both come from it -- and it is also what makes an
    unreachable target a soft failure rather than an exception raised inside
    the adjoint of an inner solve that has no solution.

    Cost. Each bisection step is a full section solve, so this is roughly
    ``n_bisection_steps`` times the cost of one section. It is a design-time
    utility, not something to put in an inner optimization loop; if you need
    that, differentiate through it once and use the linearization.

    Args:
        section: A configured :class:`MassActionSection`.
        feed: Aqueous feed stream.
        solvent: Organic solvent stream.
        target_pH: Desired pH (concentration scale).
        stage: Stage index whose pH is specified. -1 (default) is the
            raffinate end, 0 the feed end.
        T: Temperature (K).
        bracket: ``(b_lo, b_hi)`` in mol/s. None (default) uses
            :func:`base_addition_bounds`.
        n_bisection_steps: Bisection iterations. 45 halvings resolve the
            bracket to about 1e-13 of its width.
        tol: Tolerance for the augmented Newton polish.
        max_steps: Maximum Newton steps in the polish.

    Returns:
        ``(base_addition, feasible)``: the base molar flow (mol/s), and a
        boolean array that is True when the augmented system converged *and*
        the dosing it asks for lies inside :func:`base_addition_bounds`. A
        target that would need more base than the section has protons to
        accept, or that would need base *removed* that the feed does not
        carry, comes back with ``feasible=False`` and ``b`` clipped to the
        nearest bound -- soft failure, as everywhere else here, never an
        exception.

    Example:
        >>> b, ok = base_addition_for_pH(section, feed, solvent, 3.2)  # doctest: +SKIP
        >>> float(b) > 0.0                                             # doctest: +SKIP
        True
    """
    net = section.network
    schema = section.schema
    params = section.params
    n_stages = int(params.n_stages)
    n_comp = net.n_components
    ln10 = float(np.log(10.0))
    target = jnp.asarray(target_pH, dtype=jnp.float64)

    if bracket is None:
        b_lo, b_hi = base_addition_bounds(section, feed, solvent)
    else:
        b_lo = jnp.asarray(bracket[0], dtype=jnp.float64)
        b_hi = jnp.asarray(bracket[1], dtype=jnp.float64)

    # --- locate the root by bisection (no derivative flows from here) ----
    def pH_at(b):
        sol = section.solve(feed, solvent, T=T, base_addition=b)
        # A non-converged trial is treated as "too much base", which is the
        # direction it fails in: the free proton concentration has been driven
        # to zero and the section has no root at all past that point.
        return jnp.where(sol.feasible, sol.pH(net)[stage], jnp.inf)

    def bisect(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        too_high = pH_at(mid) > target
        return jnp.where(too_high, lo, mid), jnp.where(too_high, mid, hi)

    lo, hi = lax.fori_loop(0, n_bisection_steps, bisect, (b_lo, b_hi))
    b0 = lax.stop_gradient(0.5 * (lo + hi))
    start = lax.stop_gradient(
        section.solve(feed, solvent, T=T, base_addition=b0).u.reshape(-1)
    )

    # --- polish the augmented system, which is where the derivative is ---
    # The specification is simply one more row of r: the section balances plus
    # "the pH at this stage equals the target", with the base rate as one more
    # unknown. That keeps everything inside a single root find, so there is no
    # nested implicit differentiation, and an unreachable target comes back as
    # a soft failure instead of blowing up inside the adjoint of an inner solve
    # that has no solution (#196).
    b_scale = jnp.maximum(jnp.maximum(jnp.abs(b_hi), jnp.abs(b_lo)), 1e-12)
    residual_fn, _ = make_section_residual(net, n_stages, params.anion_closure)
    solvent_totals = organic_component_totals(net, schema, solvent)
    scale = lax.stop_gradient(
        section_scales(
            net,
            aqueous_component_totals(net, schema, feed, b0),
            solvent_totals,
        )
    )
    solvent_totals_f = floor_totals(net, solvent_totals, scale)

    def augmented(zz, args):
        u = zz[:-1]
        b = zz[-1] * b_scale
        feed_totals = floor_totals(
            net, aqueous_component_totals(net, schema, feed, b), scale
        )
        inner = residual_fn(
            u,
            {
                "ln_K": net.ln_K(),
                "Q_aq": jnp.asarray(
                    params.aqueous_volumetric_flow, dtype=jnp.float64
                ),
                "Q_org": jnp.asarray(
                    params.organic_volumetric_flow, dtype=jnp.float64
                ),
                "feed_totals": feed_totals,
                "solvent_totals": solvent_totals_f,
                "scale": scale,
            },
        )
        pH = -u.reshape(n_stages, n_comp)[stage, net.proton_index] / ln10
        return jnp.concatenate([inner, jnp.reshape(pH - target, (1,))])

    z0 = jnp.concatenate([start, jnp.reshape(b0 / b_scale, (1,))])
    z, _, feasible = solve_residual_system(
        augmented, z0, None, rtol=tol, atol=tol, max_steps=max_steps,
        feasible_tol=1e-8,
    )
    # Clipped to the bracket so an unreachable target comes back as the
    # nearest reachable dosing rather than as a meaningless extrapolation. It
    # comes back with feasible=False either way.
    b_raw = z[-1] * b_scale
    b = jnp.clip(jnp.where(jnp.isfinite(b_raw), b_raw, b_hi), b_lo, b_hi)
    # Converging is not the same as being realizable. The augmented system is
    # unbounded in b, so it will happily report a root that doses more base
    # than the section has protons to accept, or that *removes* more
    # counter-ion than the feed carries. Both are outside
    # :func:`base_addition_bounds` and neither is a dosing anyone can apply,
    # so they come back as infeasible with b clipped to the nearest bound.
    slack = 1e-9 * jnp.maximum(jnp.abs(b_hi - b_lo), 1e-30)
    feasible = jnp.logical_and(
        feasible,
        jnp.logical_and(b_raw >= b_lo - slack, b_raw <= b_hi + slack),
    )
    return b, feasible
