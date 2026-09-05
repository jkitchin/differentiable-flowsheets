"""Tests for the REE mass-action equilibrium closure (#196).

Issue #196: ``difflow_ree`` had no mass-action equilibrium model. The reaction
``RE3+ + 3 HA <-> REA3 + 3 H+`` and its constant appeared in a LaTeX docstring
while what ran was an empirical ``log10(D)`` correlation, so pH was a
parameter rather than a state, competitive loading was a multiplicative
correction rather than an outcome of one shared extractant balance, and
extractant selection had no physical grounding.

The three validations the issue names explicitly are
:func:`test_dilute_limit_reduces_to_correlation`,
:func:`test_every_component_conserved_to_machine_precision` and
:func:`test_check_grads_through_the_section`. The rest of the file covers the
claims those three do not reach: that the reaction network is genuinely data
(``test_counter_ion_*``, which adds #197's saponification reaction as a table
row and nothing else), that pH is an output responding to three protons per
trivalent ion, that the log-space solve stays conditioned across a cascade
spanning ten orders of magnitude, and that failure is returned rather than
raised under ``vmap``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow.eo_solver import solve_residual_system
from difflow_ree.database import get_extractant
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.mass_action import (
    ANION_CLOSURES,
    MassActionParams,
    MassActionSection,
    MassActionSolution,
    aqueous_component_totals,
    base_addition_bounds,
    base_addition_for_pH,
    charge_imbalance,
    make_section_residual,
    organic_component_totals,
    solve_section,
    solve_stage,
)
from difflow_ree.equilibrium.network import (
    NETWORK_MECHANISMS,
    Reaction,
    build_network,
    correlation_ph_slope_defect,
    get_network_template,
    list_networks,
    log_K_from_correlation,
    network_for_extractant,
)
from difflow_ree.equilibrium.schema import (
    ANION_CHARGES,
    COUNTER_ION_CHARGES,
    REEStreamSchema,
)
from difflow_ree.units.extraction import (
    EXTRACTOR_MODELS,
    REEExtractor,
    REEExtractorParams,
)


LN10 = float(np.log(10.0))


# =============================================================================
# Helpers
# =============================================================================

def make_section(
    elements=("Nd", "Dy"),
    n_stages=3,
    extractant="D2EHPA",
    calibration_pH=3.0,
    Q_aq=1.0,
    Q_org=1.0,
    **kwargs,
) -> MassActionSection:
    """Build a small configured section for the tests."""
    return MassActionSection(MassActionParams(
        n_stages=n_stages,
        extractant=extractant,
        elements=tuple(elements),
        aqueous_volumetric_flow=Q_aq,
        organic_volumetric_flow=Q_org,
        calibration_pH=calibration_pH,
        **kwargs,
    ))


def streams(section, element_flows, acid, extractant_flow=0.5,
            counter_ion=0.0, loaded=None):
    """Feed and solvent streams on a section's own schema."""
    feed = section.schema.make_aqueous(
        element_flows, acid=acid, counter_ion=counter_ion, water=55.0
    )
    solvent = section.schema.make_organic(
        extractant_flow, diluent_flow=4.0, element_flows=loaded
    )
    return feed, solvent


# =============================================================================
# The reaction network is data (#196), and it is checkable data
# =============================================================================

def test_shipped_networks_cover_the_four_mechanisms():
    """The four mechanisms are rows in a table, not four code paths."""
    names = list_networks()
    assert "cation_exchange_dimer" in names
    assert "cation_exchange_monomer" in names
    assert "solvating_nitrate" in names
    assert "anion_exchange" in names
    mechanisms = {get_network_template(n).mechanism for n in names}
    assert mechanisms <= set(NETWORK_MECHANISMS)
    # Every shipped template must be buildable and charge consistent.
    for name in names:
        net = build_network(name, ("Nd",), log10_K={"Nd": -6.0})
        implied = net.nu @ net.component_charges
        assert np.allclose(implied, net.species_charges)


def test_network_dispatch_follows_the_extractant_record():
    """Which network is used comes from the record's mechanism and basis."""
    assert network_for_extractant("D2EHPA") == "cation_exchange_dimer"
    assert network_for_extractant("PC88A") == "cation_exchange_dimer"
    assert network_for_extractant("TBP") == "solvating_nitrate"


def test_section_residual_is_square():
    """One unknown and one balance per component, per stage."""
    net = build_network(
        "cation_exchange_dimer", ("Nd", "Dy"),
        log10_K={"Nd": -7.0, "Dy": -5.0},
    )
    residual_fn, _ = make_section_residual(net, 4)
    u = jnp.full(4 * net.n_components, -3.0)
    args = {
        "ln_K": net.ln_K(),
        "Q_aq": jnp.asarray(1.0),
        "Q_org": jnp.asarray(1.0),
        "feed_totals": jnp.ones(net.n_components),
        "solvent_totals": jnp.ones(net.n_components),
        "scale": jnp.ones(net.n_components),
    }
    assert residual_fn(u, args).shape == u.shape
    jacobian = jax.jacobian(residual_fn)(u, args)
    assert jacobian.shape == (u.size, u.size)
    # The section Jacobian is what the linearization layers want, so it has to
    # be non-singular at a sane point.
    assert np.linalg.matrix_rank(np.asarray(jacobian)) == u.size


def test_per_element_expansion():
    """One table row becomes one column/row per tracked element."""
    net = build_network(
        "cation_exchange_dimer", ("Nd", "Dy"),
        log10_K={"Nd": -7.0, "Dy": -5.0},
    )
    assert net.component_names == ("Nd3+", "Dy3+", "H+", "M+", "X-", "(HA)2")
    assert net.species_names == ("Nd(HA2)3", "Dy(HA2)3")
    # Three protons released and three dimers bound, from the data.
    row = net.nu[net.element_species_index[0]]
    assert row[net.proton_index] == -3.0
    assert row[net.extractant_index] == 3.0
    assert float(net.log10_K[1]) == -5.0


def test_charge_inconsistent_network_is_rejected():
    """A mistyped stoichiometric coefficient is caught, not absorbed.

    This is the check that makes the table trustworthy: a wrong coefficient is
    otherwise invisible until the charge balance quietly drifts.
    """
    template = get_network_template("cation_exchange_dimer")
    broken = dataclasses.replace(
        template,
        species=(
            Reaction(
                name="RE(HA2)3",
                phase="organic",
                charge=0,
                # Two protons released for a trivalent ion: not neutral.
                stoichiometry={"RE3+": 1, "(HA)2": 3, "H+": -2},
            ),
        ),
        per_element_species=frozenset({"RE(HA2)3"}),
    )
    with pytest.raises(ValueError, match="not charge consistent"):
        build_network(broken, ("Nd",), log10_K={"Nd": -6.0})


def test_basis_mismatch_between_record_and_network_is_rejected():
    """The dimer/monomer basis is stated twice, so it can be checked (#191)."""
    with pytest.raises(ValueError, match="extractant_basis"):
        log_K_from_correlation(
            "cation_exchange_monomer", ("Nd",), "D2EHPA", calibration_pH=3.0
        )


# =============================================================================
# The counter-ion is already a conserved component: #197's slot
# =============================================================================

def test_counter_ion_is_a_conserved_component_in_every_shipped_network():
    """#197 needs the counter-ion conserved before it can saponify anything."""
    for name in list_networks():
        net = build_network(name, ("Nd",), log10_K={"Nd": -6.0})
        assert net.counter_ion_index is not None
        assert net.components[net.counter_ion_index].role == "counter_ion"
        assert net.components[net.counter_ion_index].charge == 1


def test_counter_ion_becomes_a_partitioning_component_with_one_table_row():
    """Saponification (#197) is one added row and no code change at all.

    The point of #196's data model is that the next mechanism is a row in a
    table. This test adds exactly the row the YAML comments describe --
    ``M(HA2)``, formed from the counter-ion component that already exists --
    and solves a section with it through the unchanged
    :func:`solve_section`. If any of the closure had been written for
    cation exchange specifically, this would fail.
    """
    template = get_network_template("cation_exchange_dimer")
    saponified = dataclasses.replace(
        template,
        species=template.species + (
            Reaction(
                name="M(HA2)",
                phase="organic",
                charge=0,
                stoichiometry={"M+": 1, "(HA)2": 1, "H+": -1},
                log10_K=-1.5,
            ),
        ),
    )
    net = build_network(saponified, ("Nd",), log10_K={"Nd": -7.45})
    assert "M(HA2)" in net.species_names

    schema = REEStreamSchema(elements=("Nd",), extractant="D2EHPA")
    feed = schema.make_aqueous({"Nd": 0.01}, acid=0.01, counter_ion=0.2,
                               water=55.0)
    solvent = schema.make_organic(0.5, diluent_flow=4.0)
    feed_totals = aqueous_component_totals(net, schema, feed)
    solvent_totals = organic_component_totals(net, schema, solvent)
    D = jnp.array([1.0])

    sol = solve_section(net, 2, feed_totals, solvent_totals, 1.0, 1.0, D)
    assert bool(sol.feasible)

    c = sol.concentrations(net)
    # Sodium now genuinely partitions: there is saponified extractant in the
    # organic phase and less free extractant than the total.
    assert float(c["M(HA2)"][0]) > 0.0
    assert float(c["(HA)2"][0]) < 0.25

    # And it is still conserved: total M in = free M+ out + M(HA2) out.
    total_M_in = float(feed_totals[net.counter_ion_index])
    total_M_out = float(c["M+"][-1] * 1.0 + c["M(HA2)"][0] * 1.0)
    assert total_M_out == pytest.approx(total_M_in, rel=1e-10)


# =============================================================================
# A different mechanism is a different row, not a different code path
# =============================================================================

def test_solvating_extraction_runs_through_the_same_closure():
    """TBP goes through ``solve_section`` unchanged, and behaves differently.

    Nothing in :mod:`difflow_ree.equilibrium.mass_action` mentions cation
    exchange. Selecting a solvating extractant selects a different row of
    ``reaction_networks.yaml`` -- the complex contains three nitrates and no
    protons -- and the same closure then predicts the two things that
    distinguish solvating extraction:

    - the pH is a spectator, constant across the section, because no proton
      appears in the complex's stoichiometry;
    - the salting effect is not a correction. ``D`` rises as the *cube* of the
      free nitrate, because the anion is a conserved component that the
      complex draws three of.
    """
    section = MassActionSection(MassActionParams(
        n_stages=2, extractant="TBP", elements=("Nd", "Dy"),
        aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
        anion="NO3", extractant_conc=1.0,
    ))
    assert section.network.name == "solvating_nitrate"
    row = section.network.nu[section.network.element_species_index[0]]
    assert row[section.network.proton_index] == 0.0
    assert row[section.network.anion_index] == 3.0

    solvent = section.schema.make_organic(1.0, diluent_flow=4.0)
    results = {}
    for extra_nitrate in (0.0, 1.0):
        feed = section.schema.make_aqueous(
            {"Nd": 1e-6, "Dy": 1e-6}, acid=0.1, counter_ion=extra_nitrate,
            water=55.0,
        )
        _, _, info = section(feed, solvent)
        assert bool(info["feasible"])
        # No proton is exchanged, so the profile is flat and equals the feed.
        profile = np.asarray(info["pH_profile"])
        assert np.allclose(profile, 1.0, atol=1e-9)
        results[extra_nitrate] = float(info["D"]["Nd"])

    # [NO3-] goes from ~0.1 M to ~1.1 M, and three of them enter the complex.
    assert results[1.0] / results[0.0] == pytest.approx(
        (1.1 / 0.1) ** 3, rel=1e-3
    )


def test_network_template_can_be_overridden():
    """The network is selectable data, not an inference from the extractant."""
    section = MassActionSection(MassActionParams(
        n_stages=2, extractant="D2EHPA", elements=("Nd",),
        aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
        network="cation_exchange_dimer",
    ))
    assert section.network.name == "cation_exchange_dimer"


def test_divalent_anion_is_refused_rather_than_mis_charged():
    """The anion charge is stated twice, so a disagreement is caught (#196)."""
    with pytest.raises(ValueError, match="declares its anion component"):
        MassActionSection(MassActionParams(
            n_stages=2, extractant="D2EHPA", elements=("Nd",),
            aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
            anion="SO4",
        ))


# =============================================================================
# Calibration against the L1 correlation
# =============================================================================

def test_log_K_from_correlation_inverts_the_dilute_limit_algebraically():
    """log10 K = log10 D_corr - sum_c nu_c log10 [C_c]^ref, by hand."""
    K = log_K_from_correlation(
        "cation_exchange_dimer", ("Nd",), "D2EHPA",
        calibration_pH=3.0, extractant_conc=0.5,
    )
    dist = REEDistribution(
        extractant="D2EHPA", elements=("Nd",), concentration=0.5
    )
    expected = (
        float(np.log10(float(dist.get_D("Nd", 3.0))))
        - 3.0 * float(np.log10(0.25))   # three dimers at 0.5 M monomer
        - 3.0 * 3.0                     # three protons at pH 3
    )
    assert K["Nd"] == pytest.approx(expected, rel=1e-12)


def test_correlation_ph_slope_defect_is_reported_not_hidden():
    """The tabulated pH slope is not 3, and the gap is quoted, not absorbed."""
    ext = get_extractant("D2EHPA")
    for element in ("La", "Nd", "Dy"):
        defect = correlation_ph_slope_defect("D2EHPA", element)
        assert defect == pytest.approx(
            3.0 - ext.ph_coefficients[element].b, abs=1e-12
        )
        # It is genuinely non-zero: mass action and the correlation do not
        # have the same pH dependence, which is the point.
        assert abs(defect) > 0.1


# =============================================================================
# Validation 1: reduction to the correlation in the dilute limit
# =============================================================================

#: Rare-earth total, as a fraction of the free acid, that defines "dilute"
#: here. At this ratio the protons released by the trace extraction shift the
#: pH by ~3e-8 units, and three protons per ion times that is the residual
#: disagreement the tolerance below allows for.
DILUTE_RE_TO_ACID = 1e-6

#: Relative agreement in D required at that dilution. The value is not
#: arbitrary: the discrepancy is 3 * ln10 * (3 * RE / acid) in log space, i.e.
#: 1.2e-5 at DILUTE_RE_TO_ACID = 1e-6, and
#: test_dilute_limit_discrepancy_is_the_released_protons shows it scales
#: exactly with the ratio. 2e-5 leaves a factor under two of margin -- tight
#: enough that a wrong free-extractant balance, a wrong dimer basis or a wrong
#: log-space conversion would all break it.
DILUTE_TOL = 2e-5


def test_dilute_limit_reduces_to_correlation():
    """The closed model reproduces the correlation where it should (#196).

    Dilute rare earth, excess extractant, buffered at the calibration pH: the
    free extractant is the total extractant and the released protons are
    negligible, so the closed model must return exactly the correlation's D.
    """
    elements = ("La", "Nd", "Dy")
    acid = 1e-3
    section = make_section(elements=elements, n_stages=1, calibration_pH=3.0)
    feed, solvent = streams(
        section,
        {el: acid * DILUTE_RE_TO_ACID for el in elements},
        acid=acid,
    )
    _, _, info = section(feed, solvent)
    assert bool(info["feasible"])
    # Genuinely dilute: the extractant is essentially untouched.
    assert float(info["theta"][0]) < 1e-6

    dist = REEDistribution(
        extractant="D2EHPA", elements=elements, concentration=0.5
    )
    for el in elements:
        D_closed = float(info["D"][el])
        D_corr = float(dist.get_D(el, 3.0))
        assert D_closed == pytest.approx(D_corr, rel=DILUTE_TOL)


def test_dilute_limit_discrepancy_is_the_released_protons():
    """What is left of the disagreement is physics, and it scales as such.

    Making the feed ten times more dilute must make the disagreement ten times
    smaller, because the only thing left is the pH shift from the protons the
    trace extraction releases. A tolerance that merely passed would not show
    this; a constant offset (a mis-set reference concentration, say) would
    fail it.
    """
    elements = ("Nd",)
    acid = 1e-3
    dist = REEDistribution(
        extractant="D2EHPA", elements=elements, concentration=0.5
    )
    D_corr = float(dist.get_D("Nd", 3.0))

    errors = []
    for ratio in (1e-4, 1e-5, 1e-6):
        section = make_section(elements=elements, n_stages=1,
                               calibration_pH=3.0)
        feed, solvent = streams(section, {"Nd": acid * ratio}, acid=acid)
        _, _, info = section(feed, solvent)
        errors.append(abs(float(info["D"]["Nd"]) - D_corr) / D_corr)

    assert errors[1] == pytest.approx(errors[0] / 10.0, rel=0.05)
    assert errors[2] == pytest.approx(errors[1] / 10.0, rel=0.05)


def test_departure_from_the_correlation_is_the_predicted_ph_slope():
    """Away from the calibration pH the two levels differ by a known amount.

    Mass action forces ``d log10 D / d pH = protons_released`` exactly. The
    correlation uses a fitted slope ``b`` and a quadratic term ``c``. The
    difference is therefore predictable in closed form, and matching it to
    seven digits is a strong statement that the closure has the right proton
    stoichiometry rather than merely a plausible one.
    """
    ext = get_extractant("D2EHPA")
    coeffs = ext.ph_coefficients["Nd"]
    dist = REEDistribution(
        extractant="D2EHPA", elements=("Nd",), concentration=0.5
    )
    cal = 3.0
    for pH in (2.0, 2.5, 3.5, 4.0):
        section = make_section(elements=("Nd",), n_stages=1,
                               calibration_pH=cal)
        feed, solvent = streams(section, {"Nd": 1e-14}, acid=10.0 ** (-pH))
        _, _, info = section(feed, solvent)
        observed = float(
            np.log10(float(info["D"]["Nd"]) / float(dist.get_D("Nd", pH)))
        )
        predicted = (
            (ext.stoichiometry_protons - coeffs.b) * (pH - cal)
            - coeffs.c * (pH ** 2 - cal ** 2)
        )
        assert observed == pytest.approx(predicted, abs=1e-7)


def test_extractant_concentration_dependence_matches_the_correlation():
    """Two independently written paths to D's [HA] dependence must agree.

    The correlation applies ``n * log10(C / C_ref)`` with ``n = 3``. The
    closed model never sees ``n``: it gets the same dependence from three
    dimers in the tableau and a free-extractant balance on the dimer basis.
    Doubling the extractant must move both by the same factor of eight, and
    the closed model calibrated at 0.5 M must reproduce the correlation
    evaluated at 1.0 M. Nothing about this is circular.
    """
    elements = ("La", "Nd", "Dy")
    base = make_section(elements=elements, n_stages=1, calibration_pH=3.0)
    log10_K = {
        el: float(base.network.log10_K[i]) for i, el in enumerate(elements)
    }
    doubled = make_section(
        elements=elements, n_stages=1, calibration_pH=3.0,
        extractant_conc=1.0, log10_K=log10_K,
    )
    feed, solvent = streams(
        doubled, {el: 1e-9 for el in elements}, acid=1e-3, extractant_flow=1.0
    )
    _, _, info = doubled(feed, solvent)

    d05 = REEDistribution(extractant="D2EHPA", elements=elements,
                          concentration=0.5)
    d10 = REEDistribution(extractant="D2EHPA", elements=elements,
                          concentration=1.0)
    for el in elements:
        assert float(d10.get_D(el, 3.0) / d05.get_D(el, 3.0)) == pytest.approx(
            8.0, rel=1e-12
        )
        assert float(info["D"][el]) == pytest.approx(
            float(d10.get_D(el, 3.0)), rel=DILUTE_TOL
        )


# =============================================================================
# Validation 2: every component conserved to machine precision
# =============================================================================

def test_every_component_conserved_to_machine_precision():
    """Not "to the solver tolerance": to floating-point round-off (#196)."""
    elements = ("La", "Nd", "Dy")
    section = make_section(elements=elements, n_stages=5, calibration_pH=3.0,
                           Q_aq=1.0, Q_org=2.0)
    feed, solvent = streams(
        section, {"La": 0.03, "Nd": 0.02, "Dy": 0.01}, acid=0.02,
        extractant_flow=1.0, counter_ion=0.05,
        loaded={"Nd": 0.002},
    )
    raffinate, extract, info = section(feed, solvent)
    assert bool(info["feasible"])

    net, schema = section.network, section.schema
    into = np.asarray(
        aqueous_component_totals(net, schema, feed)
        + organic_component_totals(net, schema, solvent)
    )
    out = np.asarray(
        aqueous_component_totals(net, schema, raffinate)
        + organic_component_totals(net, schema, extract)
    )
    scale = np.max(np.abs(into))
    for name, a, b in zip(net.component_names, into, out):
        assert abs(b - a) < 1e-15 * scale, f"{name}: {a} -> {b}"

    # The rare earths are also checkable directly on the stream keys, and the
    # proton total is the one that is NOT: a loaded organic phase carries a
    # negative H component, so "free acid in == free acid out" is false by
    # design and only the tableau total balances.
    for el in elements:
        a = float(feed[f"F_{el}"]) + float(solvent[f"F_{el}"])
        b = float(raffinate[f"F_{el}"]) + float(extract[f"F_{el}"])
        assert abs(b - a) < 1e-15 * scale, el
    assert float(raffinate["F_H"]) != pytest.approx(
        float(feed["F_H"]), rel=1e-3
    )


def test_conservation_survives_a_deliberately_crippled_solve():
    """Conservation is structural, so it holds even when equilibrium does not.

    With the globalization switched off and one Newton step allowed, the
    section does not converge -- and every component still balances, because
    the aqueous outlet is formed by difference from the organic one rather
    than read off an unconverged aqueous phase. This is what separates
    "conserved to machine precision" from "conserved to the tolerance".
    """
    section = make_section(
        elements=("Nd", "Dy"), n_stages=3,
        n_globalize_steps=3, inner_tol=1e-1, feasible_tol=1.0, max_steps=1,
    )
    feed, solvent = streams(section, {"Nd": 0.02, "Dy": 0.02}, acid=0.01)
    raffinate, extract, info = section(feed, solvent)

    # Genuinely not converged: the balances are out by a factor of order ten.
    assert float(info["residual_norm"]) > 1.0

    net, schema = section.network, section.schema
    into = np.asarray(
        aqueous_component_totals(net, schema, feed)
        + organic_component_totals(net, schema, solvent)
    )
    out = np.asarray(
        aqueous_component_totals(net, schema, raffinate)
        + organic_component_totals(net, schema, extract)
    )
    scale = np.max(np.abs(into))
    assert np.max(np.abs(out - into)) < 1e-15 * scale


def test_charge_imbalance_is_zero_for_an_electroneutral_feed():
    """Charge balance is implied by the component balances, and reported."""
    section = make_section(elements=("Nd", "Dy"), n_stages=3)
    feed, solvent = streams(section, {"Nd": 0.02, "Dy": 0.02}, acid=0.01,
                            counter_ion=0.03)
    _, _, info = section(feed, solvent)
    assert np.max(np.abs(np.asarray(info["charge_imbalance"]))) < 1e-15
    # The same number, computed straight off the solution.
    direct = charge_imbalance(section.network, section.solve(feed, solvent))
    assert np.max(np.abs(np.asarray(direct))) < 1e-15


def test_charge_imbalance_reports_a_non_electroneutral_feed():
    """An impossible feed is visible rather than absorbed into the pH."""
    section = make_section(elements=("Nd",), n_stages=2)
    feed = section.schema.make_aqueous(
        {"Nd": 0.02}, acid=0.01, anion=0.02, water=55.0
    )
    solvent = section.schema.make_organic(0.5, diluent_flow=4.0)
    _, _, info = section(feed, solvent)
    assert np.max(np.abs(np.asarray(info["charge_imbalance"]))) > 1e-3


def test_anion_closure_by_charge_balance_is_available():
    """The alternative closure agrees when the feed is electroneutral."""
    assert set(ANION_CLOSURES) == {"total", "charge"}
    kwargs = dict(elements=("Nd", "Dy"), n_stages=3, calibration_pH=3.0)
    a = make_section(**kwargs)
    b = make_section(anion_closure="charge", **kwargs)
    feed, solvent = streams(a, {"Nd": 0.02, "Dy": 0.02}, acid=0.01)
    _, _, ia = a(feed, solvent)
    _, _, ib = b(feed, solvent)
    assert bool(ia["feasible"]) and bool(ib["feasible"])
    assert np.allclose(
        np.asarray(ia["pH_profile"]), np.asarray(ib["pH_profile"]), rtol=1e-10
    )


# =============================================================================
# Validation 3: gradients
# =============================================================================

#: One section, built once. Building a section calibrates its equilibrium
#: constants from the correlation, which is concrete Python arithmetic and
#: therefore a setup step, not something to do inside a traced function.
_GRAD_SECTION = make_section(elements=("Nd", "Dy"), n_stages=3)


def _dy_extracted(acid):
    """Dy taken into the organic phase, as a function of the feed acid."""
    feed, solvent = streams(
        _GRAD_SECTION, {"Nd": 0.02, "Dy": 0.02}, acid=acid
    )
    _, extract, _ = _GRAD_SECTION(feed, solvent)
    return extract["F_Dy"]


@pytest.mark.slow
def test_check_grads_through_the_section():
    """jax.test_util.check_grads passes through the implicit solve (#196)."""
    check_grads(_dy_extracted, (0.02,), order=1, modes=["rev"], eps=1e-6)


@pytest.mark.slow
def test_gradient_matches_central_differences():
    """A second, independent check with a step chosen for this function."""
    analytic = float(jax.grad(_dy_extracted)(0.02))
    h = 1e-7
    fd = float((_dy_extracted(0.02 + h) - _dy_extracted(0.02 - h)) / (2 * h))
    assert analytic == pytest.approx(fd, rel=1e-6)
    # And it is a real dependence, not an accidental zero.
    assert abs(analytic) > 1e-3


@pytest.mark.slow
def test_gradient_with_respect_to_equilibrium_constants():
    """log10 K is traced, so extractant selection is differentiable (#196)."""
    elements = ("Nd", "Dy")
    section = make_section(elements=elements, n_stages=3)
    # Dilute and buffered, so the answer is not dominated by the feedback of
    # the released protons on the pH.
    feed, solvent = streams(section, {"Nd": 1e-6, "Dy": 1e-6}, acid=0.01)
    feed_totals, solvent_totals = section.component_totals(feed, solvent)
    D = section.correlation_D()
    base = np.asarray(section.network.log10_K)

    def extracted(log10_K):
        sol = solve_section(
            section.network, 3, feed_totals, solvent_totals, 1.0, 1.0, D,
            log10_K=log10_K,
        )
        c = sol.concentrations(section.network)
        return c["Dy(HA2)3"][0]

    g = np.asarray(jax.grad(extracted)(jnp.asarray(base)))
    h = 1e-6
    fd = float(
        (extracted(jnp.asarray(base + np.array([0.0, h])))
         - extracted(jnp.asarray(base - np.array([0.0, h])))) / (2 * h)
    )
    assert g[1] == pytest.approx(fd, rel=1e-5)
    assert g[1] > 0.0  # a larger constant extracts more


# =============================================================================
# pH is an output (#196)
# =============================================================================

def test_ph_is_an_output_and_falls_as_rare_earth_is_extracted():
    """The profile is solved for; it is not the number that was handed in."""
    section = make_section(elements=("Nd", "Dy"), n_stages=4,
                           calibration_pH=3.0)
    feed, solvent = streams(section, {"Nd": 0.02, "Dy": 0.02}, acid=0.02)
    _, _, info = section(feed, solvent)
    profile = np.asarray(info["pH_profile"])

    feed_pH = -np.log10(0.02)
    assert not np.allclose(profile, 3.0)          # not the calibration pH
    assert np.all(profile < feed_pH)              # acid is released
    # Aqueous flow runs 0 -> N-1 and picks up protons on the way.
    assert np.all(np.diff(profile) < 0.0)


def test_three_protons_are_released_per_trivalent_ion():
    """The quantitative statement behind the profile, checked as such.

    The proton balance is what makes pH an output, so the released acid must
    equal exactly ``protons_released`` times the rare earth extracted -- not
    approximately, and not a fitted proportionality.
    """
    section = make_section(elements=("Nd", "Dy"), n_stages=4,
                           calibration_pH=3.0)
    acid_in = 0.02
    feed, solvent = streams(section, {"Nd": 0.02, "Dy": 0.02}, acid=acid_in)
    raffinate, extract, _ = section(feed, solvent)

    extracted = float(extract["F_Nd"]) + float(extract["F_Dy"])
    released = float(raffinate["F_H"]) - acid_in
    p = get_extractant("D2EHPA").stoichiometry_protons
    assert released == pytest.approx(p * extracted, rel=1e-12)
    assert extracted > 0.0


def test_more_rare_earth_means_more_acid_released():
    """Doubling the feed doubles the protons released, ion for ion."""
    released = []
    extracted = []
    for scale in (1.0, 2.0):
        section = make_section(elements=("Nd",), n_stages=3)
        feed, solvent = streams(section, {"Nd": 0.01 * scale}, acid=0.02)
        raffinate, extract, _ = section(feed, solvent)
        released.append(float(raffinate["F_H"]) - 0.02)
        extracted.append(float(extract["F_Nd"]))
    assert released[1] > released[0]
    assert released[1] / extracted[1] == pytest.approx(
        released[0] / extracted[0], rel=1e-10
    )


# =============================================================================
# Competitive loading is an outcome of one shared balance (#189, #190, #191)
# =============================================================================

def test_competition_emerges_from_the_shared_extractant_balance():
    """A second element depresses the first without any correction factor.

    Nothing in the closed model multiplies D by ``(1 - theta)^3``. The
    depression comes from one free-extractant balance and one proton balance,
    both of which the added element also draws on.
    """
    alone = make_section(elements=("Nd", "Dy"), n_stages=3, calibration_pH=3.0)
    # A modest extractant charge so the competition actually bites.
    feed_a, solvent = streams(
        alone, {"Nd": 0.05, "Dy": 0.0}, acid=0.001, extractant_flow=0.05
    )
    _, extract_a, info_a = alone(feed_a, solvent)

    feed_b, _ = streams(
        alone, {"Nd": 0.05, "Dy": 0.05}, acid=0.001, extractant_flow=0.05
    )
    _, extract_b, info_b = alone(feed_b, solvent)

    assert bool(info_a["feasible"]) and bool(info_b["feasible"])
    # Dy is the stronger extractant; adding it must take Nd off the solvent.
    assert float(extract_b["F_Nd"]) < float(extract_a["F_Nd"])
    assert float(info_b["free_extractant"][0]) < float(
        info_a["free_extractant"][0]
    )


# =============================================================================
# Conditioning, jit, vmap and soft failure
# =============================================================================

@pytest.mark.slow
def test_log_space_stays_conditioned_across_ten_orders_of_magnitude():
    """A realistic cascade spans many orders of magnitude and must not care."""
    elements = ("La", "Ce", "Nd", "Sm", "Dy", "Y")
    section = make_section(elements=elements, n_stages=8, calibration_pH=3.0,
                           Q_aq=1.0, Q_org=2.0)
    element_flows = {
        "La": 1e-10, "Ce": 1e-8, "Nd": 1e-5, "Sm": 1e-3, "Dy": 0.05, "Y": 0.2,
    }
    feed, solvent = streams(section, element_flows, acid=0.05,
                            extractant_flow=1.0)
    raffinate, extract, info = section(feed, solvent)
    assert bool(info["feasible"])
    assert float(info["residual_norm"]) < 1e-10

    concentrations = section.solve(feed, solvent).concentrations(
        section.network
    )
    values = np.concatenate([
        np.asarray(concentrations[name]) for name in section.network.component_names
    ])
    spread = np.log10(values.max() / values[values > 0].min())
    assert spread > 10.0, f"only spans {spread:.1f} decades"
    assert np.all(np.isfinite(values))
    for el in elements:
        assert float(raffinate[f"F_{el}"]) >= -1e-18
        assert float(extract[f"F_{el}"]) >= -1e-18


def test_jit_gives_the_same_answer():
    """The whole section is traceable end to end."""
    eager = float(_dy_extracted(0.02))
    compiled = float(jax.jit(_dy_extracted)(0.02))
    assert compiled == pytest.approx(eager, rel=1e-14)


@pytest.mark.slow
def test_vmap_returns_soft_failures_rather_than_raising():
    """One cannot raise from inside vmap, so failure comes back as a value."""
    section = make_section(elements=("Nd", "Dy"), n_stages=3,
                           n_globalize_steps=0, max_steps=1)

    def run(acid):
        feed, solvent = streams(section, {"Nd": 0.2, "Dy": 0.2}, acid=acid)
        _, _, info = section(feed, solvent)
        return info["feasible"], info["residual_norm"]

    feasible, norms = jax.vmap(run)(jnp.array([1e-5, 1e-4, 1e-3]))
    assert feasible.dtype == jnp.bool_
    assert feasible.shape == (3,)
    assert not bool(jnp.any(feasible))   # crippled on purpose


def test_vmap_over_a_healthy_batch_converges():
    """The same batching path with the solver left alone."""
    section = make_section(elements=("Nd", "Dy"), n_stages=3)

    def run(acid):
        feed, solvent = streams(section, {"Nd": 0.02, "Dy": 0.02}, acid=acid)
        _, _, info = section(feed, solvent)
        return info["feasible"], info["pH"]

    feasible, pH = jax.vmap(run)(jnp.array([0.005, 0.02, 0.1]))
    assert bool(jnp.all(feasible))
    # More acid, lower pH -- monotone, and an output every time.
    assert np.all(np.diff(np.asarray(pH)) < 0.0)


def test_solution_is_a_pytree():
    """MassActionSolution has to survive jit/vmap boundaries."""
    section = make_section(elements=("Nd",), n_stages=2)
    feed, solvent = streams(section, {"Nd": 0.01}, acid=0.01)
    sol = section.solve(feed, solvent)
    leaves = jax.tree_util.tree_leaves(sol)
    assert len(leaves) == 4
    rebuilt = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(sol), leaves
    )
    assert isinstance(rebuilt, MassActionSolution)


def test_continuation_path_reaches_the_same_answer():
    """The optional feed ramp changes the path, never the answer.

    Continuation runs entirely under ``stop_gradient``, so turning it on must
    move only the starting point. If it changed the converged state, the
    ``stop_gradient`` would be hiding a real dependence.
    """
    kwargs = dict(elements=("Nd", "Dy"), n_stages=3, calibration_pH=3.0)
    direct = make_section(**kwargs)
    ramped = make_section(n_continuation_steps=3, **kwargs)
    feed, solvent = streams(direct, {"Nd": 0.02, "Dy": 0.02}, acid=0.01)

    a = direct.solve(feed, solvent)
    b = ramped.solve(feed, solvent)
    assert bool(a.feasible) and bool(b.feasible)
    assert np.allclose(np.asarray(a.u), np.asarray(b.u), rtol=1e-10, atol=1e-12)


def test_solve_stage_is_a_section_of_one():
    """The stage operator is the same object with n_stages = 1."""
    section = make_section(elements=("Nd",), n_stages=1)
    feed, solvent = streams(section, {"Nd": 0.01}, acid=0.01)
    feed_totals, solvent_totals = section.component_totals(feed, solvent)
    sol = solve_stage(
        section.network, feed_totals, solvent_totals, 1.0, 1.0,
        section.correlation_D(),
    )
    assert sol.u.shape == (1, section.network.n_components)
    assert bool(sol.feasible)


# =============================================================================
# The degrees-of-freedom bridge between the two levels
# =============================================================================

@pytest.mark.slow
def test_base_addition_for_ph_hits_the_target():
    """The explicit inverse problem: pH specified, base rate solved for."""
    section = make_section(elements=("Nd", "Dy"), n_stages=4,
                           calibration_pH=3.0)
    feed, solvent = streams(section, {"Nd": 0.02, "Dy": 0.02}, acid=0.02)
    _, _, before = section(feed, solvent)
    target = 2.5
    assert float(before["pH"]) < target       # unreachable without base

    b, ok = base_addition_for_pH(section, feed, solvent, target)
    assert bool(ok)
    assert float(b) > 0.0
    _, _, after = section(feed, solvent, base_addition=b)
    assert float(after["pH"]) == pytest.approx(target, abs=1e-8)
    # Raising the pH raises the extraction, which is the point of dosing.
    assert float(after["theta"][0]) > float(before["theta"][0])


@pytest.mark.slow
def test_base_addition_for_ph_is_differentiable():
    """d(base rate)/d(specified pH) falls out of the augmented solve."""
    section = make_section(elements=("Nd", "Dy"), n_stages=3,
                           calibration_pH=3.0)
    feed, solvent = streams(section, {"Nd": 0.02, "Dy": 0.02}, acid=0.02)

    def dosing(target):
        return base_addition_for_pH(section, feed, solvent, target)[0]

    analytic = float(jax.grad(dosing)(2.5))
    h = 2e-3
    fd = float((dosing(2.5 + h) - dosing(2.5 - h)) / (2 * h))
    assert analytic == pytest.approx(fd, rel=1e-3)
    assert analytic > 0.0     # a higher pH needs more base


@pytest.mark.slow
def test_unreachable_ph_target_is_a_soft_failure():
    """Outside the dosing bounds there is no realizable answer, and no raise.

    Both ends are tested, because they are unreachable for different reasons.
    Below: the feed carries no counter-ion, so there is no base to *remove*
    and the un-dosed pH is the floor. Above: the section runs out of protons
    to neutralize once essentially all the rare earth has been extracted, and
    :func:`base_addition_bounds` is where that happens.
    """
    section = make_section(elements=("Nd",), n_stages=2)
    feed, solvent = streams(section, {"Nd": 0.02}, acid=0.02)
    b_lo, b_hi = base_addition_bounds(section, feed, solvent)
    assert float(b_hi) > float(b_lo)

    _, _, undosed = section(feed, solvent)
    _, _, at_bound = section(feed, solvent, base_addition=b_hi)
    floor_pH = float(undosed["pH"])
    ceiling_pH = float(at_bound["pH"])
    assert ceiling_pH > floor_pH

    b, ok = base_addition_for_pH(section, feed, solvent, floor_pH - 1.0)
    assert not bool(ok)
    assert float(b) == pytest.approx(float(b_lo), abs=1e-12)

    b, ok = base_addition_for_pH(section, feed, solvent, ceiling_pH + 1.0)
    assert not bool(ok)
    assert float(b) == pytest.approx(float(b_hi), rel=1e-9)

    # And something in between is reached, so the bounds are not just refusing
    # everything.
    _, ok = base_addition_for_pH(
        section, feed, solvent, 0.5 * (floor_pH + ceiling_pH)
    )
    assert bool(ok)


# =============================================================================
# The shared interface, and the two things it does not hide
# =============================================================================

def test_extractor_dispatches_to_either_level():
    """Cascade code calls one class and gets (raffinate, extract, info)."""
    assert set(EXTRACTOR_MODELS) == {"correlation", "mass_action"}
    schema = REEStreamSchema(elements=("Nd", "Dy"), extractant="D2EHPA")
    feed = schema.make_aqueous({"Nd": 0.02, "Dy": 0.02}, acid=0.01, water=55.0)
    solvent = schema.make_organic(0.5, diluent_flow=4.0)

    params = REEExtractorParams(
        n_stages=4, extractant="D2EHPA", elements=("Nd", "Dy"), pH=3.0,
    )
    r1, e1, i1 = REEExtractor(params)(feed, solvent)
    closed = REEExtractor(params.update(
        model="mass_action",
        aqueous_volumetric_flow=1.0,
        organic_volumetric_flow=1.0,
    ))
    r2, e2, i2 = closed(feed, solvent)

    for stream in (r1, e1, r2, e2):
        assert "T" in stream and "P" in stream
    assert "pH_profile" in i2 and "pH_profile" not in i1
    assert closed.section is not None
    assert REEExtractor(params).section is None


def test_closed_model_refuses_a_specified_ph():
    """The DOF difference is refused, not silently absorbed (#196)."""
    schema = REEStreamSchema(elements=("Nd",), extractant="D2EHPA")
    feed = schema.make_aqueous({"Nd": 0.02}, acid=0.01, water=55.0)
    solvent = schema.make_organic(0.5, diluent_flow=4.0)
    extractor = REEExtractor(REEExtractorParams(
        n_stages=2, extractant="D2EHPA", elements=("Nd",), pH=3.0,
        model="mass_action", aqueous_volumetric_flow=1.0,
        organic_volumetric_flow=1.0,
    ))
    with pytest.raises(ValueError, match="pH is an OUTPUT"):
        extractor(feed, solvent, pH=3.0)


def test_correlation_level_refuses_base_addition():
    """And symmetrically: no acid balance, so nothing to dose into."""
    schema = REEStreamSchema(elements=("Nd",), extractant="D2EHPA")
    feed = schema.make_aqueous({"Nd": 0.02}, acid=0.01, water=55.0)
    solvent = schema.make_organic(0.5, diluent_flow=4.0)
    extractor = REEExtractor(REEExtractorParams(
        n_stages=2, extractant="D2EHPA", elements=("Nd",), pH=3.0,
    ))
    with pytest.raises(ValueError, match="base_addition"):
        extractor(feed, solvent, base_addition=0.01)


def test_closed_model_requires_phase_volumes():
    """It works in concentrations; a flow ratio is not enough (#196)."""
    with pytest.raises(ValueError, match="volumetric_flow"):
        REEExtractorParams(
            n_stages=2, extractant="D2EHPA", elements=("Nd",),
            model="mass_action",
        )


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="model must be one of"):
        REEExtractorParams(
            n_stages=2, extractant="D2EHPA", elements=("Nd",), model="magic",
        )


# =============================================================================
# The stream schema superset
# =============================================================================

def test_schema_is_the_superset_of_both_levels():
    """State width differs between levels; the vocabulary does not (#196)."""
    schema = REEStreamSchema(elements=("Nd", "Dy"), extractant="D2EHPA")
    keys = set(schema.all_keys())
    # The issue's list, in full.
    assert {"Nd", "Dy"} <= keys                 # rare earths by element
    assert schema.acid in keys                  # H
    assert schema.counter_ion in keys           # Na / NH4
    assert schema.anion in keys                 # Cl / NO3 / SO4
    assert schema.water in keys                 # water
    assert schema.extractant in keys            # extractant total
    assert schema.organic_acid in keys          # co-extracted acid
    assert schema.organic_water in keys         # water in organic
    assert schema.phase_of("Nd") == "aqueous"
    assert schema.phase_of("D2EHPA") == "organic"


def test_schema_closes_the_anion_by_electroneutrality():
    """A feed that cannot exist is not silently accepted."""
    schema = REEStreamSchema(elements=("Nd",), extractant="D2EHPA")
    feed = schema.make_aqueous({"Nd": 0.1}, acid=0.05, counter_ion=0.02,
                               water=55.0)
    assert float(feed["F_Cl"]) == pytest.approx(3 * 0.1 + 0.05 + 0.02)


def test_schema_rejects_unknown_ions():
    with pytest.raises(ValueError, match="Unknown anion"):
        REEStreamSchema(elements=("Nd",), extractant="D2EHPA", anion="ClO4")
    with pytest.raises(ValueError, match="Unknown counter_ion"):
        REEStreamSchema(elements=("Nd",), extractant="D2EHPA",
                        counter_ion="Cs")
    assert set(ANION_CHARGES) == {"Cl", "NO3", "SO4"}
    assert set(COUNTER_ION_CHARGES) == {"Na", "NH4", "K"}


def test_loaded_solvent_brings_a_negative_proton_component():
    """The tableau bookkeeping that makes a recycled solvent behave."""
    section = make_section(elements=("Nd",), n_stages=2)
    net = section.network
    solvent = section.schema.make_organic(
        0.5, diluent_flow=4.0, element_flows={"Nd": 0.01}
    )
    totals = organic_component_totals(net, section.schema, solvent)
    assert float(totals[net.proton_index]) == pytest.approx(-3 * 0.01)
    assert float(totals[net.extractant_index]) == pytest.approx(0.25)


# =============================================================================
# The generic residual entry point added to eo_solver (#196)
# =============================================================================

def test_solve_residual_system_reports_failure_softly():
    """The section-scope entry point returns a flag; it does not raise."""
    def residual(z, args):
        return z ** 2 + 1.0        # no real root

    z, norm, feasible = solve_residual_system(
        residual, jnp.array([1.0]), None, max_steps=5
    )
    assert not bool(feasible)
    assert z.shape == (1,)

    def solvable(z, args):
        return z ** 2 - args

    z, norm, feasible = solve_residual_system(solvable, jnp.array([1.0]), 2.0)
    assert bool(feasible)
    assert float(z[0]) == pytest.approx(np.sqrt(2.0), rel=1e-12)
