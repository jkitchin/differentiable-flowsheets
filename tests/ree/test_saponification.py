"""Saponified extractants and the counter-ion balance (#197).

These tests are written to *discriminate*, not to cover. Each one is chosen so
that it fails if the corresponding piece of physics is removed:

- the counter-ion is conserved to machine precision across a section, so a
  saponified solvent cannot quietly leak sodium;
- a saponified cascade holds a **markedly flatter pH profile** than the same
  reagent dosed into the aqueous feed, and the advantage *grows* with the
  stage count -- the issue's central claim, quantified rather than described;
- ``S = 0`` reproduces the unsaponified proton-exchange result bit for bit,
  so the new network is a strict generalization;
- three equivalents of base per mole of rare earth moved, checked twice: once
  read off the tableau through the extractant column, and once as the exact
  identity ``dT_H(aq) + z dT_M(aq) = 3 dRE`` that the solved section satisfies
  to round-off;
- kilograms of base per kilogram of REO, cross-checked against a hand
  calculation written out in the test;
- the organic actually buffers: perturbing the aqueous acid is absorbed by the
  organic converting ``M(HA2)`` back to ``(HA)2``, which an unsaponified
  network cannot do at all;
- the :class:`Saponifier` conserves base and sets the declared degree;
- ``jit``, ``grad`` and ``check_grads`` all work through the whole thing.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad, jit
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from difflow.streams import get_flows
from difflow_ree.database import (
    SAPONIFICATION_COUNTER_IONS,
    create_custom_extractant,
    get_extractant,
)
from difflow_ree.economics.saponification import (
    BASE_REAGENTS,
    base_for_counter_ion,
    base_per_ree_oxide,
    compare_counter_ions,
    dissolved_salt_per_ree_oxide,
    nitrogen_per_ree_oxide,
    ree_oxide_mass_flow,
    saponification_duty,
    stoichiometric_base_per_ree_oxide,
)
from difflow_ree.equilibrium.mass_action import MassActionParams, MassActionSection
from difflow_ree.equilibrium.network import build_network, get_network_template
from difflow_ree.equilibrium.saponification import (
    INDUSTRIAL_DEGREE_RANGE,
    SaponifiedParams,
    SaponifiedSection,
    divalent_counter_ion_template,
    organic_buffer_capacity,
    organic_buffer_pH,
    ph_profile_flatness,
    saponification_degree_for_pH,
    saponification_degree_profile,
    saponification_log_K,
)
from difflow_ree.units.saponification import Saponifier, SaponifierParams


# =============================================================================
# Helpers
# =============================================================================

BASE_SECTION = dict(
    extractant="D2EHPA",
    elements=("Nd", "Dy"),
    aqueous_volumetric_flow=1.0,
    organic_volumetric_flow=1.0,
    extractant_conc=0.5,
    calibration_pH=3.0,
)

#: Extractant on a monomer basis, and the dimer basis it converts to.
F_EXT = 0.5
MONOMERS_PER_COMPONENT = 2.0


def saponified_section(n_stages=4, degree=0.35, **kwargs):
    """A saponified section on the standard test configuration."""
    return SaponifiedSection(SaponifiedParams(
        n_stages=n_stages, saponification_degree=degree,
        **{**BASE_SECTION, **kwargs},
    ))


def plain_section(n_stages=4, base_addition=0.0, **kwargs):
    """The unsaponified reference section, optionally dosed with base."""
    return MassActionSection(MassActionParams(
        n_stages=n_stages, base_addition=base_addition,
        **{**BASE_SECTION, **kwargs},
    ))


def make_feed(section, acid=0.005, nd=0.02, dy=0.02):
    """Standard aqueous feed on a section's own schema."""
    return section.schema.make_aqueous(
        {"Nd": nd, "Dy": dy}, acid=acid, water=55.0
    )


def make_solvent(section, degree=None, f_ext=F_EXT):
    """Fresh solvent, saponified to ``degree`` when one is given."""
    if degree is None:
        return section.schema.make_organic(f_ext, diluent_flow=4.0)
    return section.schema.saponified_organic(
        f_ext, degree,
        monomers_per_component=MONOMERS_PER_COMPONENT,
        diluent_flow=4.0,
    )


def base_equivalents(degree, f_ext=F_EXT):
    """Base equivalents in a solvent saponified to ``degree``."""
    return degree * f_ext / MONOMERS_PER_COMPONENT


# =============================================================================
# 1. The reaction network: one added species row, and only that
# =============================================================================

def test_saponification_is_exactly_one_added_species_row():
    """#196 promised this would be one row; the table says it is (#197)."""
    plain = get_network_template("cation_exchange_dimer")
    sap = get_network_template("cation_exchange_dimer_saponified")

    # Same component basis, verbatim: the counter-ion was already conserved.
    assert [dataclasses.astuple(c) for c in plain.components] == [
        dataclasses.astuple(c) for c in sap.components
    ]
    # One added species, with the stoichiometry #196's YAML comment wrote out.
    assert len(sap.species) == len(plain.species) + 1
    salt = sap.species[-1]
    assert salt.name == "M(HA2)"
    assert salt.phase == "organic"
    assert salt.charge == 0
    assert dict(salt.stoichiometry) == {"M+": 1, "(HA)2": 1, "H+": -1}

    assert sap.is_saponified
    assert not plain.is_saponified

    # And it is charge consistent, which build_network checks for real.
    net = build_network(sap, ("Nd",), log10_K={"Nd": -7.45})
    assert np.allclose(net.nu @ net.component_charges, net.species_charges)
    assert net.counter_ion_species_index == (1,)


def test_the_unsaponified_network_is_untouched():
    """A spectator sodium salt must not start saponifying the organic."""
    net = build_network("cation_exchange_dimer", ("Nd",), log10_K={"Nd": -7.45})
    assert net.species_names == ("Nd(HA2)3",)
    assert not net.is_saponified
    assert net.counter_ion_species_index == ()
    assert net.base_equivalents_per_mole_ree == 0.0


# =============================================================================
# 2. The counter-ion is conserved
# =============================================================================

@pytest.mark.parametrize("counter_ion", ["Na", "NH4", "Mg"])
def test_counter_ion_conserved_to_machine_precision(counter_ion):
    """Total counter-ion in equals total out, aqueous plus organic (#197)."""
    section = saponified_section(counter_ion=counter_ion)
    feed = section.schema.make_aqueous(
        {"Nd": 0.02, "Dy": 0.02}, acid=0.005, counter_ion=0.01, water=55.0
    )
    solvent = make_solvent(section, degree=0.35)
    raffinate, extract, info = section(feed, solvent)
    assert bool(info["feasible"])

    key = section.schema.counter_ion
    org_key = section.schema.organic_counter_ion
    total_in = float(feed[f"F_{key}"]) + float(solvent[f"F_{org_key}"])
    total_out = float(raffinate[f"F_{key}"]) + float(extract[f"F_{org_key}"])
    assert total_in > 0.0
    assert total_out == pytest.approx(total_in, rel=1e-14, abs=1e-16)

    # The degree the section reads back off the stream is the degree that was
    # asked for -- which for a divalent counter-ion means the charge was
    # accounted for, since it carries two equivalents per ion.
    assert float(info["saponification_degree_in"]) == pytest.approx(0.35)
    assert float(solvent[f"F_{org_key}"]) == pytest.approx(
        0.35 * F_EXT / MONOMERS_PER_COMPONENT
        / section.schema.counter_ion_charge
    )

    # It genuinely partitioned: neither end holds all of it.
    assert 0.0 < float(extract[f"F_{org_key}"]) < total_in


def test_counter_ion_partitions_rather_than_being_a_spectator():
    """Sodium is in the organic phase now, which it never was before (#197)."""
    section = saponified_section()
    feed = make_feed(section)
    solvent = make_solvent(section, degree=0.35)
    _, _, info = section(feed, solvent)
    c = info["solution"].concentrations(section.network)
    assert float(c["M(HA2)"][0]) > 0.0
    # Free extractant is below the total, because some of it is the salt.
    assert float(c["(HA)2"][0]) < F_EXT / MONOMERS_PER_COMPONENT


# =============================================================================
# 3. The central claim: a flatter pH profile
# =============================================================================

def test_saponified_cascade_holds_a_markedly_flatter_ph_profile():
    """#197's central claim, with the same reagent placed two ways.

    The comparison is fair by construction: both sections get the *same*
    number of base equivalents on the same feed with the same solvent
    inventory. The only difference is where the base is -- dosed into the
    aqueous feed at the raffinate end (the "dose base into every mixer"
    practice the issue describes) or pre-neutralized onto the organic. The
    unsaponified section's pH then collapses down the section as it releases
    three protons per trivalent ion, while the saponified one's does not,
    because the organic is a buffer that spans every stage.
    """
    degree = 0.35
    eq = base_equivalents(degree)

    dosed = plain_section(n_stages=8, base_addition=eq)
    r1, e1, i1 = dosed(make_feed(dosed), make_solvent(dosed))
    sap = saponified_section(n_stages=8, degree=degree)
    r2, e2, i2 = sap(make_feed(sap), make_solvent(sap, degree=degree))

    assert bool(i1["feasible"]) and bool(i2["feasible"])
    span_dosed = float(ph_profile_flatness(i1["pH_profile"]))
    span_sap = float(ph_profile_flatness(i2["pH_profile"]))

    # Measured: 0.77 pH units of collapse against 0.26, a factor of 2.9.
    assert span_dosed > 0.6
    assert span_sap < 0.35
    assert span_dosed / span_sap > 2.0

    # And it is not an artefact of extracting less: normalize the excursion by
    # the rare earth actually moved, which is what releases the protons.
    moved_dosed = float(e1["F_Nd"] + e1["F_Dy"])
    moved_sap = float(e2["F_Nd"] + e2["F_Dy"])
    assert (span_dosed / moved_dosed) / (span_sap / moved_sap) > 1.8


def test_the_flatness_advantage_grows_with_the_stage_count():
    """The buffer is what lets a LONG cascade hold a flat profile (#197)."""
    degree = 0.35
    eq = base_equivalents(degree)
    ratios = []
    for n_stages in (4, 12):
        dosed = plain_section(n_stages=n_stages, base_addition=eq)
        _, _, i1 = dosed(make_feed(dosed), make_solvent(dosed))
        sap = saponified_section(n_stages=n_stages, degree=degree)
        _, _, i2 = sap(make_feed(sap), make_solvent(sap, degree=degree))
        assert bool(i1["feasible"]) and bool(i2["feasible"])
        ratios.append(
            float(ph_profile_flatness(i1["pH_profile"]))
            / float(ph_profile_flatness(i2["pH_profile"]))
        )
    assert ratios[1] > ratios[0]
    assert ratios[0] > 2.0


def test_an_unsaponified_section_underpredicts_loading():
    """The failure mode the issue names: plausible, closed, and wrong (#197).

    With no base anywhere the unsaponified model is the only thing difflow_ree
    could express before #197, and on this feed it says the extraction barely
    happens -- which is what "under-predicts loading, over-predicts the stage
    count" means in a number.
    """
    plain = plain_section(n_stages=8)
    _, e1, i1 = plain(make_feed(plain), make_solvent(plain))
    sap = saponified_section(n_stages=8, degree=0.35)
    _, e2, i2 = sap(make_feed(sap), make_solvent(sap, degree=0.35))
    assert bool(i1["feasible"]) and bool(i2["feasible"])
    assert float(e2["F_Nd"]) > 20.0 * float(e1["F_Nd"])


# =============================================================================
# 4. S = 0 is the unsaponified model, exactly
# =============================================================================

def test_degree_zero_reproduces_proton_exchange_exactly():
    """The saponified network is a strict generalization (#197).

    With no counter-ion anywhere the salt species has zero concentration and
    the extra tableau row is inert, so the answer must be the unsaponified
    one -- not close to it, the same one.
    """
    plain = plain_section(n_stages=4)
    sap = saponified_section(n_stages=4, degree=0.0)
    r1, e1, i1 = plain(make_feed(plain), make_solvent(plain))
    r2, e2, i2 = sap(make_feed(sap), make_solvent(sap, degree=0.0))

    assert bool(i1["feasible"]) and bool(i2["feasible"])
    assert np.allclose(
        np.asarray(i1["pH_profile"]), np.asarray(i2["pH_profile"]),
        rtol=0.0, atol=1e-12,
    )
    for key in ("F_Nd", "F_Dy", "F_H"):
        assert float(r2[key]) == pytest.approx(float(r1[key]), rel=1e-12)
    for key in ("F_Nd", "F_Dy"):
        assert float(e2[key]) == pytest.approx(float(e1[key]), rel=1e-12)
    # And the degree the section reports is zero, not merely small.
    assert float(i2["saponification_degree"]) == pytest.approx(0.0, abs=1e-20)


# =============================================================================
# 5. Three equivalents of base per mole of rare earth
# =============================================================================

@pytest.mark.parametrize(
    "network,expected",
    [
        ("cation_exchange_dimer_saponified", 3.0),
        ("cation_exchange_monomer_saponified", 3.0),
    ],
)
def test_base_equivalents_per_mole_ree_read_off_the_tableau(network, expected):
    """Three equivalents per mole, derived rather than written down (#197)."""
    net = build_network(network, ("Nd",), log10_K={"Nd": -7.45})
    assert net.base_equivalents_per_mole_ree == pytest.approx(expected)
    # The property routes through the extractant column, so agreeing with the
    # proton column is an independent check and not a restatement.
    protons = abs(float(net.nu[net.element_species_index[0], net.proton_index]))
    assert net.base_equivalents_per_mole_ree == pytest.approx(protons)


def test_divalent_counter_ion_still_needs_three_equivalents():
    """Half as many formula units of salt, twice the equivalents each (#197)."""
    template = divalent_counter_ion_template()
    net = build_network(template, ("Nd",), log10_K={"Nd": -7.45})
    salt = net.species[net.counter_ion_species_index[0]]
    assert salt.name == "M(HA2)2"
    assert dict(salt.stoichiometry) == {"M2+": 1, "(HA)2": 2, "H+": -2}
    assert net.components[net.counter_ion_index].charge == 2
    assert net.base_equivalents_per_mole_ree == pytest.approx(3.0)


@pytest.mark.parametrize("counter_ion", ["Na", "Mg"])
def test_three_equivalents_delivered_to_the_aqueous_phase(counter_ion):
    """The exact identity the tableau forces on any solved section (#197).

    Summing the proton and counter-ion component balances over the section,
    with an unloaded solvent entering,

        dT_H(aqueous) + z * dT_M(aqueous) = 3 * (rare earth extracted)

    because every extracted trivalent ion occupies three extractant
    equivalents and each one gives back either a proton or a counter-ion.
    That is the "three equivalents of base per mole of rare earth moved" of
    the issue, and it holds to round-off rather than approximately.
    """
    section = saponified_section(counter_ion=counter_ion)
    feed = make_feed(section)
    solvent = make_solvent(section, degree=0.4)
    raffinate, extract, info = section(feed, solvent)
    assert bool(info["feasible"])

    key = section.schema.counter_ion
    z = section.schema.counter_ion_charge
    d_H = float(raffinate["F_H"]) - float(feed["F_H"])
    d_M = float(raffinate[f"F_{key}"]) - float(feed[f"F_{key}"])
    d_RE = float(extract["F_Nd"] + extract["F_Dy"])

    assert d_H + z * d_M == pytest.approx(3.0 * d_RE, rel=1e-10)
    # The saponified circuit pays for it in counter-ion, not in protons: the
    # counter-ion carries more than all of it, and the proton balance is
    # actually negative because the organic also neutralized the feed acid.
    assert z * d_M > 3.0 * d_RE * 0.9
    assert d_H < 0.0


def test_an_unsaponified_section_delivers_the_same_three_as_protons():
    """The same identity with the counter-ion term identically zero (#197)."""
    section = plain_section(n_stages=4)
    feed = make_feed(section)
    raffinate, extract, info = section(feed, make_solvent(section))
    assert bool(info["feasible"])
    d_H = float(raffinate["F_H"]) - float(feed["F_H"])
    d_M = float(raffinate["F_Na"]) - float(feed["F_Na"])
    d_RE = float(extract["F_Nd"] + extract["F_Dy"])
    assert d_M == pytest.approx(0.0, abs=1e-18)
    assert d_H == pytest.approx(3.0 * d_RE, rel=1e-10)


# =============================================================================
# 6. kg base per kg REO
# =============================================================================

def test_ree_oxide_mass_uses_the_right_oxide_formula():
    """REO is not always RE2O3: Ce is CeO2, Pr is Pr6O11, Tb is Tb4O7."""
    # Nd2O3 = 336.48 g/mol for two Nd, so 168.24 g per mol Nd.
    assert float(ree_oxide_mass_flow({"Nd": 1.0})) == pytest.approx(
        336.48 / 2 / 1000.0, rel=1e-12
    )
    # CeO2 = 172.12 g/mol for one Ce.
    assert float(ree_oxide_mass_flow({"Ce": 1.0})) == pytest.approx(
        172.12 / 1000.0, rel=1e-12
    )
    # Pr6O11 = 1021.44 g/mol for six Pr.
    assert float(ree_oxide_mass_flow({"Pr": 1.0})) == pytest.approx(
        1021.44 / 6 / 1000.0, rel=1e-12
    )


def test_kg_base_per_kg_reo_matches_the_hand_calculation():
    """The headline metric, checked against arithmetic written out here.

    Three equivalents of base per mole of Nd; Nd2O3 is 336.48 g/mol for two
    moles of Nd, so 168.24 g of REO per mole of Nd:

        NaOH     3 * 39.997 / 168.24     = 0.7132 kg/kg
        NH3      3 * 17.031 / 168.24     = 0.3037 kg/kg
        Mg(OH)2  1.5 * 58.320 / 168.24   = 0.5200 kg/kg   (2 eq per mole)
    """
    reo_per_mol_nd = 336.48 / 2 / 1000.0

    hand = {
        "NaOH": 3 * 39.997 / 1000.0 / reo_per_mol_nd,
        "NH3": 3 * 17.031 / 1000.0 / reo_per_mol_nd,
        "Mg(OH)2": 1.5 * 58.320 / 1000.0 / reo_per_mol_nd,
    }
    assert hand["NaOH"] == pytest.approx(0.7132, abs=5e-5)
    assert hand["NH3"] == pytest.approx(0.3037, abs=5e-5)
    assert hand["Mg(OH)2"] == pytest.approx(0.5200, abs=5e-5)

    for base, expected in hand.items():
        moles = 3.0 / BASE_REAGENTS[base].equivalents_per_mole
        assert float(
            base_per_ree_oxide(moles, {"Nd": 1.0}, base)
        ) == pytest.approx(expected, rel=1e-12)
        assert float(
            stoichiometric_base_per_ree_oxide({"Nd": 1.0}, base)
        ) == pytest.approx(expected, rel=1e-12)


def test_kg_base_per_kg_reo_from_a_solved_section():
    """The same balance that makes the cascade correct predicts the duty.

    The base consumed is the counter-ion the section released into the
    raffinate; nothing extra is computed to get it.
    """
    section = saponified_section(n_stages=6, degree=0.4)
    feed = make_feed(section)
    solvent = make_solvent(section, degree=0.4)
    _, extract, info = section(feed, solvent)
    assert bool(info["feasible"])

    reagent = base_for_counter_ion("Na")
    released = float(info["counter_ion_released"])
    assert released > 0.0
    base_moles = released / reagent.counter_ion_per_mole
    product = {"Nd": extract["F_Nd"], "Dy": extract["F_Dy"]}

    duty = saponification_duty(
        base_moles, product, reagent,
        equivalents_per_mole_ree=section.network.base_equivalents_per_mole_ree,
    )
    assert float(duty.kg_base_per_kg_reo) > 0.0
    # No circuit beats the three-equivalent floor, and this one does not.
    assert float(duty.reagent_efficiency) <= 1.0 + 1e-12
    assert float(duty.kg_base_per_kg_reo) >= float(
        duty.kg_base_per_kg_reo_stoichiometric
    ) - 1e-12
    # Sodium: a saline raffinate, no nitrogen.
    assert float(duty.kg_nitrogen_per_kg_reo) == 0.0
    assert float(duty.kg_salt_per_kg_reo) > float(duty.kg_base_per_kg_reo)
    assert "kg base / kg REO" in duty.report()


def test_counter_ion_choice_is_a_computable_environmental_trade_off():
    """Ammonium nitrogen against a saline raffinate (#197).

    Neither number existed before this feature, because there was no
    counter-ion anywhere in the extraction path.
    """
    duties = compare_counter_ions({"Nd": 1.0})
    assert set(duties) == {"Na", "NH4", "Mg"}

    # Ammonia is the cheapest base per kg of product...
    assert (
        float(duties["NH4"].kg_base_per_kg_reo)
        < float(duties["Na"].kg_base_per_kg_reo)
    )
    # ... and the only one with a nitrogen effluent, which is the point.
    assert float(duties["NH4"].kg_nitrogen_per_kg_reo) == pytest.approx(
        3 * 14.007 / 1000.0 / (336.48 / 2 / 1000.0), rel=1e-12
    )
    assert float(duties["Na"].kg_nitrogen_per_kg_reo) == 0.0
    assert float(duties["Mg"].kg_nitrogen_per_kg_reo) == 0.0

    # Sodium trades it for salt: NaCl at 58.443 g/mol, three per mole of Nd.
    assert float(duties["Na"].kg_salt_per_kg_reo) == pytest.approx(
        3 * (22.990 + 35.453) / 1000.0 / (336.48 / 2 / 1000.0), rel=1e-12
    )
    assert float(
        nitrogen_per_ree_oxide(3.0, {"Nd": 1.0}, "NaOH")
    ) == 0.0
    assert float(
        dissolved_salt_per_ree_oxide(3.0, {"Nd": 1.0}, "NaOH", anion="NO3")
    ) > float(duties["Na"].kg_salt_per_kg_reo)


# =============================================================================
# 7. The organic is a buffer
# =============================================================================

def test_the_organic_absorbs_an_aqueous_acid_perturbation():
    """The buffering mechanism, shown directly (#197).

    Adding acid to the feed is taken up by the organic converting ``M(HA2)``
    back to ``(HA)2`` and releasing its counter-ion. An unsaponified network
    cannot do that at all -- it has no conjugate base -- so its counter-ion
    release stays identically zero and the acid lands entirely on the pH.
    """
    delta = 0.005
    degree = 0.35

    sap = saponified_section(n_stages=4, degree=degree)
    solvent = make_solvent(sap, degree=degree)
    lo = sap(make_feed(sap, acid=0.005), solvent)[2]
    hi = sap(make_feed(sap, acid=0.005 + delta), solvent)[2]
    assert bool(lo["feasible"]) and bool(hi["feasible"])

    uptake = float(hi["counter_ion_released"]) - float(lo["counter_ion_released"])
    # More than a third of the added acid is absorbed by the organic phase,
    # released as counter-ion rather than showing up as free protons.
    assert uptake / delta > 0.35
    d_pH_sap = abs(float(hi["pH_profile"][-1]) - float(lo["pH_profile"][-1]))

    dosed = plain_section(n_stages=4, base_addition=base_equivalents(degree))
    solvent_p = make_solvent(dosed)
    lo_p = dosed(make_feed(dosed, acid=0.005), solvent_p)[2]
    hi_p = dosed(make_feed(dosed, acid=0.005 + delta), solvent_p)[2]
    assert bool(lo_p["feasible"]) and bool(hi_p["feasible"])
    d_pH_plain = abs(
        float(hi_p["pH_profile"][-1]) - float(lo_p["pH_profile"][-1])
    )

    # The buffered section moves less, on the same perturbation.
    assert d_pH_sap < d_pH_plain / 1.5


def test_henderson_hasselbalch_round_trip_and_capacity():
    """The organic acid-base pair, stated and inverted (#197)."""
    log_K = saponification_log_K(0.35, pH=3.0, counter_ion_conc=0.1)
    assert log_K == pytest.approx(np.log10(0.35 / 0.65 * 1e-3 / 1e-1), rel=1e-12)
    # The YAML default is this arithmetic, rounded.
    yaml_K = get_network_template(
        "cation_exchange_dimer_saponified"
    ).species[-1].log10_K
    assert yaml_K == pytest.approx(log_K, abs=5e-5)

    assert float(organic_buffer_pH(log_K, 0.35, 0.1)) == pytest.approx(3.0)
    # Ten times the counter-ion is one pH unit lower, as the relation says.
    assert float(organic_buffer_pH(log_K, 0.35, 1.0)) == pytest.approx(2.0)

    # Capacity is maximal at half neutralization and vanishes at either end.
    beta = [float(organic_buffer_capacity(0.25, S))
            for S in (0.01, 0.25, 0.5, 0.75, 0.99)]
    assert beta[2] == max(beta)
    assert beta[2] == pytest.approx(np.log(10.0) * 0.25 * 0.25, rel=1e-12)
    assert beta[0] < 0.05 * beta[2]

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            saponification_log_K(bad)


def test_the_degree_is_an_output_of_the_section_not_a_parameter():
    """The organic re-equilibrates; the inlet degree is only an inlet (#197)."""
    section = saponified_section(n_stages=6, degree=0.4)
    feed = make_feed(section)
    solvent = make_solvent(section, degree=0.4)
    _, _, info = section(feed, solvent)
    profile = np.asarray(info["saponification_degree_profile"])

    assert float(info["saponification_degree_in"]) == pytest.approx(0.4)
    # Spent hardest where the rare earth is richest: the feed end.
    assert profile[0] < profile[-1]
    assert profile[0] < 0.4
    assert np.all(profile >= 0.0) and np.all(profile <= 1.0)
    assert np.allclose(
        profile,
        np.asarray(
            saponification_degree_profile(section.network, info["solution"])
        ),
    )

    # It is a *fraction of the extractant*, recomputed here from the raw
    # concentrations rather than trusted: salt over (free + salt + three
    # dimers per loaded rare earth).
    c = info["solution"].concentrations(section.network)
    salt = np.asarray(c["M(HA2)"])
    total = (
        np.asarray(c["(HA)2"]) + salt
        + 3.0 * (np.asarray(c["Nd(HA2)3"]) + np.asarray(c["Dy(HA2)3"]))
    )
    assert np.allclose(profile, salt / total, rtol=1e-12)
    # And that denominator is the conserved extractant total, per stage.
    assert np.allclose(
        total, F_EXT / MONOMERS_PER_COMPONENT
        / float(section.params.organic_volumetric_flow), rtol=1e-8,
    )


# =============================================================================
# 8. The Saponifier unit
# =============================================================================

def test_saponifier_sets_the_degree_and_conserves_the_base():
    """The reagent duty is tracked, not assumed (#197)."""
    params = SaponifierParams(extractant="D2EHPA", saponification_degree=0.35)
    unit = Saponifier(params)
    organic = unit.schema.make_organic(F_EXT, diluent_flow=4.0)
    out, spent, info = unit(organic)

    assert float(info["saponification_degree"]) == pytest.approx(0.35)
    # Equivalents = S * F_ext / monomers_per_component = 0.35 * 0.5 / 2.
    assert float(info["base_equivalents_transferred"]) == pytest.approx(0.0875)
    assert float(info["base_flow"]) == pytest.approx(0.0875)
    assert float(info["base_mass_flow"]) == pytest.approx(
        0.0875 * 39.997 / 1000.0
    )
    assert float(out["F_Na_org"]) == pytest.approx(0.0875)

    # Counter-ion in equals counter-ion out, to round-off.
    assert float(info["counter_ion_imbalance"]) == pytest.approx(0.0, abs=1e-18)
    assert float(spent["F_Na"]) == pytest.approx(0.0, abs=1e-18)
    # Water is produced one for one with the neutralized protons.
    assert float(info["water_produced"]) == pytest.approx(0.0875)
    # The extractant itself is untouched.
    assert float(out["F_D2EHPA"]) == pytest.approx(F_EXT)


def test_saponifier_conserves_base_that_does_not_reach_the_organic():
    """Utilization below one is paid for and discharged, not lost (#197)."""
    unit = Saponifier(SaponifierParams(
        extractant="D2EHPA", saponification_degree=0.35, base_utilization=0.8,
    ))
    organic = unit.schema.make_organic(F_EXT, diluent_flow=4.0)
    out, spent, info = unit(organic)

    assert float(info["saponification_degree"]) == pytest.approx(0.35)
    # 25% more reagent for the same degree.
    assert float(info["base_flow"]) == pytest.approx(0.0875 / 0.8)
    assert float(info["counter_ion_imbalance"]) == pytest.approx(0.0, abs=1e-18)
    # The shortfall leaves in the spent aqueous phase, electroneutral.
    assert float(spent["F_Na"]) == pytest.approx(0.0875 / 0.8 - 0.0875)
    assert float(spent["F_OH"]) == pytest.approx(float(spent["F_Na"]))


def test_saponifier_credits_a_partially_saponified_recycle():
    """A solvent still carrying counter-ion needs less fresh base (#197)."""
    unit = Saponifier(SaponifierParams(
        extractant="D2EHPA", saponification_degree=0.35,
    ))
    organic = unit.schema.make_organic(
        F_EXT, diluent_flow=4.0, counter_ion=0.03
    )
    out, _, info = unit(organic)
    assert float(info["saponification_degree_in"]) == pytest.approx(0.03 / 0.25)
    assert float(info["base_flow"]) == pytest.approx(0.0875 - 0.03)
    assert float(info["saponification_degree"]) == pytest.approx(0.35)
    assert float(out["F_Na_org"]) == pytest.approx(0.0875)


def test_saponifier_cannot_neutralize_past_the_extractant_inventory():
    """Overdosing raises the duty and the effluent, not the degree (#197)."""
    unit = Saponifier(SaponifierParams(
        extractant="D2EHPA", saponification_degree=1.0,
    ))
    organic = unit.schema.make_organic(F_EXT, diluent_flow=4.0)
    out, spent, info = unit(organic, base_flow=0.5)
    assert float(info["saponification_degree"]) == pytest.approx(1.0)
    assert float(info["base_equivalents_transferred"]) == pytest.approx(0.25)
    assert float(spent["F_Na"]) == pytest.approx(0.5 - 0.25)
    assert float(info["counter_ion_imbalance"]) == pytest.approx(0.0, abs=1e-18)


@pytest.mark.parametrize(
    "counter_ion,base,eq_per_mole",
    [("Na", "NaOH", 1.0), ("NH4", "NH3", 1.0), ("Mg", "Mg(OH)2", 2.0)],
)
def test_saponifier_picks_the_right_reagent(counter_ion, base, eq_per_mole):
    """Which base is dosed decides the effluent, so it is data (#197)."""
    unit = Saponifier(SaponifierParams(
        extractant="D2EHPA", saponification_degree=0.35,
        counter_ion=counter_ion,
    ))
    assert unit.base.name == base
    organic = unit.schema.make_organic(F_EXT, diluent_flow=4.0)
    out, _, info = unit(organic)
    assert float(info["base_flow"]) == pytest.approx(0.0875 / eq_per_mole)
    # The organic key carries moles of counter-ion, not of equivalents.
    assert float(out[f"F_{counter_ion}_org"]) == pytest.approx(
        0.0875 / unit.schema.counter_ion_charge
    )


def test_saponifier_and_section_compose_into_a_reagent_duty():
    """The whole point: the duty comes off the flowsheet, not a rule of thumb."""
    unit = Saponifier(SaponifierParams(
        extractant="D2EHPA", elements=("Nd", "Dy"),
        saponification_degree=0.4,
    ))
    section = saponified_section(n_stages=6, degree=0.4)
    organic = unit.schema.make_organic(F_EXT, diluent_flow=4.0)
    solvent, _, sap_info = unit(organic)

    feed = make_feed(section)
    _, extract, info = section(feed, solvent)
    assert bool(info["feasible"])
    # The section reads the degree straight off the stream the unit wrote.
    assert float(info["saponification_degree_in"]) == pytest.approx(0.4)

    duty = saponification_duty(
        sap_info["base_flow"],
        {"Nd": extract["F_Nd"], "Dy": extract["F_Dy"]},
        unit.base,
        equivalents_per_mole_ree=section.network.base_equivalents_per_mole_ree,
    )
    assert float(duty.kg_base_per_kg_reo) > 0.0
    assert float(duty.usd_per_kg_reo) > 0.0


def test_saponifier_rejects_impossible_configurations():
    with pytest.raises(ValueError, match="no acidic proton"):
        SaponifierParams(extractant="TBP")
    with pytest.raises(ValueError, match="cannot disagree"):
        SaponifierParams(extractant="D2EHPA", counter_ion="Na", base="NH3")
    with pytest.raises(ValueError, match="base_utilization"):
        SaponifierParams(extractant="D2EHPA", base_utilization=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SaponifierParams(extractant="D2EHPA", saponification_degree=1.5)


# =============================================================================
# 9. The extractant record (issue point 1)
# =============================================================================

def test_extractant_records_carry_the_saponification_fields():
    """A `saponification_degree` and a `counter_ion` on the record (#197)."""
    for name in ("D2EHPA", "PC88A", "Cyanex272"):
        ext = get_extractant(name)
        assert ext.counter_ion == "Na"
        assert INDUSTRIAL_DEGREE_RANGE[0] <= ext.saponification_degree \
            <= INDUSTRIAL_DEGREE_RANGE[1]
        assert ext.is_saponified
        # No measured constant ships; it is calibrated from the degree.
        assert ext.saponification_log10_K is None
        assert ext.saponification_reference_pH == 3.0

    # A neutral solvating extractant has nothing to neutralize.
    tbp = get_extractant("TBP")
    assert tbp.counter_ion is None
    assert tbp.saponification_degree == 0.0
    assert not tbp.is_saponified

    assert SAPONIFICATION_COUNTER_IONS == ("H", "Na", "NH4", "Mg")


def test_custom_extractants_can_declare_saponification():
    coeffs = {"Nd": {"a": -7.5, "b": 2.4, "c": 0.01}}
    ext = create_custom_extractant(
        name="MySapExtractant", full_name="x", formula="x",
        molecular_weight=300.0, ph_coefficients=coeffs,
        temperature_coefficients={"Nd": -1700},
        counter_ion="NH4", saponification_degree=0.45,
    )
    assert ext.counter_ion == "NH4"
    assert ext.saponification_degree == 0.45
    assert ext.is_saponified

    with pytest.raises(ValueError, match="between 0 and 1"):
        create_custom_extractant(
            name="Bad", full_name="x", formula="x", molecular_weight=300.0,
            ph_coefficients=coeffs, temperature_coefficients={"Nd": -1700},
            counter_ion="Na", saponification_degree=1.4,
        )
    with pytest.raises(ValueError, match="no counter_ion"):
        create_custom_extractant(
            name="Bad2", full_name="x", formula="x", molecular_weight=300.0,
            ph_coefficients=coeffs, temperature_coefficients={"Nd": -1700},
            saponification_degree=0.4,
        )
    with pytest.raises(ValueError, match="un-neutralized proton exchange"):
        create_custom_extractant(
            name="Bad3", full_name="x", formula="x", molecular_weight=300.0,
            ph_coefficients=coeffs, temperature_coefficients={"Nd": -1700},
            counter_ion="H", saponification_degree=0.4,
        )
    with pytest.raises(ValueError, match="nothing for a base to neutralize"):
        create_custom_extractant(
            name="Bad4", full_name="x", formula="x", molecular_weight=300.0,
            ph_coefficients=coeffs, temperature_coefficients={"Nd": -1700},
            counter_ion="Na", saponification_degree=0.4,
            stoichiometry_protons=0,
        )


def test_saponified_section_rejects_what_cannot_be_saponified():
    with pytest.raises(ValueError, match="cannot be saponified"):
        SaponifiedParams(
            n_stages=2, extractant="TBP", elements=("Nd",),
            aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
            anion="NO3",
        )
    with pytest.raises(ValueError, match="needs a counter_ion"):
        SaponifiedParams(
            n_stages=2, extractant="D2EHPA", elements=("Nd",),
            aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
            counter_ion=None,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SaponifiedParams(
            n_stages=2, extractant="D2EHPA", elements=("Nd",),
            aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
            saponification_degree=-0.1,
        )


# =============================================================================
# 10. The manipulated variable: differentiable, jittable, invertible
# =============================================================================

def test_saponification_degree_is_differentiable_and_jits():
    """The primary manipulated variable is a real handle (#197)."""
    section = saponified_section(n_stages=3, degree=0.35)
    feed = make_feed(section)

    def recovery(degree):
        solvent = make_solvent(section, degree=degree)
        _, extract, _ = section(feed, solvent)
        return extract["F_Nd"] / feed["F_Nd"]

    value = float(recovery(0.35))
    assert 0.0 < value < 1.0
    assert float(jit(recovery)(0.35)) == pytest.approx(value, rel=1e-10)

    g = float(grad(recovery)(0.35))
    fd = float((recovery(0.3502) - recovery(0.3498)) / 0.0004)
    assert g == pytest.approx(fd, rel=1e-4)
    # More base extracts more: the sign is the physics, not an accident.
    assert g > 0.0
    check_grads(recovery, (0.35,), order=1, modes=["rev"],
                atol=2e-3, rtol=2e-3)


def test_base_flow_is_differentiable_through_the_saponifier():
    """The reagent pump is the handle a control layer actually turns (#197)."""
    unit = Saponifier(SaponifierParams(
        extractant="D2EHPA", saponification_degree=0.35,
    ))
    organic = unit.schema.make_organic(F_EXT, diluent_flow=4.0)

    def degree(base_flow):
        _, _, info = unit(organic, base_flow=base_flow)
        return info["saponification_degree"]

    # dS/dn = 1 / (F_ext / m) = 1 / 0.25.
    assert float(grad(degree)(0.05)) == pytest.approx(4.0, rel=1e-12)
    assert float(jit(degree)(0.05)) == pytest.approx(0.2, rel=1e-12)
    check_grads(degree, (0.05,), order=2, modes=["rev"])


def test_saponification_degree_for_ph_inverts_the_section():
    """A pH specification maps onto the degree, not onto stage setpoints."""
    section = saponified_section(n_stages=3, degree=0.35)
    feed = make_feed(section)
    solvent = make_solvent(section, degree=0.35)
    base_pH = float(section(feed, solvent)[2]["pH_profile"][-1])

    target = base_pH + 0.12
    degree, ok = saponification_degree_for_pH(
        section, feed, solvent, target, n_bisection_steps=25
    )
    assert bool(ok)
    assert float(degree) > 0.35   # more base for a higher pH

    achieved = section(
        feed, make_solvent(section, degree=degree)
    )[2]["pH_profile"][-1]
    assert float(achieved) == pytest.approx(target, abs=1e-8)


# =============================================================================
# 11. Wiring
# =============================================================================

def test_public_api_is_reachable():
    import difflow_ree.economics as economics
    import difflow_ree.equilibrium as equilibrium
    import difflow_ree.units as units

    assert equilibrium.SaponifiedSection is SaponifiedSection
    assert equilibrium.saponification_log_K is saponification_log_K
    assert units.Saponifier is Saponifier
    assert economics.saponification_duty is saponification_duty


def test_the_schema_carries_the_organic_counter_ion():
    section = saponified_section()
    schema = section.schema
    assert schema.organic_counter_ion == "Na_org"
    assert schema.organic_counter_ion in schema.organic_keys()
    assert schema.phase_of("Na_org") == "organic"
    assert schema.phase_of("Na") == "aqueous"
    # An unsaponified organic stream is exactly what it always was.
    plain = schema.make_organic(F_EXT, diluent_flow=4.0)
    assert "Na_org" not in get_flows(plain)
