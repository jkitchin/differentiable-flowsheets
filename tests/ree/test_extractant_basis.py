"""One solvent, one extractant basis, at both modelling levels (#374).

The plugin's units, circuits and correlations state ``extractant_conc`` (and a
solvent stream's extractant flow) on the extractant record's own basis: DIMER
for D2EHPA, PC88A and Cyanex272, so ``extractant_conc = 0.5`` is 0.5 M dimer
= 1.0 M formal. #374 kept that basis. The mass-action layer
(``MassActionSection``, ``log_K_from_correlation``, the stream schema) states
it on the formal monomer basis. Two things were wrong at that boundary:

- inside the mass-action layer, the correlation that calibrates ``log10 K``
  (and starts the solve) was handed the monomer charge as if it were the
  record's basis, so D2EHPA's ``D_corr`` was taken at twice the dimer
  concentration the closure then ran at;
- ``REEExtractor(model="mass_action")`` handed its record-basis
  ``extractant_conc`` and solvent stream to that layer unconverted.

The two errors cancelled in ``D`` for a unit-level call and showed in the
loading: the closure saw half the extractant, so it reported twice the
loading fraction and half the free extractant of the same solvent.
"""

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.database import get_extractant
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.mass_action import MassActionParams, MassActionSection
from difflow_ree.equilibrium.schema import REEStreamSchema
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams


#: The agreement the closure and the correlation are stated to reach in the
#: dilute limit (tests/ree/test_mass_action.py, DILUTE_TOL and the
#: derivation in front of it): the residual is the pH shift of the protons
#: the trace extraction releases.
DILUTE_TOL = 2e-5
DILUTE_RE_TO_ACID = 1e-7

#: Records on each basis, and the monomers in one unit of it.
DIMERIC = ("D2EHPA", "PC88A", "Cyanex272")
MONOMERIC = ("TBP", "naphthenic_acid")


@pytest.mark.parametrize("name", DIMERIC + MONOMERIC)
def test_monomers_per_basis_unit_comes_from_the_declared_stoichiometry(name):
    ext = get_extractant(name)
    expected = 2.0 if name in DIMERIC else 1.0
    assert ext.monomers_per_basis_unit == expected
    assert ext.monomers_per_basis_unit == ext.monomers_per_ree / ext.basis_units_per_ree


@pytest.mark.parametrize("name", DIMERIC + ("naphthenic_acid",))
def test_mass_action_layer_evaluates_the_correlation_on_the_record_basis(name):
    """A section given 2C M formal reproduces the correlation at C M dimer.

    ``MassActionParams.extractant_conc`` is formal monomer. The correlation
    it is calibrated against must be evaluated at the same solvent on the
    record's basis, ``extractant_conc / monomers_per_basis_unit``; it used to
    be evaluated at ``extractant_conc`` itself (#374).
    """
    ext = get_extractant(name)
    m = ext.monomers_per_basis_unit
    elements = ("La", "Nd")
    pH = ext.reference_pH
    acid = 10.0 ** (-pH)
    formal = 0.5 * m              # 0.5 M on the record basis
    section = MassActionSection(MassActionParams(
        n_stages=1, extractant=name, elements=elements,
        aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
        extractant_conc=formal, calibration_pH=pH,
    ))
    feed = section.schema.make_aqueous(
        {el: acid * DILUTE_RE_TO_ACID for el in elements}, acid=acid, water=55.0)
    solvent = section.schema.make_organic(formal, diluent_flow=4.0)
    _, _, info = section(feed, solvent)
    assert bool(info["feasible"])

    dist = REEDistribution(name, elements, concentration=0.5)
    for el in elements:
        assert float(info["D"][el]) == pytest.approx(
            float(dist.get_D(el, pH)), rel=DILUTE_TOL)
    # The free extractant is on the component basis, the record's: 0.5 M.
    assert float(info["free_extractant"][0]) == pytest.approx(0.5, rel=1e-6)


def _same_solvent_both_levels(name, extractant_conc, re_to_acid):
    """Run one feed and one record-basis solvent through both levels."""
    ext = get_extractant(name)
    elements = ("La", "Nd", "Dy")
    pH = ext.reference_pH
    acid = 10.0 ** (-pH)
    schema = REEStreamSchema(elements=elements, extractant=name)
    feed = schema.make_aqueous(
        {el: acid * re_to_acid for el in elements}, acid=acid, water=55.0)
    F_aq = float(sum(get_flows(feed).values()))
    # A circuit's solvent: the diluent carries the organic volume (#373) and
    # the extractant entry is extractant_conc per unit of it, record basis.
    # Q_org = 1 L/s, so the closure reads extractant_conc mol/s as M.
    solvent = make_stream(
        {"kerosene": F_aq, name: extractant_conc,
         **{el: 0.0 for el in elements}}, 298.15, 101325.0)
    params = REEExtractorParams(
        n_stages=1, extractant=name, elements=elements, pH=pH,
        extractant_conc=extractant_conc, include_loading=False)
    corr = REEExtractor(params)(feed, solvent)
    closed = REEExtractor(params.update(
        model="mass_action", aqueous_volumetric_flow=1.0,
        organic_volumetric_flow=1.0))(feed, solvent)
    return ext, elements, pH, feed, solvent, corr, closed


@pytest.mark.parametrize("name", DIMERIC)
def test_same_solvent_gives_the_same_D_and_loading_at_both_levels(name):
    """The #374 boundary, through the one class cascade code calls."""
    conc = 0.5
    ext, elements, pH, feed, solvent, corr, closed = _same_solvent_both_levels(
        name, conc, DILUTE_RE_TO_ACID)
    raff_c, extr_c, _ = corr
    raff_m, extr_m, info = closed
    assert bool(info["feasible"])

    # D: the closure against the correlation at the unit's own charge.
    dist = REEDistribution(name, elements, concentration=conc)
    for el in elements:
        assert float(info["D"][el]) == pytest.approx(
            float(dist.get_D(el, pH)), rel=DILUTE_TOL)
        # ... and so the same split of the same feed.
        assert float(raff_m[f"F_{el}"]) == pytest.approx(
            float(raff_c[f"F_{el}"]), rel=DILUTE_TOL)
        assert float(extr_m[f"F_{el}"]) == pytest.approx(
            float(extr_c[f"F_{el}"]), rel=DILUTE_TOL)

    # Loading: the closure's free extractant is the unit's whole charge
    # (dilute), on the record basis, and its loading fraction is the one the
    # correlation's capacity uses, basis_units_per_ree * REE / extractant.
    # Before the fix the closure saw half the extractant: 0.25 M free, and
    # twice the loading.
    assert float(info["free_extractant"][0]) == pytest.approx(conc, rel=1e-6)
    loaded = sum(float(extr_m[f"F_{el}"]) for el in elements)
    assert float(info["theta"][0]) == pytest.approx(
        ext.basis_units_per_ree * loaded / conc, rel=1e-9)
    # The extract leaves on the unit's basis, as the solvent came in.
    assert float(extr_m[f"F_{name}"]) == pytest.approx(
        float(solvent[f"F_{name}"]), rel=1e-15)

