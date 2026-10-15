"""Counter-ion exchange mechanism: validation paths (#266)."""


import math

import pytest

from difflow_ree import SaponifiedCorrelationWarning
from difflow_ree.database import (
    Extractant,
    PHCoefficients,
    get_extractant_database,
)
from difflow_ree.equilibrium.distribution import REEDistribution

REFERENCE_COUNTER_ION = 0.5  # M, the anchor this synthetic record is fitted at


def _record(name, *, basis="saponified", with_block=True):
    """A saponified extractant, with or without a counter-ion correlation."""
    return Extractant(
        name=name,
        full_name="synthetic saponified carboxylic acid",
        formula="C10H18O2",
        molecular_weight=170.0,
        density=0.92,
        pKa=4.9,
        extractant_type="acidic_carboxylic",
        typical_concentration=0.45,
        stoichiometry_protons=3,
        stoichiometry_extractant=3,
        stoichiometry_basis="monomer",
        ph_coefficients={
            "Nd": PHCoefficients(a=-12.0, b=3.0, c=0.0),
            "Sm": PHCoefficients(a=-11.6, b=3.0, c=0.0),
        },
        temperature_coefficients={"Nd": 0.0, "Sm": 0.0},
        valid_ph_range=(3.5, 5.5),
        valid_temp_range=(288.0, 323.0),
        reference_concentration=0.45,
        concentration_exponent=3.0,
        cost_usd_kg=5.0,
        counter_ion="NH4",
        saponification_degree=0.8,
        correlation_basis=basis,
        counter_ion_coefficients=(
            {"Nd": PHCoefficients(a=-0.93, b=0.0, c=0.0),
             "Sm": PHCoefficients(a=-0.53, b=0.0, c=0.0)}
            if with_block else None
        ),
        counter_ion_reference=REFERENCE_COUNTER_ION if with_block else None,
        mechanism="counter_ion_exchange" if with_block else "cation_exchange",
    )


@pytest.fixture
def saponified(request):
    """Register a synthetic record on the singleton, and take it away after."""
    db = get_extractant_database()
    name = f"SynthSap_{request.node.name[:24]}"
    db.add_extractant(name, _record(name))
    yield name
    db.remove_extractant(name)


@pytest.fixture
def saponified_on_ph(request):
    db = get_extractant_database()
    name = f"SynthPh_{request.node.name[:24]}"
    db.add_extractant(name, _record(name, with_block=False))
    yield name
    db.remove_extractant(name)


def test_absolute_d_requires_the_counter_ion_concentration(saponified):
    dist = REEDistribution(extractant=saponified, elements=("Nd",))
    assert math.isfinite(float(dist.get_D("Nd", counter_ion_conc=0.5)))
    with pytest.raises(ValueError, match="counter_ion_conc"):
        dist.get_D("Nd")


def test_a_block_without_a_reference_is_refused():
    from difflow_ree.database import _load_counter_ion_block
    with pytest.raises(ValueError, match="reference_counter_ion"):
        _load_counter_ion_block({"elements": {"Nd": {"a": -0.9}}})


def test_the_mechanism_without_the_block_raises(saponified_on_ph):
    with pytest.raises(ValueError, match="counter_ion_coefficients"):
        REEDistribution(extractant=saponified_on_ph, elements=("Nd",),
                        mechanism="counter_ion_exchange")


def test_a_saponified_basis_on_the_ph_path_warns(saponified_on_ph):
    with pytest.warns(SaponifiedCorrelationWarning):
        REEDistribution(extractant=saponified_on_ph, elements=("Nd",))


def test_a_saponified_basis_needs_a_counter_ion():
    record = _record("Bad", with_block=False)
    with pytest.raises(ValueError, match="correlation_basis"):
        Extractant(**{**record.__dict__, "saponification_degree": 0.0,
                      "counter_ion": None})
