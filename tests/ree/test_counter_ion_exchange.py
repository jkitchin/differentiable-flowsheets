"""A saponified circuit runs a counter-ion exchange, not a proton one (#266).

`mechanism: cation_exchange` describes

    RE3+ + 3 HA(o) <-> REA3(o) + 3 H+                       (Q1 Eq. 2.87)

giving `log10(D) = a + b*pH` with b near 3. Industrial rare-earth circuits
saponify 30-80% of the extractant first, and the reaction that then runs is

    HL(o) + NH4OH = NH4L(o) + H2O                           (Z1 Eq. 4.95)
    RE3+ + 3 NH4L(o) = REL3(o) + 3 NH4+                     (Z1 Eq. 4.96)

with no proton on either side of the extraction. A `b` of 3 there is not a
slope that is too large; it is a slope on the wrong axis. Until #266 the
correlation layer had nowhere to put the real reaction -- `EXTRACTION_MECHANISMS`
was `("cation_exchange", "solvating")` -- so the caveat lived in prose.
"""

import warnings

import jax
import pytest

from difflow_ree import SaponifiedCorrelationWarning
from difflow_ree.database import (
    CORRELATION_BASES,
    EXTRACTION_MECHANISMS,
    Extractant,
    PHCoefficients,
    get_extractant_database,
)
from difflow_ree.equilibrium.distribution import (
    REEDistribution,
    get_separation_factor,
)

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


class TestTheMechanismExists:
    def test_it_is_a_third_mechanism(self):
        assert "counter_ion_exchange" in EXTRACTION_MECHANISMS

    def test_d_falls_as_the_counter_ion_builds_up(self, saponified):
        """RE3+ + p ML = REL_p + p M+, so D goes as [M+]^-p."""
        dist = REEDistribution(extractant=saponified, elements=("Nd",))
        low = float(dist.get_D("Nd", counter_ion_conc=REFERENCE_COUNTER_ION))
        high = float(dist.get_D("Nd", counter_ion_conc=5 * REFERENCE_COUNTER_ION))
        p = get_extractant_database().get(saponified).counter_ion_monomers
        assert high == pytest.approx(low * 5.0 ** (-p), rel=1e-6)

    def test_a_at_the_reference_is_log10_d(self, saponified):
        """At both references, D is 10**a and nothing else."""
        dist = REEDistribution(extractant=saponified, elements=("Nd",),
                               concentration=0.45)  # the record's own [HA]_ref
        D = float(dist.get_D("Nd", counter_ion_conc=REFERENCE_COUNTER_ION))
        assert D == pytest.approx(10 ** -0.93, rel=1e-6)

    def test_the_extractant_concentration_term_still_applies(self, saponified):
        """D goes as [HA]^n on this path too: more saponified extractant,
        more extraction. Only the *driving ion* changed."""
        ref = REEDistribution(extractant=saponified, elements=("Nd",),
                              concentration=0.45)
        rich = REEDistribution(extractant=saponified, elements=("Nd",),
                               concentration=0.90)
        at_ref = float(ref.get_D("Nd", counter_ion_conc=0.5))
        at_rich = float(rich.get_D("Nd", counter_ion_conc=0.5))
        assert at_rich == pytest.approx(at_ref * 2.0 ** 3.0, rel=1e-6)

    def test_ph_is_not_a_driving_variable(self, saponified):
        dist = REEDistribution(extractant=saponified, elements=("Nd",))
        at_4 = float(dist.get_D("Nd", pH=4.0, counter_ion_conc=0.5))
        at_5 = float(dist.get_D("Nd", pH=5.0, counter_ion_conc=0.5))
        assert at_4 == at_5

    def test_it_differentiates_through_the_counter_ion(self, saponified):
        dist = REEDistribution(extractant=saponified, elements=("Nd",))
        g = jax.grad(lambda m: dist.get_D("Nd", counter_ion_conc=m))(0.5)
        assert float(g) < 0.0  # D falls as [M+] rises


class TestTheCounterIonTermCancels:
    """The practical consequence: stage counts are untouched by all of this."""

    @pytest.mark.parametrize("m", [0.05, 0.5, 5.0])
    def test_the_separation_factor_is_independent_of_the_counter_ion(
        self, saponified, m
    ):
        dist = REEDistribution(extractant=saponified, elements=("Sm", "Nd"))
        beta = float(dist.get_separation_factor("Sm", "Nd",
                                                counter_ion_conc=m))
        assert beta == pytest.approx(10 ** (-0.53 + 0.93), rel=1e-9)

    def test_a_separation_factor_needs_no_counter_ion_at_all(self, saponified):
        """The term cancels, so requiring it would be asking for nothing."""
        beta = float(get_separation_factor("Sm", "Nd", saponified))
        assert beta == pytest.approx(10 ** 0.40, rel=1e-9)


class TestNoAnchorIsInvented:
    def test_absolute_d_requires_the_counter_ion_concentration(self, saponified):
        dist = REEDistribution(extractant=saponified, elements=("Nd",))
        with pytest.raises(ValueError, match="counter_ion_conc"):
            dist.get_D("Nd")

    def test_a_block_without_a_reference_is_refused(self, tmp_path):
        from difflow_ree.database import _load_counter_ion_block
        with pytest.raises(ValueError, match="reference_counter_ion"):
            _load_counter_ion_block({"elements": {"Nd": {"a": -0.9}}})

    def test_no_shipped_record_carries_the_block(self):
        """There is no measured anchor for [M+] in the sources cited here."""
        db = get_extractant_database()
        for name in db.list_extractants():
            assert db.get(name).counter_ion_coefficients is None, name

    def test_the_mechanism_without_the_block_raises(self, saponified_on_ph):
        with pytest.raises(ValueError, match="counter_ion_coefficients"):
            REEDistribution(extractant=saponified_on_ph, elements=("Nd",),
                            mechanism="counter_ion_exchange")


class TestTheWrongAxisIsReported:
    def test_a_saponified_basis_on_the_ph_path_warns(self, saponified_on_ph):
        with pytest.warns(SaponifiedCorrelationWarning, match="wrong|pH slope"):
            REEDistribution(extractant=saponified_on_ph, elements=("Nd",))

    def test_the_warning_says_separation_factors_survive(self, saponified_on_ph):
        with pytest.warns(SaponifiedCorrelationWarning) as record:
            REEDistribution(extractant=saponified_on_ph, elements=("Nd",))
        assert "SEPARATION FACTORS ARE UNAFFECTED" in str(record[0].message)

    @pytest.mark.parametrize(
        "extractant", ["D2EHPA", "PC88A", "Cyanex272", "TBP"])
    def test_a_shipped_record_does_not_warn(self, extractant):
        """The trigger is the declared basis, not the operating degree.

        Every acidic record here ships with saponification_degree 0.35 and
        proton-exchange coefficients. Warning on the degree would put a
        warning on every REE calculation in the package, where nothing is
        wrong -- and a warning that fires every time is read as noise.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", SaponifiedCorrelationWarning)
            REEDistribution(extractant=extractant, elements=("Nd",),
                            nitrate_conc=3.0)

    def test_the_basis_is_declared_data(self):
        db = get_extractant_database()
        for name in db.list_extractants():
            assert db.get(name).correlation_basis in CORRELATION_BASES

    def test_a_saponified_basis_needs_a_counter_ion(self):
        with pytest.raises(ValueError, match="correlation_basis"):
            record = _record("Bad", with_block=False)
            Extractant(**{**record.__dict__,
                          "saponification_degree": 0.0,
                          "counter_ion": None})
