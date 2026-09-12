"""The correlation's `[HA]` is FREE extractant, not total (issue #267).

`Q1` Eq. 2.88 is `lg D = lg K_ex + 3 lg (HA)o + 3 pH`, and Eq. 2.89 defines
`(HA)o = [HA]_0 - 3 (REA3)` -- total charged minus three monomers per extracted
RE(III). `REEDistribution` applies the same functional form against the total
charged. The two agree at low loading and diverge where the cascade works
hardest, and the error flatters the model: total overstates what is available,
so D is overpredicted exactly there.

Past "inaccurate" it goes to "impossible". `Z1` Table 4.36 system 2 implies an
organic loading of 152% of stoichiometric capacity; Eq. 2.89 returns a negative
free extractant and `log10` of it is undefined. difflow_ree never formed that
quantity, so it returned a finite D.
"""

import warnings

import jax
import pytest

from difflow_ree import (
    ExtractantCapacityWarning,
    check_loading_capacity,
    implied_loading_fraction,
    solve_free_extractant,
)
from difflow_ree.database import (
    Extractant,
    PHCoefficients,
    get_extractant_database,
)
from difflow_ree.equilibrium.distribution import REEDistribution

# Q1's naphthenic-acid form: three MONOMERS per RE(III), correlation exponent 3.
NAPHTHENIC_HA = 0.45     # mol/L, Z1 Table 4.36 system 1
NAPHTHENIC_PH = 4.19


def _naphthenic_like(name):
    """A record in Q1 Eq. 2.88's shape: n = 3, m = 3, anchored at 0.45 M.

    `a` is chosen so that D(Dy) against TOTAL extractant is 0.370 at the
    anchor pH -- the number the issue reports for the total basis.
    """
    import math
    a = math.log10(0.370) - 3.0 * NAPHTHENIC_PH
    return Extractant(
        name=name,
        full_name="naphthenic-acid-like carboxylic extractant",
        formula="C10H18O2",
        molecular_weight=170.0,
        density=0.92,
        pKa=4.9,
        extractant_type="acidic_carboxylic",
        typical_concentration=NAPHTHENIC_HA,
        stoichiometry_protons=3,
        stoichiometry_extractant=3,
        stoichiometry_basis="monomer",     # m = 3, as Eq. 2.89 counts
        ph_coefficients={"Dy": PHCoefficients(a=a, b=3.0, c=0.0)},
        temperature_coefficients={"Dy": 0.0},
        valid_ph_range=(3.8, 5.0),
        valid_temp_range=(288.0, 323.0),
        reference_concentration=NAPHTHENIC_HA,
        concentration_exponent=3.0,        # n = 3, as Eq. 2.88 has it
        cost_usd_kg=5.0,
    )


@pytest.fixture
def naphthenic(request):
    db = get_extractant_database()
    name = f"NaphLike_{request.node.name[:24]}"
    db.add_extractant(name, _naphthenic_like(name))
    yield name
    db.remove_extractant(name)


class TestTheOverprediction:
    def test_total_and_free_agree_on_a_clean_solvent(self, naphthenic):
        """At zero loading there is nothing bound, so the bases coincide."""
        dist = REEDistribution(extractant=naphthenic, elements=("Dy",),
                               concentration=NAPHTHENIC_HA)
        result = solve_free_extractant(dist, "Dy", 1e-9, pH=NAPHTHENIC_PH)
        assert float(result.overprediction) == pytest.approx(1.0, abs=1e-6)

    def test_the_anchor_case_overpredicts_by_a_third(self, naphthenic):
        """The number the issue reports: 0.370 against total, 0.277 free.

        pH 4.19, 0.45 M, a modestly loaded organic -- 0.05 M REE in the
        aqueous, which is inside the 0.1-0.8 M Y window Eq. 2.88 is stated
        for.
        """
        dist = REEDistribution(extractant=naphthenic, elements=("Dy",),
                               concentration=NAPHTHENIC_HA)
        result = solve_free_extractant(dist, "Dy", 0.05, pH=NAPHTHENIC_PH)

        assert float(result.D_total_basis) == pytest.approx(0.370, rel=1e-3)
        assert float(result.D) == pytest.approx(0.277, rel=0.02)
        assert float(result.overprediction) == pytest.approx(1.34, rel=0.02)

    def test_the_error_grows_with_loading(self, naphthenic):
        """It flatters the model exactly where the cascade works hardest."""
        dist = REEDistribution(extractant=naphthenic, elements=("Dy",),
                               concentration=NAPHTHENIC_HA)
        ratios = [
            float(solve_free_extractant(dist, "Dy", c, pH=NAPHTHENIC_PH)
                  .overprediction)
            for c in (1e-4, 1e-2, 5e-2, 1e-1)
        ]
        assert ratios == sorted(ratios)
        assert ratios[0] < 1.01 < ratios[-1]

    def test_d_varies_with_loading_again(self, naphthenic):
        """What #204's closing note said had been lost.

        Keeping the correlation's fixed-parameter concentration term left D
        independent of stage loading. Solving Eq. 2.89 restores the
        dependence without reintroducing the double count #190 found: the
        free extractant enters the correlation, it is not a factor applied
        afterwards.
        """
        dist = REEDistribution(extractant=naphthenic, elements=("Dy",),
                               concentration=NAPHTHENIC_HA)
        lean = solve_free_extractant(dist, "Dy", 1e-3, pH=NAPHTHENIC_PH)
        rich = solve_free_extractant(dist, "Dy", 1e-1, pH=NAPHTHENIC_PH)
        assert float(rich.D) < float(lean.D)
        assert float(rich.free_extractant) < float(lean.free_extractant)


class TestTheBalanceCloses:
    def test_the_solution_satisfies_both_equations(self, naphthenic):
        dist = REEDistribution(extractant=naphthenic, elements=("Dy",),
                               concentration=NAPHTHENIC_HA)
        c_aq = 0.05
        r = solve_free_extractant(dist, "Dy", c_aq, pH=NAPHTHENIC_PH)
        # Eq. 2.89: free = total - m * c_org, with m = 3
        assert float(r.free_extractant) == pytest.approx(
            NAPHTHENIC_HA - 3.0 * float(r.c_org), rel=1e-8)
        # and D is the distribution ratio it claims to be
        assert float(r.c_org) == pytest.approx(float(r.D) * c_aq, rel=1e-8)

    def test_it_never_returns_more_than_capacity(self, naphthenic):
        """The fixed point cannot land where Eq. 2.89 has no logarithm."""
        dist = REEDistribution(extractant=naphthenic, elements=("Dy",),
                               concentration=NAPHTHENIC_HA)
        for c_aq in (0.1, 1.0, 10.0, 100.0):
            r = solve_free_extractant(dist, "Dy", c_aq, pH=NAPHTHENIC_PH)
            assert 0.0 <= float(r.loading_fraction) <= 1.0
            assert float(r.free_extractant) >= 0.0

    def test_it_differentiates(self, naphthenic):
        """Implicit diff through the root, not an unrolled iteration."""
        dist = REEDistribution(extractant=naphthenic, elements=("Dy",),
                               concentration=NAPHTHENIC_HA)

        def loaded(c_aq):
            return solve_free_extractant(dist, "Dy", c_aq,
                                         pH=NAPHTHENIC_PH).c_org

        g = float(jax.grad(loaded)(0.05))
        assert g > 0.0                       # more in the aqueous, more bound
        fd = (float(loaded(0.0501)) - float(loaded(0.0499))) / 2e-4
        assert g == pytest.approx(fd, rel=1e-4)

    def test_a_dimeric_record_uses_its_own_stoichiometry(self):
        """m is monomers_per_ree, which is 6 for the dimeric extractants."""
        dist = REEDistribution(extractant="D2EHPA", elements=("Nd",),
                               concentration=0.5)
        r = solve_free_extractant(dist, "Nd", 0.01, pH=3.0)
        assert float(r.free_extractant) == pytest.approx(
            0.5 - 6.0 * float(r.c_org), rel=1e-8)


class TestTheImpossibleLoading:
    """`Z1` Table 4.36 system 2, which had to be rejected by hand."""

    Z1_SYSTEM_2_HA = 0.13      # mol/L naphthenic acid
    Z1_SYSTEM_2_ORG = 0.066    # mol/L RE implied by the tabulated D values

    def test_the_rejected_row_is_over_capacity(self, naphthenic):
        theta = float(implied_loading_fraction(
            naphthenic, self.Z1_SYSTEM_2_ORG, self.Z1_SYSTEM_2_HA))
        assert theta == pytest.approx(1.523, rel=1e-3)
        assert theta > 1.0

    def test_the_accepted_row_is_not(self, naphthenic):
        """System 1, at 19% of capacity, is why it became the anchor."""
        theta = float(implied_loading_fraction(naphthenic, 0.0285, 0.45))
        assert 0.15 < theta < 0.25

    def test_it_warns_rather_than_returning_a_number(self, naphthenic):
        with pytest.warns(ExtractantCapacityWarning, match="152|capacity"):
            check_loading_capacity(naphthenic, self.Z1_SYSTEM_2_ORG,
                                   self.Z1_SYSTEM_2_HA)

    def test_it_can_raise_instead(self, naphthenic):
        with pytest.raises(ValueError, match="capacity"):
            check_loading_capacity(naphthenic, self.Z1_SYSTEM_2_ORG,
                                   self.Z1_SYSTEM_2_HA, action="raise")

    def test_a_possible_loading_is_silent(self, naphthenic):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExtractantCapacityWarning)
            check_loading_capacity(naphthenic, 0.0285, 0.45)

    def test_an_unknown_action_is_refused(self, naphthenic):
        with pytest.raises(ValueError, match="action"):
            check_loading_capacity(naphthenic, 0.01, 0.45, action="shout")
