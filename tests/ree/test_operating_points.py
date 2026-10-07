"""Section pHs come off the D curves (2026 operating-point audit, R1).

The circuit defaults used to be fixed fractions of the extractant's fitted
pH window (#270). After the refit those did not sit between the groups: a
default D2EHPA ExtractStripCircuit left 67 % of the Sm and 99.8 % of the Y on
the barren organic, and an ExtractScrubStripCircuit asked for Gd/Tb/Dy/Y gave
a product of 0.09 % target purity and 0.2 % Y recovery. The design helpers
ignored their targets (always 10/5/5 stages) and raised KeyError for
naphthenic acid. These tests pin the cut rule, the circuits that use it, and
designs that meet what they were asked for in simulation.
"""

import math
import warnings

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.operating_points import (
    CUT_FACTOR,
    OperatingPointWarning,
    circuit_phase_ratios,
    cut_pHs,
    kremser_fraction,
    pH_where,
)
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
    design_extract_scrub_strip,
)
from difflow_ree.flowsheets.extract_strip import (
    ExtractStripCircuit,
    ExtractStripParams,
    design_extract_strip,
)

# A bastnaesite-like leach liquor, about 0.1 M total REE.
COMP = dict(La=0.025, Ce=0.045, Pr=0.005, Nd=0.015, Sm=0.002, Eu=0.0005,
            Gd=0.001, Tb=0.0002, Dy=0.0005, Y=0.002)
ELS = tuple(COMP)
HEAVY = ("Gd", "Tb", "Dy", "Y")


def _feed():
    return make_stream({"H2O": 55.5, **COMP}, 298.15, 101325.0)


@pytest.fixture(autouse=True)
def _quiet():
    # Stripping the heavies needs strong acid, below the fitted window; the
    # distribution model reports that extrapolation, which is not under test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


class TestCutRule:
    def test_pH_where_hits_the_factor(self):
        d = REEDistribution("D2EHPA", ("Nd", "Dy"), on_out_of_range="ignore")
        pH = pH_where(d, ("Nd", "Dy"), 1.0, phase_ratio=1.5)
        D = d.get_D_all(pH=pH)
        assert math.sqrt(float(D["Nd"]) * float(D["Dy"])) * 1.5 == \
            pytest.approx(1.0, rel=1e-9)

    def test_boundary_cut(self):
        oa = circuit_phase_ratios(1.0, 0.2, 0.5, 0.5)
        ph = cut_pHs("D2EHPA", ELS, HEAVY, extraction_OA=oa["extraction"],
                     scrub_OA=oa["scrubbing"], strip_OA=oa["stripping"])
        assert ph.basis == "cut" and ph.boundary == ("Gd", "Eu")
        d = REEDistribution("D2EHPA", ELS, on_out_of_range="ignore")
        for duty in ("extraction", "scrubbing"):
            D = d.get_D_all(pH=getattr(ph, duty))
            assert math.sqrt(float(D["Gd"]) * float(D["Eu"])) * oa[duty] == \
                pytest.approx(CUT_FACTOR[duty], rel=1e-9)
        D = d.get_D_all(pH=ph.stripping)
        assert max(float(D[e]) for e in HEAVY) * oa["stripping"] == \
            pytest.approx(CUT_FACTOR["stripping"], rel=1e-9)

    def test_bulk_cut_without_targets(self):
        ph = cut_pHs("PC88A", ELS, extraction_OA=1.5, strip_OA=3.0)
        d = REEDistribution("PC88A", ELS, on_out_of_range="ignore")
        D = d.get_D_all(pH=ph.extraction)
        assert min(float(D[e]) for e in ELS) * 1.5 == pytest.approx(10.0, rel=1e-9)
        D = d.get_D_all(pH=ph.stripping)
        assert max(float(D[e]) for e in ELS) * 3.0 == pytest.approx(0.1, rel=1e-9)
        assert ph.scrubbing == ph.extraction

    def test_solvating_has_no_pH_cut(self):
        assert cut_pHs("TBP", ELS, extraction_OA=1.5, strip_OA=3.0,
                       nitrate_conc=3.0) is None

    def test_targets_that_are_not_the_extractable_group_warn(self):
        with pytest.warns(OperatingPointWarning, match="no more extractable"):
            cut_pHs("D2EHPA", ELS, ("La",), extraction_OA=1.5, scrub_OA=7.5,
                    strip_OA=3.0)

    def test_kremser_fraction_matches_the_closed_form(self):
        assert float(kremser_fraction(2.0, 3)) == pytest.approx(1.0 / 15.0)
        assert float(kremser_fraction(1.0, 4)) == pytest.approx(0.2)
        assert float(kremser_fraction(1e300, 5)) == 0.0


class TestCircuitDefaults:
    @pytest.mark.parametrize("ext", ["D2EHPA", "PC88A", "Cyanex272"])
    def test_extract_strip_leaves_nothing_on_the_solvent(self, ext):
        """Before: D2EHPA left Sm 0.67 ... Y 0.998 on the barren organic."""
        r = ExtractStripCircuit(ExtractStripParams(extractant=ext, elements=ELS))(_feed())
        barren = get_flows(r["barren_organic"])
        for e in ELS:
            assert float(barren[e]) / COMP[e] < 1e-3, e
        assert float(r["recovery"]) > 0.99

    @pytest.mark.parametrize("ext", ["D2EHPA", "PC88A"])
    def test_scrub_circuit_separates_the_heavies(self, ext):
        """Before: purity 0.0009, Y recovery 0.002 on D2EHPA."""
        r = ExtractScrubStripCircuit(ExtractScrubStripParams(
            extractant=ext, elements=ELS, target_elements=HEAVY))(_feed())
        assert float(r["target_purity"]) > 0.99
        assert float(r["target_recovery"]["Y"]) > 0.95

    def test_given_pHs_are_kept(self):
        p = ExtractScrubStripParams(extractant="D2EHPA", elements=ELS,
                                    target_elements=HEAVY, extraction_pH=1.1,
                                    scrubbing_pH=0.7, stripping_pH=0.2)
        assert (p.extraction_pH, p.scrubbing_pH, p.stripping_pH) == (1.1, 0.7, 0.2)
        p = ExtractStripParams(extractant="D2EHPA", elements=ELS,
                               extraction_pH=1.3, stripping_pH=0.4)
        assert (p.extraction_pH, p.stripping_pH) == (1.3, 0.4)


class TestDesign:
    @pytest.mark.parametrize("ext", ["D2EHPA", "PC88A"])
    def test_extract_strip_design_meets_its_target(self, ext):
        p = design_extract_strip(COMP, ext, target_recovery=0.99)
        r = ExtractStripCircuit(p)(_feed())
        assert float(r["recovery"]) >= 0.99

    def test_strip_stages_follow_the_strip_D(self):
        """A strip pH where D is larger needs more strip stages."""
        easy = design_extract_strip(COMP, "PC88A", target_recovery=0.99)
        hard = design_extract_strip(COMP, "PC88A", target_recovery=0.99,
                                    stripping_pH=easy.stripping_pH + 0.5)
        assert hard.n_stripping_stages > easy.n_stripping_stages

    @pytest.mark.parametrize("ext,purity,recovery", [
        ("D2EHPA", 0.90, 0.80),
        ("PC88A", 0.90, 0.80),
        ("PC88A", 0.95, 0.80),
    ])
    def test_scrub_design_meets_its_targets(self, ext, purity, recovery):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings("ignore", message=".*outside the validity")
            p = design_extract_scrub_strip(COMP, HEAVY, ext, target_purity=purity,
                                           target_recovery=recovery)
        r = ExtractScrubStripCircuit(p)(_feed())
        assert float(r["target_purity"]) >= purity
        assert min(float(v) for v in r["target_recovery"].values()) >= recovery

    def test_scrub_design_follows_its_targets(self):
        """It used to return 10/5/5 whatever was asked."""
        a = design_extract_scrub_strip(COMP, HEAVY, "PC88A", 0.90, 0.80)
        b = design_extract_scrub_strip(COMP, HEAVY, "PC88A", 0.95, 0.80)
        assert (a.n_scrubbing_stages, a.scrubbing_pH) != (b.n_scrubbing_stages, b.scrubbing_pH)

    def test_infeasible_targets_warn(self):
        with pytest.warns(UserWarning, match="no design within the search"):
            design_extract_scrub_strip(COMP, HEAVY, "D2EHPA", 0.999, 0.999)

    def test_naphthenic_acid_no_longer_raises(self):
        """An unused separation-factor lookup raised KeyError here."""
        p = design_extract_scrub_strip(COMP, HEAVY, "naphthenic_acid")
        assert p.extractant == "naphthenic_acid"


class TestScans:
    """optimal_pH_for_separation and optimal_scrub_pH (audit R3)."""

    @pytest.mark.parametrize("ext", ["D2EHPA", "PC88A", "Cyanex272", "naphthenic_acid"])
    def test_flat_SF_returns_the_cut_inside_the_window_and_says_so(self, ext):
        """Before: argmax of 5e-15 noise, e.g. 4.879 for Cyanex272 (window
        [1.5, 3.5]) and 1.121 for naphthenic acid (D ~ 1e-10)."""
        from difflow_ree.equilibrium.distribution import SeparationFactorFlatWarning

        d = REEDistribution(ext, ("Pr", "Nd"))
        with pytest.warns(SeparationFactorFlatWarning, match="does not depend on pH"):
            pH, sf = d.optimal_pH_for_separation("Nd", "Pr")
        lo, hi = d._ext_data.valid_ph_range
        assert lo <= pH <= hi
        assert sf == pytest.approx(float(d.get_separation_factor("Nd", "Pr", 1.0)))
        q = REEDistribution(ext, ("Pr", "Nd"), on_out_of_range="ignore")
        cut = pH_where(q, ("Nd", "Pr"), 1.0)
        assert pH == pytest.approx(min(max(cut, lo), hi))

    def test_scrub_pH_honours_the_retention(self):
        """Before: always the top of (1, 4): pH 4, where D(Nd) = 1.6e11."""
        from difflow_ree.units.scrubbing import optimal_scrub_pH

        for keep in (0.9, 0.99):
            pH, D_t, D_i = optimal_scrub_pH("D2EHPA", "Nd", "La",
                                            min_target_retention=keep)
            assert 0.0 <= pH <= 2.0
            S = 1.0 / (D_t * 7.5)
            assert float(kremser_fraction(S, 5)) == pytest.approx(keep, abs=1e-6)
            assert D_i < D_t
        # Asking to keep more of the target costs impurity removal: higher pH.
        assert optimal_scrub_pH("D2EHPA", "Nd", "La", 0.99)[0] > \
            optimal_scrub_pH("D2EHPA", "Nd", "La", 0.9)[0]

    def test_scrub_pH_for_naphthenic_is_in_its_window(self):
        """Before: silently 1.0, three units below the window."""
        from difflow_ree.units.scrubbing import optimal_scrub_pH

        pH, _, _ = optimal_scrub_pH("naphthenic_acid", "Nd", "La")
        assert 4.0 <= pH <= 5.0

    def test_unreachable_retention_warns(self):
        from difflow_ree.units.scrubbing import optimal_scrub_pH

        with pytest.warns(UserWarning, match="No scrub pH"):
            optimal_scrub_pH("naphthenic_acid", "Nd", "La",
                             min_target_retention=0.999999, n_stages=1)
