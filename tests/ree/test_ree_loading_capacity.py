"""Loading, capacity and phase-flow tests for the REE extraction units.

One class per filed issue:

- #189: the loading correction must be a dimensionless fraction, so results
  cannot depend on the unit molar flows are expressed in.
- #190: free-extractant depletion is applied exactly once. That one place is
  the correlation's own ``n * log10([HA]/C_ref)`` term, which uses the fixed
  ``extractant_conc`` parameter -- so the stage D carries no loading term at
  all, rather than the m-th power a first reading of the issue suggests.
- #191: capacity and stoichiometry come from one source of truth, and the
  extractant balance closes.
- #192: REEExtractor and REEMixerSettler share one definition of the phase
  flows, and a missing phase raises instead of defaulting.
- #193: *both* loading limiters are smooth, so gradients are trustworthy at
  the constraints that bind, and the condition is observable in ``info``.

Every test in this file was checked against a faithful reimplementation of the
pre-fix arithmetic (``git show HEAD:src/difflow_ree/units/extraction.py`` plus
the pre-#191 ``LoadingIsotherm``) to confirm it actually fails against the
behaviour it claims to have fixed. Where a test cannot discriminate -- because
the property it asserts is an identity, or because the issue's own prediction
about the bug was wrong -- its docstring says so rather than implying a proof
it does not deliver.
"""

from pathlib import Path

import jax
import numpy as np
import pytest
import yaml
from jax.test_util import check_grads

import difflow_ree
from difflow.streams import make_stream, get_flows
from difflow_ree.database import get_extractant, list_extractants
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.loading import (
    EXTRACTANT_CAPACITIES,
    LoadingIsotherm,
    get_loading_isotherm,
    loading_correction,
)
from difflow_ree.units.extraction import (
    MixerSettlerParams,
    REEExtractor,
    REEExtractorParams,
    REEMixerSettler,
    _phase_flows,
    _smooth_free_fraction,
    _soft_saturation,
)


def _extract_totals(extractor, feed_flows, solvent_flows, elements):
    """Run an extractor on plain dicts and return (raffinate, extract) dicts."""
    feed = make_stream(feed_flows, T=298.15, P=101325.0)
    solvent = make_stream(solvent_flows, T=298.15, P=101325.0)
    raffinate, extract, info = extractor(feed, solvent)
    return get_flows(raffinate), get_flows(extract), info


# =============================================================================
# #189 -- the loading correction is dimensionless
# =============================================================================

class TestIssue189_DimensionlessLoading:
    """#189: the loading correction must be a genuine dimensionless fraction.

    A warning about what does *not* test this, because the first version of
    this class got it wrong. #189 predicted that the pre-fix
    ``avg_loading = F_in * 0.5 / F_org`` would make results depend on the unit
    the flows were written in. That prediction was simply false: it is a
    *ratio* of molar flows, so multiplying every flow by one constant leaves it
    -- and every recovery -- byte-identical. A uniform-rescale test therefore
    passes against the bug it claims to catch, and the two that used to live
    here proved nothing. Reimplementing the pre-fix path
    (``git show HEAD:src/difflow_ree/units/extraction.py``) confirms this.

    What the pre-fix expression actually was is a quantity with the units of
    L/mol -- ``(mol/s) / (mol/s) / (mol/L)`` once ``loading_fraction`` divided
    by ``max_ree_conc`` -- assembled from the wrong phases: it read the
    *aqueous feed* flow of the element and the *whole organic* flow, diluent
    included. So the discriminating tests are the ones that hold the loading
    physically fixed and move a quantity a true loading fraction cannot depend
    on:

    * how much inert diluent the solvent carries (the extractant is equally
      loaded either way), and
    * how much REE is in the aqueous feed (loading is a property of the
      organic phase; a clean solvent is clean whatever it is about to meet).

    Both are checked below, and both fail against the pre-fix path: the
    diluent sweep moves the pre-fix free-extractant multiplier over
    0.518 -> 0.988 (spread 0.47) where the fixed path holds it to 1.1e-16, and
    the feed sweep moves it 0.9998 -> 0.4176 where the fixed path holds it at
    exactly 1.0.
    """

    @staticmethod
    def _free_fraction_multiplier(info, elements, F_aq, F_org, pH=3.0):
        """Back out E / (D * F_org / F_aq): the entering-solvent free fraction.

        Uses the correlation D rather than the reported stage D, so it is the
        multiplier the *stage* applied however the model chose to apply it --
        pre-fix that factor sat inside D, post-fix it sits in E.
        """
        D_corr = float(
            REEDistribution(
                extractant="D2EHPA", elements=elements, concentration=0.5
            ).get_D(elements[0], pH=pH)
        )
        E = float(info["profiles"][elements[0]]["E"])
        return E * F_aq / (D_corr * F_org)

    def test_loading_does_not_depend_on_the_diluent_flow(self):
        """Diluting the solvent with inert does not change how loaded it is.

        theta = m * F_REE(organic) / F_extractant involves no diluent. The
        pre-fix ``initial_loading / capacity = (F_solv / F_org) / max_ree_conc``
        divided by the *whole* organic flow, so adding inert kerosene made the
        model think the solvent had become less loaded.
        """
        elements = ("Nd",)
        params = REEExtractorParams(
            n_stages=4, extractant="D2EHPA", elements=elements, pH=3.0,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)

        multipliers = []
        thetas = []
        for diluent_flow in (1.0, 5.0, 25.0, 100.0):
            feed_flows = {"H2O": 10.0, "Nd": 0.1}
            solvent_flows = {
                "D2EHPA": 1.0, "kerosene": diluent_flow, "Nd": 0.05,
            }
            _, _, info = _extract_totals(
                extractor, feed_flows, solvent_flows, elements
            )
            multipliers.append(
                self._free_fraction_multiplier(
                    info, elements,
                    F_aq=sum(feed_flows.values()),
                    F_org=1.0 + diluent_flow,
                )
            )
            thetas.append(float(info["theta_solvent"]))

        # theta is m * F_REE / F_extractant = 6 * 0.05 / 1.0, diluent-free
        for theta in thetas:
            assert theta == pytest.approx(0.3, rel=1e-12)
        for mult in multipliers:
            assert mult == pytest.approx(multipliers[0], rel=1e-12), (
                f"the loading correction moved with the diluent flow: "
                f"{multipliers}"
            )

    def test_loading_does_not_depend_on_the_aqueous_feed(self):
        """A clean solvent is unloaded whatever the feed carries.

        The pre-fix ``avg_loading = F_in * 0.5 / F_org`` read the aqueous feed
        flow of the element, so a clean solvent was reported as progressively
        more loaded the richer the feed got -- a category error, and the
        substance of #189.
        """
        elements = ("Nd",)
        params = REEExtractorParams(
            n_stages=4, extractant="D2EHPA", elements=elements, pH=3.0,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)

        for feed_nd in (1e-4, 1e-2, 0.1, 0.5):
            feed_flows = {"H2O": 10.0, "Nd": feed_nd}
            _, _, info = _extract_totals(
                extractor, feed_flows, {"D2EHPA": 1.0, "kerosene": 5.0},
                elements,
            )
            assert float(info["theta_solvent"]) == 0.0
            mult = self._free_fraction_multiplier(
                info, elements,
                F_aq=sum(feed_flows.values()), F_org=6.0,
            )
            assert mult == pytest.approx(1.0, rel=1e-12), (
                f"a clean solvent picked up a loading correction "
                f"{mult} from a feed of {feed_nd}"
            )

    def test_theta_solvent_is_m_times_the_flow_ratio(self):
        """theta_solvent is exactly m * F_REE(solvent) / F_extractant."""
        elements = ("Nd", "Dy")
        params = REEExtractorParams(
            n_stages=4, extractant="D2EHPA", elements=elements, pH=3.0,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)
        m = extractor._isotherm.m
        _, _, info = _extract_totals(
            extractor,
            {"H2O": 10.0, "Nd": 0.1, "Dy": 0.1},
            {"D2EHPA": 2.0, "kerosene": 5.0, "Nd": 0.05, "Dy": 0.02},
            elements,
        )
        assert float(info["theta_solvent"]) == pytest.approx(
            m * (0.05 + 0.02) / 2.0, rel=1e-14
        )

    @pytest.mark.parametrize(
        "unit_scale", [1e12, 1e6, 1.0, 1e-6, 1e-12, 1e-30, 1e-150]
    )
    def test_recovery_invariant_over_the_whole_float64_range(self, unit_scale):
        """A uniform rescale of every flow must not move any recovery.

        Honest scope: this does *not* discriminate against #189's original
        bug -- ``F_in * 0.5 / F_org`` was already invariant to a uniform
        rescale, so the issue's own prediction was wrong (see the class
        docstring). What it does discriminate against is the absolute
        ``1e-10`` floor that ``F_aq = jnp.maximum(F_aq, 1e-10)`` used to put on
        the aqueous flow, which is a hidden unit and broke this invariance
        below about 1e-11: the case below returned recovery 0.32564 at scale 1
        and 0.03297 at scale 1e-12. The floor is now relative to the streams'
        own total flow, so the supported range is the whole float64 range.
        """
        elements = ("La", "Nd", "Dy")
        params = REEExtractorParams(
            n_stages=4,
            extractant="D2EHPA",
            elements=elements,
            pH=2.0,
            include_loading=True,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)
        base_feed = {"H2O": 10.0, "La": 0.20, "Nd": 0.15, "Dy": 0.05}
        base_solvent = {"D2EHPA": 1.0, "kerosene": 5.0}

        def recoveries(scale):
            feed_flows = {k: v * scale for k, v in base_feed.items()}
            solvent_flows = {k: v * scale for k, v in base_solvent.items()}
            _, ext_flows, _ = _extract_totals(
                extractor, feed_flows, solvent_flows, elements
            )
            return {e: float(ext_flows[e]) / feed_flows[e] for e in elements}

        reference = recoveries(1.0)
        scaled = recoveries(unit_scale)
        for elem in reference:
            assert scaled[elem] == pytest.approx(reference[elem], rel=1e-12), (
                f"{elem} recovery depends on the flow unit at scale "
                f"{unit_scale}: {scaled[elem]} vs {reference[elem]}"
            )

    @pytest.mark.parametrize("unit_scale", [1e12, 1.0, 1e-12, 1e-150])
    def test_include_loading_false_is_also_unit_invariant(self, unit_scale):
        """The no-isotherm path had its own absolute floor; it is gone too.

        ``loading_ratio = F_solvent / max(F_in + F_solvent, 1e-10)`` is not the
        ratio it claims to be once both flows fall below 1e-10.
        """
        elements = ("Nd", "Dy")
        params = REEExtractorParams(
            n_stages=4, extractant="D2EHPA", elements=elements, pH=2.0,
            include_loading=False, extractant_conc=0.5,
        )
        extractor = REEExtractor(params)
        base_feed = {"H2O": 10.0, "Nd": 0.15, "Dy": 0.05}
        base_solvent = {"D2EHPA": 1.0, "kerosene": 5.0, "Nd": 0.02}

        def recoveries(scale):
            feed_flows = {k: v * scale for k, v in base_feed.items()}
            solvent_flows = {k: v * scale for k, v in base_solvent.items()}
            _, ext_flows, _ = _extract_totals(
                extractor, feed_flows, solvent_flows, elements
            )
            return {e: float(ext_flows[e]) / feed_flows[e] for e in elements}

        reference = recoveries(1.0)
        scaled = recoveries(unit_scale)
        for elem in reference:
            assert scaled[elem] == pytest.approx(reference[elem], rel=1e-12), (
                f"{elem} at scale {unit_scale}: {scaled[elem]} vs "
                f"{reference[elem]}"
            )

    def test_recovery_invariant_at_capacity(self):
        """Invariance must also hold where the capacity limiter is active."""
        # Feed far above capacity so the limiter dominates the answer.
        elements = ("Nd", "Dy")

        def recoveries(unit_scale):
            params = REEExtractorParams(
                n_stages=10,
                extractant="D2EHPA",
                elements=elements,
                pH=3.0,
                extractant_conc=0.5,
            )
            extractor = REEExtractor(params)
            feed_flows = {
                k: v * unit_scale
                for k, v in {"H2O": 10.0, "Nd": 5.0, "Dy": 5.0}.items()
            }
            solvent_flows = {
                k: v * unit_scale
                for k, v in {"D2EHPA": 1.0, "kerosene": 5.0}.items()
            }
            _, ext, info = _extract_totals(
                extractor, feed_flows, solvent_flows, elements
            )
            return (
                {e: float(ext[e]) / feed_flows[e] for e in elements},
                float(info["capacity_scale"]),
            )

        rec_mol, scale_mol = recoveries(1.0)
        rec_tiny, scale_tiny = recoveries(1e-20)

        assert scale_mol < 0.5, "limiter should be active in this test"
        assert scale_mol == pytest.approx(scale_tiny, rel=1e-12)
        for elem in rec_mol:
            assert rec_mol[elem] == pytest.approx(rec_tiny[elem], rel=1e-12)

    def test_apparent_D_takes_a_fraction(self):
        """apparent_D's argument is theta, and theta = 1 means saturated."""
        iso = get_loading_isotherm("D2EHPA", 0.5)
        D_inf = 10.0

        assert float(iso.apparent_D(D_inf, 0.0)) == pytest.approx(D_inf)
        # Fully loaded: only the 0.01 floor on the free fraction remains
        assert float(iso.apparent_D(D_inf, 1.0)) == pytest.approx(
            D_inf * 0.01 ** iso.m
        )
        # Half loaded: exactly (1/2)^m
        assert float(iso.apparent_D(D_inf, 0.5)) == pytest.approx(
            D_inf * 0.5 ** iso.m
        )


# =============================================================================
# #191 -- one source of truth for stoichiometry and capacity
# =============================================================================

class TestIssue191_Stoichiometry:
    """#191: capacity is 1/m with m read from the extraction mechanism."""

    def test_stoichiometry_matches_the_yaml_declaration(self):
        """m comes from the YAML mechanism, not from a literal in the code.

        ``max_loading * monomers_per_ree == 1`` is *not* worth asserting:
        ``max_loading`` is defined as ``1 / monomers_per_ree``, so it holds for
        any value of m and would have passed against the 0.33-vs-3-dimers
        discrepancy #191 was filed about. The independent statement is that m
        equals what ``data/extractants.yaml`` declares -- ``extractant_molecules``
        times 2 on a dimer basis -- which is the source the issue says should
        be read and which the pre-#191 code ignored entirely.
        """
        yaml_path = (
            Path(difflow_ree.__file__).parent / "data" / "extractants.yaml"
        )
        with open(yaml_path) as fh:
            records = yaml.safe_load(fh)["extractants"]

        checked = 0
        for name in list_extractants():
            if name not in records:
                continue  # runtime-registered extractant, not from the YAML
            stoich = records[name]["stoichiometry"]
            per_species = 2.0 if stoich.get("basis") == "dimer" else 1.0
            expected = stoich["extractant_molecules"] * per_species
            ext = get_extractant(name)
            assert ext.monomers_per_ree == pytest.approx(expected, rel=1e-15), (
                f"{name}: database says {ext.monomers_per_ree} monomers per "
                f"REE, the YAML declares {stoich['extractant_molecules']} "
                f"{stoich.get('basis', 'monomer')}(s) = {expected}"
            )
            # 0.33 was the pre-#191 literal for every acidic extractant; the
            # YAML says 3 dimers, i.e. 1/6.
            assert ext.max_loading == pytest.approx(1.0 / expected, rel=1e-15)
            checked += 1

        assert checked >= 4, "expected at least the four YAML extractants"

    def test_isotherm_exponent_is_monomers_per_ree(self):
        """The isotherm exponent equals the database stoichiometry."""
        for name in list_extractants():
            ext = get_extractant(name)
            iso = get_loading_isotherm(name, 0.5)
            assert iso.m == ext.monomers_per_ree, (
                f"{name}: isotherm exponent {iso.m} != "
                f"monomers_per_ree {ext.monomers_per_ree}"
            )
            assert iso.max_loading == pytest.approx(ext.max_loading, rel=1e-15)

    def test_dimer_extractants_bind_six_monomers(self):
        """The acidic extractants are declared as 3 dimers = 6 monomers."""
        for name in ("D2EHPA", "PC88A", "Cyanex272"):
            assert get_extractant(name).monomers_per_ree == 6.0
            assert get_loading_isotherm(name, 0.5).max_loading == pytest.approx(
                1.0 / 6.0
            )
        # TBP is a solvating extractant declared on a monomer basis
        assert get_extractant("TBP").monomers_per_ree == 3.0

    def test_isotherm_capacity_follows_m(self):
        """LoadingIsotherm cannot hold a capacity that disagrees with m."""
        iso = LoadingIsotherm(m=6.0, extractant_conc=0.5)
        assert iso.max_loading == pytest.approx(1.0 / 6.0)
        assert iso.max_ree_conc == pytest.approx(0.5 / 6.0)

    def test_total_extractant_conservation(self):
        """Bound extractant never exceeds the extractant fed.

        Scope, stated plainly because #191 asked for a conservation test
        believing it would have caught the stoichiometry discrepancy, and it
        cannot. Without an independent free-extractant state (#196, not
        implemented) there is nothing to conserve *against*: defining
        ``free := F * (1 - theta)`` with ``theta := m * X / F`` makes
        ``free + m * X == F`` the identity ``F - m*X + m*X == F``, true of any
        model output whatsoever. That identity is therefore only asserted here
        as a cheap check that ``info["theta_total"]`` really is the
        model's own loading fraction on the flow basis the test assumes.

        The load-bearing assertion is the *inequality* ``free > 0``: the model
        must never bind more extractant than was fed. That one does
        discriminate. The pre-fix capacity was ``max_ree_conc * F_org``, a
        concentration times the whole organic flow, which for the case below
        is 0.165 * 6 = 0.99 mol/s of REE -- around 3 mol/s of extractant bound
        out of 1.0 mol/s fed. Reimplementing the pre-fix path gives
        ``free = -2.0000`` at ``feed_ree = 1.0``, i.e. a hard failure here.

        A second independent check: ``theta_total`` is recomputed from the
        returned *streams*, not read from ``info``, so the reported loading
        cannot drift from the flows the unit actually produced.
        """
        elements = ("La", "Nd", "Dy")
        F_extractant = 1.0
        params = REEExtractorParams(
            n_stages=8,
            extractant="D2EHPA",
            elements=elements,
            pH=3.0,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)
        m = extractor._isotherm.m

        # Sweep from dilute to well past saturation
        for feed_ree in (1e-4, 1e-2, 0.05, 0.2, 1.0, 5.0):
            feed_flows = {"H2O": 10.0}
            feed_flows.update({e: feed_ree for e in elements})
            solvent_flows = {"D2EHPA": F_extractant, "kerosene": 5.0}
            _, ext_flows, info = _extract_totals(
                extractor, feed_flows, solvent_flows, elements
            )

            bound_ree = sum(float(ext_flows[e]) for e in elements)
            bound_extractant = m * bound_ree

            # Independent of info: theta from the returned streams alone.
            theta_from_streams = bound_extractant / F_extractant
            assert theta_from_streams == pytest.approx(
                float(info["theta_total"]), rel=1e-12
            ), (
                f"info['theta_total'] {float(info['theta_total'])} disagrees "
                f"with the outlet streams {theta_from_streams} at feed "
                f"{feed_ree}"
            )

            # THE discriminating assertion: the model must never bind more
            # extractant than was fed. The pre-fix capacity gives -2.0 here.
            free_extractant = F_extractant - bound_extractant
            assert free_extractant > 0.0, (
                f"negative free extractant at feed {feed_ree}: "
                f"{bound_extractant} bound of {F_extractant} fed"
            )

            # Tautology by construction (see the docstring); kept only to pin
            # the basis theta_total is expressed on.
            assert free_extractant + bound_extractant == pytest.approx(
                F_extractant, rel=0, abs=1e-12
            ), f"extractant balance does not close at feed {feed_ree}"


# =============================================================================
# #192 -- one definition of the phase flows
# =============================================================================

class TestIssue192_PhaseFlows:
    """#192: REEExtractor and REEMixerSettler agree on the phase flows."""

    @pytest.mark.parametrize(
        "feed_flows",
        [
            {"H2O": 10.0, "Nd": 0.1, "Dy": 0.05},
            # Concentrated liquor: H2O is a poor proxy for the aqueous flow
            {"H2O": 10.0, "Nd": 3.0, "Dy": 2.0, "Fe": 1.5},
        ],
    )
    def test_single_stage_equivalence(self, feed_flows):
        """N=1 Kremser and one 100%-efficient mixer-settler must agree.

        This is the Kremser identity frac_remaining = 1/(1 + E) with
        E = D * F_org / F_aq, so it holds only if both units compute F_aq and
        F_org the same way.
        """
        elements = ("Nd", "Dy")
        solvent_flows = {"D2EHPA": 1.0, "kerosene": 5.0}

        ext_params = REEExtractorParams(
            n_stages=1,
            extractant="D2EHPA",
            elements=elements,
            pH=2.0,
            include_loading=False,  # isolate the phase-flow definition
            extractant_conc=0.5,
        )
        stage_params = MixerSettlerParams(
            extractant="D2EHPA",
            elements=elements,
            pH=2.0,
            extractant_conc=0.5,
            stage_efficiency=1.0,
        )

        feed = make_stream(feed_flows, T=298.15, P=101325.0)
        solvent = make_stream(solvent_flows, T=298.15, P=101325.0)

        _, extract, _ = REEExtractor(ext_params)(feed, solvent)
        _, organic_out, _ = REEMixerSettler(stage_params)(feed, solvent)

        ext_flows = get_flows(extract)
        org_flows = get_flows(organic_out)

        for elem in elements:
            rec_kremser = float(ext_flows[elem]) / feed_flows[elem]
            rec_stage = float(org_flows[elem]) / feed_flows[elem]
            assert rec_kremser == pytest.approx(rec_stage, rel=1e-10), (
                f"{elem}: REEExtractor recovery {rec_kremser} != "
                f"REEMixerSettler recovery {rec_stage}"
            )

    def test_aqueous_flow_is_the_whole_aqueous_phase(self):
        """Everything that is not extractant or diluent is aqueous."""
        flows = {"H2O": 10.0, "Nd": 3.0, "Fe": 1.5, "D2EHPA": 1.0, "kerosene": 5.0}
        F_aq, F_org = _phase_flows(flows, "D2EHPA", "kerosene")
        assert float(F_aq) == pytest.approx(14.5)
        assert float(F_org) == pytest.approx(6.0)

    def test_missing_organic_phase_raises(self):
        """A solvent with no extractant and no diluent is an error, not 1.0."""
        with pytest.raises(ValueError, match="no organic phase"):
            _phase_flows({"H2O": 1.0, "Nd": 0.1}, "D2EHPA", "kerosene")

    def test_missing_aqueous_phase_raises(self):
        """An aqueous stream that is entirely organic is an error."""
        with pytest.raises(ValueError, match="no aqueous phase"):
            _phase_flows({"D2EHPA": 1.0, "kerosene": 5.0}, "D2EHPA", "kerosene")

    def test_error_names_the_species_present(self):
        """The message must name the missing phase and what was there."""
        with pytest.raises(ValueError) as exc:
            _phase_flows({"Organic": 10.0, "Nd": 0.0}, "D2EHPA", "kerosene")
        message = str(exc.value)
        assert "D2EHPA" in message and "kerosene" in message
        assert "Organic" in message

    def test_extractor_raises_on_solvent_without_carrier(self):
        """The unit surfaces the error rather than defaulting F_org to 1.0."""
        params = REEExtractorParams(
            n_stages=3, extractant="D2EHPA", elements=("Nd",), pH=2.0
        )
        feed = make_stream({"H2O": 10.0, "Nd": 0.1}, T=298.15, P=101325.0)
        solvent = make_stream({"Organic": 10.0, "Nd": 0.0}, T=298.15, P=101325.0)
        with pytest.raises(ValueError, match="no organic phase"):
            REEExtractor(params)(feed, solvent)

    def test_mixer_settler_raises_on_solvent_without_carrier(self):
        """Same contract in REEMixerSettler."""
        params = MixerSettlerParams(
            extractant="D2EHPA", elements=("Nd",), pH=2.0
        )
        aq = make_stream({"H2O": 10.0, "Nd": 0.1}, T=298.15, P=101325.0)
        org = make_stream({"Organic": 10.0, "Nd": 0.0}, T=298.15, P=101325.0)
        with pytest.raises(ValueError, match="no organic phase"):
            REEMixerSettler(params)(aq, org)


# =============================================================================
# #193 -- the capacity limiter is smooth and observable
# =============================================================================

def _recovery_vs_solvent(solvent_scale, elements=("Nd", "Dy"), n_stages=6):
    """Total REE recovery as a function of the solvent flow (traceable)."""
    params = REEExtractorParams(
        n_stages=n_stages,
        extractant="D2EHPA",
        elements=elements,
        pH=3.0,
        extractant_conc=0.5,
    )
    extractor = REEExtractor(params)
    feed = make_stream(
        {"H2O": 10.0, "Nd": 0.5, "Dy": 0.5}, T=298.15, P=101325.0
    )
    solvent = make_stream(
        {
            "D2EHPA": 1.0 * solvent_scale,
            "kerosene": 5.0 * solvent_scale,
        },
        T=298.15,
        P=101325.0,
    )
    _, extract, info = extractor(feed, solvent)
    ext_flows = get_flows(extract)
    total_extracted = sum(ext_flows[e] for e in elements)
    return total_extracted / 1.0, info


class TestIssue193_SmoothCapacityLimiter:
    """#193: no kink in the gradient at the binding capacity constraint."""

    @staticmethod
    def _capacity_gap(scale):
        """uncapped_extracted - capacity, the quantity the old clamp switched on."""
        _, info = _recovery_vs_solvent(scale)
        return float(info["uncapped_extracted"]) - float(info["capacity"])

    def _solvent_scale_at_capacity(self):
        """Bisect for the solvent flow where extraction just fills capacity."""
        lo, hi = 1e-3, 1e3
        assert self._capacity_gap(lo) > 0, "expected capacity-limited at low solvent"
        assert self._capacity_gap(hi) < 0, "expected capacity-rich at high solvent"
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self._capacity_gap(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-13 * max(1.0, hi):
                break
        return 0.5 * (lo + hi)

    def test_gradient_at_the_capacity_constraint(self):
        """check_grads on recovery w.r.t. solvent flow, at the switch point."""
        scale_star = self._solvent_scale_at_capacity()
        _, info = _recovery_vs_solvent(scale_star)
        assert float(info["uncapped_extracted"]) == pytest.approx(
            float(info["capacity"]), rel=1e-6
        ), "bisection did not land on the capacity point"

        def recovery(scale):
            rec, _ = _recovery_vs_solvent(scale)
            return rec

        check_grads(recovery, (scale_star,), order=1, modes=["rev"])

    def test_gradient_is_continuous_across_the_constraint(self):
        """Left and right derivatives must agree at the constraint."""
        scale_star = self._solvent_scale_at_capacity()

        def recovery(scale):
            rec, _ = _recovery_vs_solvent(scale)
            return rec

        d = jax.grad(recovery)
        eps = 1e-6 * scale_star
        g_left = float(d(scale_star - eps))
        g_right = float(d(scale_star + eps))
        assert g_left == pytest.approx(g_right, rel=1e-4), (
            f"gradient jumps at the capacity constraint: "
            f"{g_left} (below) vs {g_right} (above)"
        )

    def test_capacity_condition_is_reported(self):
        """A design pinned at the wall must be distinguishable in info."""
        elements = ("Nd", "Dy")
        params = REEExtractorParams(
            n_stages=10,
            extractant="D2EHPA",
            elements=elements,
            pH=3.0,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)

        # Comfortably below capacity
        _, _, info_lo = _extract_totals(
            extractor,
            {"H2O": 10.0, "Nd": 1e-3, "Dy": 1e-3},
            {"D2EHPA": 1.0, "kerosene": 5.0},
            elements,
        )
        # Far beyond capacity
        _, _, info_hi = _extract_totals(
            extractor,
            {"H2O": 10.0, "Nd": 5.0, "Dy": 5.0},
            {"D2EHPA": 1.0, "kerosene": 5.0},
            elements,
        )

        for key in (
            "theta_total",
            "capacity",
            "capacity_scale",
            "capacity_clamped_fraction",
            "uncapped_extracted",
        ):
            assert key in info_lo, f"info is missing {key}"

        assert float(info_lo["capacity_scale"]) == pytest.approx(1.0, abs=1e-6)
        assert float(info_lo["capacity_clamped_fraction"]) < 1e-6
        assert float(info_lo["theta_total"]) < 0.05

        assert float(info_hi["capacity_scale"]) < 0.1
        assert float(info_hi["capacity_clamped_fraction"]) > 0.9
        assert float(info_hi["theta_total"]) == pytest.approx(1.0, abs=0.05)
        # Strictly under saturation: the limiter never binds more extractant
        # than was fed.
        assert float(info_hi["theta_total"]) < 1.0

    def test_capacity_is_a_flow_from_the_extractant_only(self):
        """capacity == F_extractant / m, with no diluent and no concentration."""
        elements = ("Nd",)
        params = REEExtractorParams(
            n_stages=3,
            extractant="D2EHPA",
            elements=elements,
            pH=3.0,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)
        _, _, info = _extract_totals(
            extractor,
            {"H2O": 10.0, "Nd": 0.1},
            {"D2EHPA": 2.0, "kerosene": 30.0},
            elements,
        )
        m = extractor._isotherm.m
        assert float(info["capacity"]) == pytest.approx(2.0 / m, rel=1e-12)

    def test_solvent_without_extractant_flow_raises(self):
        """Capacity is F_extractant / m, so the flow must be declared.

        Otherwise the capacity is identically zero and the unit would
        silently return zero recovery.
        """
        params = REEExtractorParams(
            n_stages=3,
            extractant="D2EHPA",
            elements=("Nd",),
            pH=3.0,
            include_loading=True,
        )
        feed = make_stream({"H2O": 10.0, "Nd": 0.1}, T=298.15, P=101325.0)
        solvent = make_stream({"kerosene": 5.0}, T=298.15, P=101325.0)
        with pytest.raises(ValueError, match="does not declare a flow"):
            REEExtractor(params)(feed, solvent)

        # include_loading=False has no capacity, so it is allowed
        params_no_loading = REEExtractorParams(
            n_stages=3,
            extractant="D2EHPA",
            elements=("Nd",),
            pH=3.0,
            include_loading=False,
        )
        _, extract, _ = REEExtractor(params_no_loading)(feed, solvent)
        assert float(get_flows(extract)["Nd"]) > 0.0

    def test_sharpness_controls_the_transition(self):
        """Larger capacity_sharpness approaches the hard clamp."""
        elements = ("Nd", "Dy")
        scales = {}
        for k in (2, 4, 16):
            params = REEExtractorParams(
                n_stages=10,
                extractant="D2EHPA",
                elements=elements,
                pH=3.0,
                extractant_conc=0.5,
                capacity_sharpness=k,
            )
            # Half of capacity: the hard clamp would not fire at all
            _, _, info = _extract_totals(
                REEExtractor(params),
                {"H2O": 10.0, "Nd": 0.04, "Dy": 0.04},
                {"D2EHPA": 1.0, "kerosene": 5.0},
                elements,
            )
            scales[k] = float(info["capacity_scale"])
            assert float(info["uncapped_extracted"]) < float(info["capacity"])

        assert scales[2] < scales[4] < scales[16] <= 1.0
        assert scales[16] == pytest.approx(1.0, abs=1e-3)

    def test_third_phase_margin_is_a_signed_constraint(self):
        """#193: the third-phase limit is usable as an inequality."""
        elements = ("Nd", "Dy")
        params = MixerSettlerParams(
            extractant="D2EHPA",
            elements=elements,
            pH=3.0,
            extractant_conc=0.5,
            third_phase_loading_limit=0.1,
        )
        stage = REEMixerSettler(params)
        org = make_stream({"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0)

        # Lightly loaded: feasible, positive margin
        aq_lo = make_stream(
            {"H2O": 10.0, "Nd": 0.001, "Dy": 0.001}, T=298.15, P=101325.0
        )
        _, _, info_lo = stage(aq_lo, org)
        assert not bool(info_lo["third_phase_formed"])
        assert float(info_lo["third_phase_margin"]) > 0.0
        assert float(info_lo["third_phase_margin"]) == pytest.approx(
            0.1 - float(info_lo["organic_loading"])
        )

        # Heavily loaded: infeasible, negative margin
        aq_hi = make_stream(
            {"H2O": 10.0, "Nd": 1.0, "Dy": 1.0}, T=298.15, P=101325.0
        )
        _, _, info_hi = stage(aq_hi, org)
        assert bool(info_hi["third_phase_formed"])
        assert float(info_hi["third_phase_margin"]) < 0.0

    def test_third_phase_margin_is_differentiable(self):
        """The margin has a usable gradient; the boolean has none."""
        def margin(nd_flow):
            params = MixerSettlerParams(
                extractant="D2EHPA",
                elements=("Nd",),
                pH=3.0,
                extractant_conc=0.5,
                third_phase_loading_limit=0.1,
            )
            stage = REEMixerSettler(params)
            aq = make_stream({"H2O": 10.0, "Nd": nd_flow}, T=298.15, P=101325.0)
            org = make_stream(
                {"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0
            )
            _, _, info = stage(aq, org)
            return info["third_phase_margin"]

        g = float(jax.grad(margin)(0.05))
        assert g < 0.0, "more REE must reduce the third-phase margin"
        check_grads(margin, (0.05,), order=1, modes=["rev"])


# =============================================================================
# #190 -- free-extractant depletion applied exactly once
# =============================================================================

class TestIssue190_SingleDepletionCorrection:
    """#190: exactly one free-extractant depletion mechanism, in the correlation.

    The stage D therefore carries *no* loading dependence -- see
    ``test_D_has_no_free_extractant_falloff_at_all``, which corrects the
    "D falls off as the m-th power" reading this class was first written with.
    """

    ELEMENTS = ("Nd",)
    F_EXTRACTANT = 1.0

    def _sweep(self):
        """Sweep aqueous REE dilute -> near-saturation at fixed extractant.

        Returns a list of (theta, free_fraction, D_apparent, D_correlation)
        with theta from an explicit extractant balance computed here in the
        test rather than read from the model.
        """
        params = REEExtractorParams(
            n_stages=1,
            extractant="D2EHPA",
            elements=self.ELEMENTS,
            pH=3.0,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)
        m = extractor._isotherm.m
        D_corr = float(
            REEDistribution(
                extractant="D2EHPA", elements=self.ELEMENTS, concentration=0.5
            ).get_D("Nd", pH=3.0)
        )

        rows = []
        for feed_nd in np.geomspace(1e-5, 1.0, 24):
            feed_flows = {"H2O": 10.0, "Nd": float(feed_nd)}
            solvent_flows = {"D2EHPA": self.F_EXTRACTANT, "kerosene": 5.0}
            raff, ext, _ = _extract_totals(
                extractor, feed_flows, solvent_flows, self.ELEMENTS
            )
            F_org_ree = float(ext["Nd"])
            F_aq_ree = float(raff["Nd"])

            # Explicit free-extractant balance, in monomer equivalents:
            #     [HA]_total = [HA]_free + m * [RE-complex]
            free = self.F_EXTRACTANT - m * F_org_ree
            free_fraction = free / self.F_EXTRACTANT
            theta = 1.0 - free_fraction

            F_aq = sum(feed_flows.values())
            F_org = self.F_EXTRACTANT + 5.0
            # Apparent D from the outlet split (a ratio of concentrations,
            # expressed with the phase flows)
            D_app = (F_org_ree / F_org) / ((F_aq_ree / F_aq) + 1e-300)
            rows.append((theta, free_fraction, D_app, D_corr))
        return m, rows

    def test_dilute_limit_recovers_the_correlation(self):
        """At zero loading the stage D is the correlation D, uncorrected."""
        m, rows = self._sweep()
        theta, free_fraction, D_app, D_corr = rows[0]
        assert theta < 1e-3, "first sweep point should be effectively dilute"
        assert D_app == pytest.approx(D_corr, rel=1e-6), (
            "a second free-extractant correction is being applied on top of "
            "the correlation's concentration term (#190)"
        )

    def test_D_has_no_free_extractant_falloff_at_all(self):
        """After #190 the stage D carries *no* loading dependence whatsoever.

        The previous version of this test claimed "D falls off as the m-th
        power of free extractant" and asserted an effective exponent
        ``p <= m + 0.5 = 6.5`` against a measured ``p ~ 0.03``, a bound 200x
        looser than the value. Both the claim and the bound were wrong.

        What the code does: the only free-extractant term left is
        ``n * log10(concentration / C_ref)`` inside ``REEDistribution.get_D``,
        and ``concentration`` there is the fixed ``extractant_conc`` parameter,
        so **nothing in D varies with stage loading**. The apparent exponent
        measured from the outlet split is therefore ~0, not m; the small
        residual comes from the capacity limiter reshaping the split, not from
        D. That is the intended state: exactly one depletion mechanism, and it
        lives in the correlation's calibration rather than in a per-stage
        factor.

        The assertions below are chosen to fail if the doubled correction
        returns. Reimplementing the pre-fix path measures ``p = 4.32`` and
        drives ``D_app / D_corr`` down to 0.121; the fixed path measures
        ``p = 0.03`` and never goes below 0.66.
        """
        m, rows = self._sweep()
        _, _, D0, D_corr = rows[0]

        # 1. D_app must stay of the same order as the correlation over the
        #    whole sweep. A doubled (1-theta)^m factor drops it by ~8x here.
        ratios = [D / D_corr for _, _, D, _ in rows]
        assert min(ratios) > 0.5, (
            f"the outlet split implies D fell to {min(ratios):.3g} x the "
            f"correlation value; the doubled correction (#190) reaches 0.121"
        )
        assert max(ratios) <= 1.0 + 1e-9

        # 2. The effective exponent in D_app/D_corr ~ free**p, over the region
        #    where loading bites, must be ~0 and nowhere near m or 2m.
        pts = [(fr, D) for theta, fr, D, _ in rows if 0.05 < theta < 0.9]
        assert len(pts) >= 4, "sweep did not cover the loaded region"
        xs = np.log(np.array([fr for fr, _ in pts]))
        ys = np.log(np.array([D / D_corr for _, D in pts]))
        p = float(np.polyfit(xs, ys, 1)[0])
        assert abs(p) < 1.0, (
            f"the stage D shows a free-extractant power law of exponent "
            f"{p:.3f}; after #190 there is no loading term in D at all, and "
            f"a single (1-theta)^m factor would give p ~ {m}, a doubled one "
            f"p ~ {2 * m}"
        )

        # 3. Directly: the model must be far above the single m-th power
        #    curve, let alone the doubled one.
        for theta, free_fraction, D_app, _ in rows:
            if free_fraction < 0.02 or theta < 1e-6:
                continue  # power law meaningless this close to saturation
            single = D_corr * free_fraction ** m
            assert D_app >= single * (1 - 1e-9), (
                f"D falls off at least as fast as free**{m} at "
                f"theta={theta:.4f}: {D_app:.6g} < {single:.6g}"
            )

    def test_apparent_D_matches_the_explicit_balance(self):
        """LoadingIsotherm.apparent_D is exactly one m-th power."""
        iso = get_loading_isotherm("D2EHPA", 0.5)
        m = iso.m
        F_extractant = 1.0
        D_inf = 25.0

        # Stay above the 0.01 floor apparent_D puts on the free fraction
        for bound_ree in (0.0, 0.01, 0.05, 0.1, 0.15):
            free = F_extractant - m * bound_ree
            theta = m * bound_ree / F_extractant
            assert float(iso.apparent_D(D_inf, theta)) == pytest.approx(
                D_inf * (free / F_extractant) ** m, rel=1e-12
            )

    def test_stage_D_carries_no_loading_factor(self):
        """The reported stage D is the correlation D at every loading."""
        params = REEExtractorParams(
            n_stages=3,
            extractant="D2EHPA",
            elements=("Nd",),
            pH=3.0,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)
        D_corr = float(
            REEDistribution(
                extractant="D2EHPA", elements=("Nd",), concentration=0.5
            ).get_D("Nd", pH=3.0)
        )
        for feed_nd in (1e-5, 1e-2, 0.5, 5.0):
            _, _, info = _extract_totals(
                extractor,
                {"H2O": 10.0, "Nd": feed_nd},
                {"D2EHPA": 1.0, "kerosene": 5.0},
                ("Nd",),
            )
            assert float(info["profiles"]["Nd"]["D"]) == pytest.approx(
                D_corr, rel=1e-12
            ), "the stage path is still applying a second depletion factor"


# =============================================================================
# #193 (round 2) -- the entering-solvent free fraction is smooth too
# =============================================================================

def _raffinate_vs_loaded_solvent(nd_org, n_stages=5, sharpness=8):
    """Nd left in the raffinate as a function of the solvent's Nd loading.

    This is the recycled-solvent path: an extract-strip circuit returns a
    partly loaded solvent to the extractor, so ``theta_solvent`` is a lever a
    gradient-based optimizer moves. With ``m = 6`` and 1.0 mol/s of D2EHPA the
    solvent saturates at ``nd_org = 1/6``.
    """
    params = REEExtractorParams(
        n_stages=n_stages,
        extractant="D2EHPA",
        elements=("Nd",),
        pH=3.0,
        capacity_sharpness=sharpness,
    )
    extractor = REEExtractor(params)
    feed = make_stream({"H2O": 10.0, "Nd": 0.1}, T=298.15, P=101325.0)
    solvent = make_stream(
        {"D2EHPA": 1.0, "kerosene": 5.0, "Nd": nd_org}, T=298.15, P=101325.0
    )
    raffinate, _, _ = extractor(feed, solvent)
    return get_flows(raffinate)["Nd"]


class TestIssue193_SmoothEnteringSolventLoading:
    """#193's own complaint, on the loaded-solvent path.

    ``free_fraction_in = jnp.maximum(1.0 - theta_solvent, 0.0)`` was a hard
    kink with a *dead lever* beyond it -- exactly the pathology #193 was filed
    about and exactly the dead column ``difflow.planning.health`` warns about.
    Measured against that code, at ``nd_org = 0.1666`` (theta = 0.9996) the
    gradient was 0.1959 and at ``nd_org = 0.1667`` (theta = 1.0002) it was
    identically 0.0, and stayed 0.0 forever after. Because ``theta_solvent`` is
    computed against ``F_extractant / m`` rather than ``max_ree_conc * F_org``,
    the wall sat about 6x lower in solvent loading than the code it replaced,
    so it bit earlier.
    """

    SATURATION = 1.0 / 6.0  # nd_org where theta_solvent == 1

    def test_gradient_at_the_saturation_point(self):
        """check_grads on the raffinate w.r.t. solvent loading, at theta = 1."""
        check_grads(
            _raffinate_vs_loaded_solvent,
            (self.SATURATION,),
            order=1,
            modes=["rev"],
        )

    def test_gradient_is_continuous_across_saturation(self):
        """No jump in the derivative at theta_solvent == 1."""
        d = jax.grad(_raffinate_vs_loaded_solvent)
        eps = 1e-7
        g_left = float(d(self.SATURATION - eps))
        g_right = float(d(self.SATURATION + eps))
        assert g_left == pytest.approx(g_right, rel=1e-4), (
            f"gradient jumps at solvent saturation: {g_left} vs {g_right}"
        )

    @pytest.mark.parametrize("nd_org", [0.1667, 0.2, 0.5, 1.0, 2.0])
    def test_gradient_is_non_zero_past_saturation(self, nd_org):
        """The lever must stay alive beyond saturation, not go dead.

        The clipped version returned exactly 0.0 at every one of these points.
        """
        g = float(jax.grad(_raffinate_vs_loaded_solvent)(nd_org))
        assert g != 0.0, (
            f"dead lever at nd_org={nd_org} (theta={6 * nd_org:.2f}): the "
            f"gradient is identically zero, so no optimizer can move it"
        )
        assert np.isfinite(g)
        # More REE already in the solvent must leave more REE behind.
        assert g > 0.0

    def test_free_fraction_is_positive_and_monotone(self):
        """The smooth free fraction never reaches zero and never increases."""
        thetas = np.concatenate([
            np.linspace(0.0, 2.0, 41), np.geomspace(2.0, 1e3, 20),
        ])
        values = [float(_smooth_free_fraction(t, 8.0)) for t in thetas]
        assert values[0] == pytest.approx(1.0)
        for t, v in zip(thetas, values):
            assert v > 0.0, f"free fraction hit zero at theta={t}"
        for a, b in zip(values, values[1:]):
            assert b <= a + 1e-15, "free fraction is not monotone decreasing"

    def test_free_fraction_matches_the_hard_clamp_when_far_below_capacity(self):
        """Well below saturation the smooth form is the hard one to 1e-6."""
        for theta in (0.0, 0.05, 0.1, 0.2):
            assert float(_smooth_free_fraction(theta, 8.0)) == pytest.approx(
                1.0 - theta, abs=1e-6
            )

    def test_free_fraction_at_saturation_is_the_stated_value(self):
        """At theta = 1 the free fraction is exactly 1 - 2**(-1/k)."""
        for k in (2.0, 4.0, 8.0, 16.0):
            assert float(_smooth_free_fraction(1.0, k)) == pytest.approx(
                1.0 - 2.0 ** (-1.0 / k), rel=1e-12
            )

    def test_free_fraction_is_reported(self):
        """The entering-solvent free fraction is observable in info."""
        params = REEExtractorParams(
            n_stages=5, extractant="D2EHPA", elements=("Nd",), pH=3.0
        )
        _, _, info = _extract_totals(
            REEExtractor(params),
            {"H2O": 10.0, "Nd": 0.1},
            {"D2EHPA": 1.0, "kerosene": 5.0, "Nd": 0.05},
            ("Nd",),
        )
        assert float(info["theta_solvent"]) == pytest.approx(0.3, rel=1e-12)
        assert float(info["free_fraction_in"]) == pytest.approx(
            float(_smooth_free_fraction(0.3, 8.0)), rel=1e-12
        )

    def test_soft_saturation_degenerate_inputs_are_finite(self):
        """capacity == 0 and total == capacity == 0 must not produce nan."""
        assert float(_soft_saturation(1.0, 0.0, 8.0)) == 0.0
        assert float(_soft_saturation(0.0, 0.0, 8.0)) == 1.0
        assert float(_soft_saturation(0.0, 1.0, 8.0)) == pytest.approx(1.0)
        assert float(_soft_saturation(1.0, 1.0, 8.0)) == pytest.approx(
            2.0 ** (-1.0 / 8.0)
        )
        # No overflow even where total**k would be inf
        assert float(_soft_saturation(1e60, 1.0, 16.0)) == pytest.approx(
            1e-60, rel=1e-9
        )
        for args in ((1.0, 0.0), (0.0, 0.0), (2.0, 1.0)):
            g = float(jax.grad(lambda t, c=args[1]: _soft_saturation(t, c, 8.0))(
                args[0]
            ))
            assert np.isfinite(g), f"non-finite gradient at {args}"


# =============================================================================
# Zero-flow phases and missing extractant (round 2)
# =============================================================================

class TestZeroFlowPhases:
    """A declared-but-empty phase must fail loudly, not be floored.

    ``_phase_flows`` used to check key *presence* only, so
    ``{"H2O": 0.0, "D2EHPA": 1.0, "kerosene": 5.0}`` passed, ``F_aq`` was
    floored at a magic ``1e-10`` and the extraction factor came out around
    3.3e10. That is the same defect class as the ``1.0`` default #192 objected
    to: a mis-specified stream hidden behind a plausible-looking number.
    """

    def test_zero_aqueous_flow_raises(self):
        with pytest.raises(ValueError, match="total flow is 0.0"):
            _phase_flows(
                {"H2O": 0.0, "D2EHPA": 1.0, "kerosene": 5.0},
                "D2EHPA", "kerosene",
            )

    def test_zero_organic_flow_raises(self):
        with pytest.raises(ValueError, match="total flow is 0.0"):
            _phase_flows(
                {"H2O": 10.0, "Nd": 0.1, "D2EHPA": 0.0, "kerosene": 0.0},
                "D2EHPA", "kerosene", require=("organic",),
            )

    def test_extractor_raises_on_an_empty_aqueous_feed(self):
        params = REEExtractorParams(
            n_stages=5, extractant="D2EHPA", elements=("Nd",), pH=3.0
        )
        feed = make_stream({"H2O": 0.0, "Nd": 0.0}, T=298.15, P=101325.0)
        solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0
        )
        with pytest.raises(ValueError, match="total flow is"):
            REEExtractor(params)(feed, solvent)

    def test_mixer_settler_raises_on_an_empty_aqueous_inlet(self):
        params = MixerSettlerParams(
            extractant="D2EHPA", elements=("Nd",), pH=3.0
        )
        aq = make_stream({"H2O": 0.0, "Nd": 0.0}, T=298.15, P=101325.0)
        org = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0
        )
        with pytest.raises(ValueError, match="total flow is"):
            REEMixerSettler(params)(aq, org)

    def test_a_tiny_but_real_aqueous_flow_is_accepted(self):
        """The guard is on zero, not on smallness: 1e-200 mol/s still runs."""
        F_aq, F_org = _phase_flows(
            {"H2O": 1e-200, "D2EHPA": 1e-200, "kerosene": 5e-200},
            "D2EHPA", "kerosene",
        )
        assert float(F_aq) == pytest.approx(1e-200)
        assert float(F_org) == pytest.approx(6e-200)

    def test_mixer_settler_third_phase_needs_the_extractant_flow(self):
        """#192/#193: loading is mol REE per mol extractant, so say how much.

        Without this, a solvent of pure diluent reported an organic loading of
        9.85e+28 and no error at all.
        """
        params = MixerSettlerParams(
            extractant="D2EHPA",
            elements=("Nd",),
            pH=3.0,
            third_phase_loading_limit=0.1,
        )
        aq = make_stream({"H2O": 10.0, "Nd": 0.5}, T=298.15, P=101325.0)
        org = make_stream({"kerosene": 5.0}, T=298.15, P=101325.0)
        with pytest.raises(ValueError, match="does not declare a flow"):
            REEMixerSettler(params)(aq, org)

    def test_mixer_settler_without_the_limit_still_accepts_pure_diluent(self):
        """No loading-dependent quantity requested, no extractant required."""
        params = MixerSettlerParams(
            extractant="D2EHPA", elements=("Nd",), pH=3.0
        )
        aq = make_stream({"H2O": 10.0, "Nd": 0.5}, T=298.15, P=101325.0)
        org = make_stream({"kerosene": 5.0}, T=298.15, P=101325.0)
        aq_out, org_out, info = REEMixerSettler(params)(aq, org)
        assert "organic_loading" not in info
        total = float(get_flows(aq_out)["Nd"]) + float(get_flows(org_out)["Nd"])
        assert total == pytest.approx(0.5, rel=1e-12)

    def test_reported_loading_is_finite_and_physical(self):
        """With the extractant declared, the loading is order 1, not 1e29."""
        params = MixerSettlerParams(
            extractant="D2EHPA",
            elements=("Nd",),
            pH=3.0,
            third_phase_loading_limit=0.1,
        )
        aq = make_stream({"H2O": 10.0, "Nd": 0.5}, T=298.15, P=101325.0)
        org = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0
        )
        _, org_out, info = REEMixerSettler(params)(aq, org)
        loading = float(info["organic_loading"])
        assert 0.0 < loading < 1.0, f"unphysical organic loading {loading}"
        assert loading == pytest.approx(
            float(get_flows(org_out)["Nd"]) / 1.0, rel=1e-12
        )


# =============================================================================
# #191 / #189 -- public API that changed shape (D10)
# =============================================================================

class TestChangedPublicAPI:
    """The #191 fix broke exported API. Pin the new contract."""

    def test_max_loading_is_no_longer_a_constructor_field(self):
        """LoadingIsotherm(max_loading=0.33) now raises; m=3.0 replaces it."""
        with pytest.raises(TypeError):
            LoadingIsotherm(max_loading=0.33)
        assert LoadingIsotherm(m=3.0).max_loading == pytest.approx(1.0 / 3.0)
        with pytest.raises(AttributeError):
            LoadingIsotherm(m=3.0).max_loading = 0.5

    def test_extractant_capacities_no_longer_carries_capacity(self):
        """The public dict holds Langmuir constants only."""
        for name, record in EXTRACTANT_CAPACITIES.items():
            assert set(record) == {"typical_K_L"}, (
                f"{name} still carries {sorted(set(record) - {'typical_K_L'})}"
            )

    def test_loading_correction_uses_the_isotherm_exponent(self):
        """loading_correction is (1 - theta_total)**isotherm.m, not **3.

        Untested and unused in-tree before this, while #191 silently changed
        both its exponent (3 -> m = 6 for the acidic extractants) and its theta
        basis (max_ree_conc halved), moving its output by a factor of 25.6 at
        the point below.
        """
        iso = get_loading_isotherm("D2EHPA", 0.5)
        assert iso.m == 6.0
        assert iso.max_ree_conc == pytest.approx(0.5 / 6.0)

        c_org = {"Nd": 0.0413}
        theta = 0.0413 / iso.max_ree_conc
        expected = 10.0 * (1.0 - theta) ** iso.m
        out = loading_correction({"Nd": 10.0}, c_org, iso)
        assert float(out["Nd"]) == pytest.approx(expected, rel=1e-12)
        # The pre-#191 form, for the record: (1 - 0.0413/0.165)**3 * 10
        old = 10.0 * (1.0 - 0.0413 / (0.33 * 0.5)) ** 3
        assert old / float(out["Nd"]) == pytest.approx(25.59, rel=1e-3)

    def test_loading_correction_is_multiplicative_and_shared(self):
        """Total loading, one factor, applied to every element's D."""
        iso = get_loading_isotherm("D2EHPA", 0.5)
        D_in = {"Nd": 10.0, "Dy": 200.0}
        c_org = {"Nd": 0.02, "Dy": 0.01}
        out = loading_correction(D_in, c_org, iso)
        assert set(out) == set(D_in)
        theta = (0.02 + 0.01) / iso.max_ree_conc
        factor = (1.0 - theta) ** iso.m
        for elem, D in D_in.items():
            assert float(out[elem]) == pytest.approx(D * factor, rel=1e-12)
        # One shared factor: the ratio of D values is untouched.
        assert float(out["Dy"]) / float(out["Nd"]) == pytest.approx(20.0)

    def test_loading_correction_floors_the_free_fraction(self):
        """Beyond saturation the correction saturates at 0.01**m, not 0."""
        iso = get_loading_isotherm("D2EHPA", 0.5)
        out = loading_correction({"Nd": 10.0}, {"Nd": 10.0}, iso)
        assert float(out["Nd"]) == pytest.approx(10.0 * 0.01 ** iso.m)

    def test_loading_correction_is_differentiable(self):
        """It is a model term, so grad must work through it.

        Checked against the analytic derivative rather than ``check_grads``:
        ``(1 - theta)**6`` is steep enough that the finite-difference estimate
        is only good to about 3e-5 relative, and the exact derivative is a
        stronger statement anyway.
        """
        iso = get_loading_isotherm("D2EHPA", 0.5)

        def corrected(c):
            return loading_correction({"Nd": 10.0}, {"Nd": c}, iso)["Nd"]

        c0 = 0.02
        theta = c0 / iso.max_ree_conc
        expected = (
            -10.0 * iso.m * (1.0 - theta) ** (iso.m - 1) / iso.max_ree_conc
        )
        g = float(jax.grad(corrected)(c0))
        assert g < 0.0, "more loading must lower D"
        assert g == pytest.approx(expected, rel=1e-12)
