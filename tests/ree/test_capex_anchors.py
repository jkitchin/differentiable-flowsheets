"""The capital cost estimate is anchored to published projects.

These tests are about PROVENANCE and ARITHMETIC, not about whether $302 M was
the right number for a plant nobody built. What they hold fixed:

* every anchor's ``source`` resolves to a real entry in ``sources.yaml``,
  because an anchor whose citation does not exist is worse than no anchor;
* the anchor is reproduced exactly at its own capacity, year and base-case
  stage count -- that is the definition of anchoring, and it is the one
  property a scaling law can get wrong silently;
* rows sum to the total, at every scope and stage count;
* the scopes stay ordered by how much plant they enclose;
* an unknown year or scope raises rather than guessing;
* the section split, which no anchor published in full, stays consistent with
  the one section figure that IS published -- Avalon's solvent-extraction
  circuit at 33 % of total capital.
"""

import math

import pytest

from difflow_ree.economics.costs import (
    CAPEX_ANCHORS,
    CEPCI,
    _DISCLOSED_SX_SHARE,
    _SECTION_SHARES,
    _SX_EQUIPMENT,
    capex_basis,
    estimate_capex,
)
from difflow_ree.provenance import load_sources


class TestAnchorProvenance:
    def test_every_anchor_cites_a_source_that_exists(self):
        sources = load_sources()
        for scope, anchor in CAPEX_ANCHORS.items():
            key = anchor["source"]
            assert key in sources, (
                f"scope {scope!r} cites {key!r}, which is not in sources.yaml"
            )

    def test_anchor_sources_are_disclosed_not_estimated(self):
        """A project cost published by its owner is DISCLOSED, not ESTIMATED.

        The distinction is the whole point of this rewrite: these figures are
        checkable against a filing. If one ever falls back to ESTIMATED the
        function has lost its basis and should say so.
        """
        sources = load_sources()
        for scope, anchor in CAPEX_ANCHORS.items():
            cls = sources[anchor["source"]]["cls"]
            assert cls == "DISCLOSED", f"{scope!r} anchor is {cls}"

    def test_every_anchor_has_a_url(self):
        sources = load_sources()
        for anchor in CAPEX_ANCHORS.values():
            assert sources[anchor["source"]].get("url", "").startswith("http")

    def test_every_anchor_states_its_capacity_basis(self):
        """Feed REO and separated REO differ by a factor of several."""
        for scope, anchor in CAPEX_ANCHORS.items():
            assert anchor["capacity_basis"] in ("REO feed", "separated REO")

    def test_every_anchor_declares_battery_limits(self):
        for scope, anchor in CAPEX_ANCHORS.items():
            assert anchor["includes"], scope
            assert anchor["excludes"], scope

    def test_anchor_sections_are_known_line_items(self):
        for scope, anchor in CAPEX_ANCHORS.items():
            for section in anchor["sections"]:
                assert section in _SECTION_SHARES, (scope, section)

    def test_derived_class_is_wider_than_the_anchor_class(self):
        """Scaling a bankable estimate does not give a bankable estimate.

        The Phase 2 anchor is AACE Class 3 (-20%/+30%). Capacity-factoring it
        to another plant is Class 5, and the reported accuracy has to widen
        accordingly or the output overstates what it knows.
        """
        basis = capex_basis("integrated")
        assert basis["aace_class"].startswith("AACE Class 3")
        assert "Class 5" in basis["derived_class"]
        lo_a, hi_a = basis["accuracy"]
        lo_d, hi_d = basis["derived_accuracy"]
        assert lo_d < lo_a and hi_d > hi_a

    def test_unknown_scope_raises_and_names_the_alternatives(self):
        with pytest.raises(ValueError, match="unknown scope"):
            capex_basis("whatever")
        with pytest.raises(ValueError, match="sx_retrofit"):
            capex_basis("whatever")


class TestAnchorReproduction:
    """At the anchor's own basis, the answer IS the anchor."""

    @pytest.mark.parametrize("scope", sorted(CAPEX_ANCHORS))
    def test_anchor_reproduced_at_its_own_capacity_and_year(self, scope):
        anchor = CAPEX_ANCHORS[scope]
        capex = estimate_capex(
            annual_ree_tonnes=anchor["capacity_tpy"],
            n_stages_extraction=10,
            n_stages_scrubbing=5,
            n_stages_stripping=5,
            include_precipitation=True,
            include_ce_removal=True,
            year=anchor["year"],
            scope=scope,
            n_stages_reference=20,
        )
        assert capex["total"] == pytest.approx(anchor["capex_usd"], rel=1e-12)

    @pytest.mark.parametrize("scope", sorted(CAPEX_ANCHORS))
    def test_no_stage_reference_means_no_stage_assumption(self, scope):
        """Default stage handling must not smuggle in an invented base case.

        No anchor publishes a stage count, so with ``n_stages_reference=None``
        the stage count must not move the total at all -- otherwise the
        default answer depends on a number nobody reported.
        """
        anchor = CAPEX_ANCHORS[scope]
        kw = dict(
            annual_ree_tonnes=anchor["capacity_tpy"],
            include_precipitation=True,
            include_ce_removal=True,
            year=anchor["year"],
            scope=scope,
        )
        few = estimate_capex(n_stages_extraction=4, n_stages_scrubbing=2,
                             n_stages_stripping=2, **kw)["total"]
        many = estimate_capex(n_stages_extraction=40, n_stages_scrubbing=20,
                              n_stages_stripping=20, **kw)["total"]
        assert few == pytest.approx(anchor["capex_usd"], rel=1e-12)
        assert many == pytest.approx(few, rel=1e-12)


class TestArithmetic:
    @pytest.mark.parametrize("scope", sorted(CAPEX_ANCHORS))
    @pytest.mark.parametrize("n_ref", [None, 16])
    def test_rows_sum_to_total(self, scope, n_ref):
        capex = estimate_capex(
            500, 8, 4, 4, scope=scope, n_stages_reference=n_ref,
        )
        rows = sum(v for k, v in capex.items() if k != "total")
        assert rows == pytest.approx(capex["total"], rel=1e-12)

    def test_all_values_are_plain_floats(self):
        """examples/09 iterates the dict and divides -- keep it printable."""
        capex = estimate_capex(500, 8, 4, 4)
        for key, value in capex.items():
            assert isinstance(key, str)
            assert isinstance(value, float), key
            assert math.isfinite(value)

    def test_capacity_scaling_is_the_stated_power_law(self):
        anchor = CAPEX_ANCHORS["separation_plant"]
        base = estimate_capex(
            anchor["capacity_tpy"], 8, 4, 4, year=anchor["year"],
        )["total"]
        tenth = estimate_capex(
            anchor["capacity_tpy"] / 10, 8, 4, 4, year=anchor["year"],
        )["total"]
        assert tenth / base == pytest.approx(0.1 ** 0.6, rel=1e-12)

    def test_year_moves_the_answer_by_the_cepci_ratio(self):
        a = estimate_capex(500, 8, 4, 4, year=2018)["total"]
        b = estimate_capex(500, 8, 4, 4, year=2024)["total"]
        assert b / a == pytest.approx(CEPCI[2024] / CEPCI[2018], rel=1e-12)

    def test_unknown_year_raises_rather_than_extrapolating(self):
        with pytest.raises(ValueError, match="no CEPCI value"):
            estimate_capex(500, 8, 4, 4, year=1975)

    def test_stage_count_moves_only_the_stage_driven_fraction(self):
        """Doubling the cascade must not double the civils and the licence."""
        kw = dict(annual_ree_tonnes=500, scope="separation_plant",
                  n_stages_reference=16)
        single = estimate_capex(n_stages_extraction=8, n_stages_scrubbing=4,
                                n_stages_stripping=4, **kw)["total"]
        double = estimate_capex(n_stages_extraction=16, n_stages_scrubbing=8,
                                n_stages_stripping=8, **kw)["total"]
        assert 1.0 < double / single < 2.0

    def test_a_retrofit_is_more_stage_sensitive_than_a_refinery(self):
        """Almost all of a retrofit's capital IS the cascade."""
        def ratio(scope):
            kw = dict(annual_ree_tonnes=500, scope=scope,
                      n_stages_reference=16, include_precipitation=False)
            big = estimate_capex(n_stages_extraction=16, n_stages_scrubbing=8,
                                 n_stages_stripping=8, **kw)["total"]
            small = estimate_capex(n_stages_extraction=8, n_stages_scrubbing=4,
                                   n_stages_stripping=4, **kw)["total"]
            return big / small

        assert ratio("sx_retrofit") > ratio("separation_plant")


class TestScopeOrdering:
    def test_scopes_are_ordered_by_how_much_plant_they_enclose(self):
        """The ordering is the lesson: battery limits dominate the answer.

        Same capacity, same year, same stages; the only difference is what the
        estimate is taken to include. A reader who quotes the retrofit number
        for a standalone refinery is out by more than an order of magnitude,
        which is why ``scope`` has no silent default behaviour worth guessing.
        """
        kw = dict(annual_ree_tonnes=500, n_stages_extraction=8,
                  n_stages_scrubbing=4, n_stages_stripping=4,
                  include_precipitation=False, year=2024)
        retrofit = estimate_capex(scope="sx_retrofit", **kw)["total"]
        plant = estimate_capex(scope="separation_plant", **kw)["total"]
        integrated = estimate_capex(scope="integrated", **kw)["total"]
        assert retrofit < plant < integrated
        assert plant / retrofit > 10

    def test_default_scope_is_not_the_cheapest_one(self):
        kw = dict(annual_ree_tonnes=500, n_stages_extraction=8,
                  n_stages_scrubbing=4, n_stages_stripping=4)
        assert (estimate_capex(**kw)["total"]
                > estimate_capex(scope="sx_retrofit", **kw)["total"])

    def test_a_retrofit_has_no_precipitation_section_to_include(self):
        """Asking for one is not an error, it is just absent from the scope."""
        with_it = estimate_capex(500, 8, 4, 4, scope="sx_retrofit",
                                 include_precipitation=True)
        without = estimate_capex(500, 8, 4, 4, scope="sx_retrofit",
                                 include_precipitation=False)
        assert "precipitation" not in with_it
        assert with_it["total"] == pytest.approx(without["total"], rel=1e-12)

    def test_dropping_precipitation_drops_its_row_and_its_money(self):
        """The row going away is not enough; the total has to fall.

        Reallocating a dropped section over the remaining rows would quote the
        price of a plant WITH precipitation to a caller who asked for one
        without, and the rows would still sum to the total, so nothing would
        look wrong.
        """
        without = estimate_capex(500, 8, 4, 4, include_precipitation=False)
        with_it = estimate_capex(500, 8, 4, 4, include_precipitation=True)
        assert "precipitation" not in without
        assert "precipitation" in with_it
        assert without["total"] < with_it["total"]
        # And by about the section's own share, not by some other amount.
        dropped = with_it["precipitation"] / with_it["total"]
        assert without["total"] / with_it["total"] == pytest.approx(
            1.0 - dropped, rel=0.02)


class TestAgainstTheOneDisclosedSectionShare:
    """``_SECTION_SHARES`` is ESTIMATED. One published figure constrains it.

    Avalon's release breaks out exactly one section: "the largest capital
    expense is the solvent extraction circuit consisting of over 1,000
    mixer-settlers, and makes up 33% of the total capital cost at US$101
    million". Nothing else any anchor publishes says anything about a section.

    The release does not define where "the solvent extraction circuit" stops,
    so these tests hold a bracket rather than a match. Without them the
    section split is unfalsifiable.
    """

    def _shares(self, scope="separation_plant"):
        sections = CAPEX_ANCHORS[scope]["sections"]
        total = sum(_SECTION_SHARES[s] for s in sections)
        return {s: _SECTION_SHARES[s] / total for s in sections}

    def test_mixer_settlers_alone_come_in_under_the_disclosed_share(self):
        """A vessel count is less than a circuit."""
        assert self._shares()["mixer_settlers"] < _DISCLOSED_SX_SHARE

    def test_the_whole_sx_equipment_group_comes_in_over_it(self):
        """Vessels plus their tanks, pumps and instruments is more."""
        shares = self._shares()
        group = sum(v for k, v in shares.items() if k in _SX_EQUIPMENT)
        assert group > _DISCLOSED_SX_SHARE

    def test_the_sx_group_lands_near_the_disclosed_dollar_figure(self):
        """Not just ordered -- the right size.

        At Avalon's own capacity and year the estimate reproduces US$302 M by
        construction, so this tests the SPLIT: does the SX equipment group
        land near the US$101 million the release puts on that circuit? It
        comes out about a quarter high, which is the expected sign: the group
        is the wider of the two readings of the disclosed boundary.
        """
        anchor = CAPEX_ANCHORS["separation_plant"]
        c = estimate_capex(
            annual_ree_tonnes=anchor["capacity_tpy"],
            n_stages_extraction=8, n_stages_scrubbing=4, n_stages_stripping=4,
            year=anchor["year"], scope="separation_plant",
        )
        group = sum(v for k, v in c.items() if k in _SX_EQUIPMENT)
        assert 0.70 < group / 101.0e6 < 1.30

    def test_a_stage_does_not_cost_fifty_thousand_dollars(self):
        """The published price of a mixer-settler, used as a floor.

        US$101 M over "more than 1,000" mixer-settlers is at MOST about
        US$101,000 per unit in 2011 dollars -- an upper bound, since more
        units means less each, and it covers a circuit rather than bare
        vessels. difflow's mixer-settler row at Avalon's basis must not imply
        a per-unit price cheaper than that by an order of magnitude, which is
        the failure a round $50,000-per-stage assumption walks into.
        """
        anchor = CAPEX_ANCHORS["separation_plant"]
        c = estimate_capex(
            annual_ree_tonnes=anchor["capacity_tpy"],
            n_stages_extraction=8, n_stages_scrubbing=4, n_stages_stripping=4,
            year=anchor["year"], scope="separation_plant",
        )
        per_unit = c["mixer_settlers"] / 1000.0
        assert per_unit > 50.0e3


class TestPlausibility:
    """Cross-checks against the anchors, not against a textbook correlation."""

    def test_unit_capital_is_in_the_disclosed_range(self):
        """$/annual tonne should bracket the projects it came from.

        Avalon is $30 k/t in 2012 money and Energy Fuels Phase 2 is $54 k/t;
        a separation-plant estimate at a comparable capacity has to land in
        that neighbourhood or the scaling has gone wrong.
        """
        capex = estimate_capex(10000, 10, 5, 5, year=2024)["total"]
        per_tonne = capex / 10000
        assert 25e3 < per_tonne < 120e3

    def test_a_few_hundred_tonne_refinery_is_an_eight_figure_project(self):
        """The failure this replaced returned single-digit millions here."""
        total = estimate_capex(500, 8, 4, 4, include_precipitation=False)["total"]
        assert 20e6 < total < 200e6
