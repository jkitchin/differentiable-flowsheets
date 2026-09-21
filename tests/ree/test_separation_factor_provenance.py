"""The tabulated separation factors describe the same physics as the
correlations they sit beside (issue #265).

``separation_factors.yaml`` and ``extractants.yaml`` were two independent
hand-authored answers to "what is the separation factor for this pair on this
extractant?" -- one tabulating it, the other supplying ``ph_coefficients`` you
subtract. They disagreed by up to 8x: the coefficients ran 1.3-2.8x high on 24
of 27 pairs, and all three Y/Dy pairs ran 4-8x low, which is a disagreement
about that pair rather than a calibration offset. A user got a different answer
depending on which API they reached for, with nothing warning them.

Neither set was measured, so the tie is broken by which one the simulator runs
on: the correlations. The table is now derived from them.
"""

import pytest
import yaml

from difflow_ree import get_sf_database
from difflow_ree.database import DATA_DIR, SeparationFactorDatabase
from difflow_ree.equilibrium.distribution import get_separation_factor


@pytest.fixture(scope="module")
def sf_db():
    return get_sf_database()


@pytest.fixture(scope="module")
def yaml_data():
    with open(DATA_DIR / "separation_factors.yaml") as handle:
        return yaml.safe_load(handle)


def _conditions_kwargs(conditions):
    kwargs = {"pH": conditions.get("pH"),
              "T": float(conditions.get("temperature_K", 298.15))}
    if "concentration_M" in conditions:
        kwargs["concentration"] = float(conditions["concentration_M"])
    if "nitrate_M" in conditions:
        kwargs["nitrate_conc"] = float(conditions["nitrate_M"])
    return kwargs


class TestTheTwoDescriptionsAgree:
    """The regression for #265, stated as the property that was violated."""

    def test_every_shipped_pair_matches_its_correlation(self, sf_db):
        mismatches = []
        for extractant in sf_db.list_extractants():
            data = sf_db.get(extractant)
            kwargs = _conditions_kwargs(data.conditions)
            pairs = {**data.adjacent_pairs, **data.group_pairs}
            for pair, tabulated in pairs.items():
                heavier, lighter = pair.split("_")
                correlated = float(get_separation_factor(
                    heavier, lighter, extractant, **kwargs))
                if abs(tabulated - correlated) > 1e-9 * max(correlated, 1.0):
                    mismatches.append(
                        f"{extractant} {pair}: table {tabulated:.4g} vs "
                        f"correlation {correlated:.4g}")
        assert not mismatches, "\n".join(mismatches)

    def test_the_yaml_ships_no_authored_numbers(self, yaml_data):
        """A list of pairs is derived; a mapping would be an override.

        Overrides are allowed and honoured -- but each one is a claim that a
        measured factor exists which the correlations cannot reproduce, and
        none such ships with difflow_ree.
        """
        for extractant, block in yaml_data["separation_factors"].items():
            for key in ("adjacent_pairs", "group_pairs"):
                assert isinstance(block[key], list), (
                    f"{extractant}.{key} gives values, so they are authored "
                    "and need a citation")

    def test_every_shipped_pair_is_reported_as_derived(self, sf_db):
        for extractant in sf_db.list_extractants():
            data = sf_db.get(extractant)
            pairs = set(data.adjacent_pairs) | set(data.group_pairs)
            assert data.derived == pairs

    def test_the_nitrate_driven_extractant_derives_too(self, sf_db):
        """TBP is solvating, so its factors come from ``nitrate_coefficients``.

        The issue had to exclude it: its conditions block states solvent
        strength as vol %, its coefficients want mol/L, and no conversion
        ships. It needs none -- what drives D there is the aqueous nitrate,
        which the same block declares in mol/L.
        """
        data = sf_db.get("TBP")
        assert "nitrate_M" in data.conditions
        assert data.derived
        assert data.adjacent_pairs["Ce_La"] == pytest.approx(1.38, abs=0.05)

    def test_the_y_dy_pair_follows_the_correlations(self, sf_db):
        """The pair the two descriptions disagreed about in *direction*.

        #265 settled it on the coefficients, and #270 finished it. Both of
        the descriptions #265 was choosing between put Y BELOW Dy on all
        three acidic extractants, and that is wrong on all three: Y(III) has
        no f electrons, so its place in the series is set by the donor, and
        every donor here puts it in the heavy group.

        The three numbers now come from three different printed statements
        about where Y belongs, not from one guess repeated:

          PC88A      3.14  Tanaka (2021), log Ke 2.24 for Y against 1.67 for
                           Dy -- Y between Dy and Ho.
          D2EHPA     4.01  Peppard (1957) via Stevenson & Nervik Fig. 51:
                           Y at apparent Z 67.4, between Ho and Er, and
                           Mason (1976) measures Y's level directly (#283;
                           3.69 under #270, on PPH63's shallower steps).
          Cyanex272  3.59  Zhang & Li (1993) Table 4.25, Y printed between
                           Ho and Er.

        The direction assertion at the bottom is the part that matters: it is
        the claim the pre-#270 records got backwards on two of three.
        """
        for extractant, expected in (("D2EHPA", 4.011),
                                     ("PC88A", 3.138),
                                     ("Cyanex272", 3.593)):
            got = sf_db.get_sf(extractant, "Y_Dy")
            assert got == pytest.approx(expected, rel=0.02)
        for extractant in ("D2EHPA", "PC88A", "Cyanex272"):
            assert sf_db.get_sf(extractant, "Y_Dy") > 1.0, extractant

    def test_the_pc88a_factors_do_not_move_with_the_conditions(self, sf_db):
        """One slope for every element makes beta exactly pH-independent.

        Since #270 PC88A's `b` is the stoichiometric 3 for all ten elements,
        so log10 beta = a_i - a_j and the pH, temperature and concentration
        terms cancel identically. This is what killed the "optimal pH = 5.0"
        artifact: with per-element slopes, beta drifted with pH and an
        optimizer chased it out of the fitted window.
        """
        from difflow_ree.equilibrium.distribution import get_separation_factor

        at_1 = float(get_separation_factor("Y", "Dy", "PC88A", pH=1.0,
                                           T=298.0, concentration=0.5))
        at_2 = float(get_separation_factor("Y", "Dy", "PC88A", pH=2.0,
                                           T=298.0, concentration=0.2))
        assert at_1 == pytest.approx(at_2, rel=1e-12)

    def test_stages_for_99_purity_is_derived_from_the_factor(self, sf_db):
        """The 18-integer table went the way the factors went (#270)."""
        import math

        for extractant in ("D2EHPA", "PC88A"):
            for pair, beta in sf_db.get(extractant).adjacent_pairs.items():
                got = sf_db.get_stages_needed(extractant, pair)
                assert got == math.ceil(2 * math.log(99) / abs(math.log(beta)))
        # It is a floor, and a floor that moves the right way: the harder
        # pair needs more stages than the easy one.
        assert (sf_db.get_stages_needed("PC88A", "Ce_La")
                > sf_db.get_stages_needed("PC88A", "Sm_Nd"))

    def test_an_authored_stage_count_still_wins(self, sf_db):
        from difflow_ree.database import SeparationFactorDatabase

        db = SeparationFactorDatabase()
        derived = db.get_stages_needed("PC88A", "Nd_Pr")
        db.remove_pair("PC88A", "Nd_Pr")
        db.add_pair("PC88A", "Nd_Pr", 2.142, adjacent=True, stages_99=derived + 40)
        assert db.get_stages_needed("PC88A", "Nd_Pr") == derived + 40


class TestAuthoredOverrides:
    def test_a_mapping_is_taken_as_given(self, tmp_path):
        override = tmp_path / "sf.yaml"
        override.write_text(
            "separation_factors:\n"
            "  D2EHPA:\n"
            "    conditions: {pH: 1.0, temperature_K: 298, concentration_M: 0.5}\n"
            "    adjacent_pairs: {Nd_Pr: 1.23}\n"
            "    group_pairs: [Dy_Nd]\n"
        )
        db = SeparationFactorDatabase(override)
        data = db.get("D2EHPA")
        assert data.adjacent_pairs["Nd_Pr"] == 1.23
        assert "Nd_Pr" not in data.derived
        # and the list entry beside it is still derived
        assert "Dy_Nd" in data.derived
        assert data.group_pairs["Dy_Nd"] == pytest.approx(
            float(get_separation_factor("Dy", "Nd", "D2EHPA", pH=1.0, T=298.0,
                                        concentration=0.5)))

    def test_an_added_pair_is_authored_not_derived(self, sf_db):
        db = SeparationFactorDatabase()
        db.add_pair("PC88A", "Ho_Dy", 1.4, adjacent=True, stages_99=20)
        assert db.get_sf("PC88A", "Ho_Dy") == 1.4
        assert "Ho_Dy" not in db.get("PC88A").derived

    def test_a_malformed_pair_name_says_so(self, tmp_path):
        bad = tmp_path / "sf.yaml"
        bad.write_text(
            "separation_factors:\n"
            "  D2EHPA:\n"
            "    conditions: {pH: 3.0, temperature_K: 298, concentration_M: 0.5}\n"
            "    adjacent_pairs: [NdPr]\n"
            "    group_pairs: []\n"
        )
        with pytest.raises(ValueError, match="heavier.*lighter|NdPr"):
            SeparationFactorDatabase(bad)


def test_conditions_are_load_bearing(sf_db):
    """The declared conditions are where the coefficients get evaluated.

    They used to change the FACTORS -- the pre-#270 records carried a
    different `b` for every element, so beta drifted with pH and a block
    declaring pH 3.0 reported different numbers from one declaring 1.5. They
    no longer do: all four acidic records now share b = 3, the stoichiometric
    slope of `RE3+ + 3 (HA)2 <-> RE(HA2)3 + 3 H+`, so log10 beta = a_i - a_j
    and every conditions term cancels. That is a stronger property, not a
    weaker one, and it is asserted here as such.

    What the conditions still bear is D itself, and whether the evaluation
    point is inside the record's fitted window at all. Both are checked.
    """
    data = sf_db.get("D2EHPA")
    at_declared = data.adjacent_pairs["Nd_Pr"]
    elsewhere = float(get_separation_factor("Nd", "Pr", "D2EHPA", pH=1.5,
                                            T=298.0, concentration=0.5))
    assert at_declared == pytest.approx(elsewhere, rel=1e-9)

    # D does move, by three decades a pH unit, which is why the block has to
    # declare a pH at all.
    from difflow_ree.equilibrium.distribution import REEDistribution

    dist = REEDistribution(extractant="D2EHPA", elements=("Nd",),
                           on_out_of_range="ignore")
    lo = float(dist.get_D("Nd", pH=0.82))
    hi = float(dist.get_D("Nd", pH=1.82))
    assert hi / lo == pytest.approx(1000.0, rel=1e-6)

    # And every declared point is inside the record it is evaluated against,
    # which is what #270 had to move three of these four blocks to restore.
    from difflow_ree.database import get_extractant

    for extractant in sf_db.list_extractants():
        pH = sf_db.get(extractant).conditions.get("pH")
        if pH is None:            # TBP is solvating; it declares nitrate
            continue
        low, high = get_extractant(extractant).valid_ph_range
        assert low <= pH <= high, f"{extractant} declares pH {pH}, window "\
                                  f"[{low}, {high}]"
