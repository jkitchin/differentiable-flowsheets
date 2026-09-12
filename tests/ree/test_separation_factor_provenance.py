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

        Both said Y extracts less readily than Dy; they differed by 4-8x on
        how much. The correlations now answer it everywhere.
        """
        for extractant, expected in (("D2EHPA", 0.214),
                                     ("PC88A", 0.140),
                                     ("Cyanex272", 0.100)):
            got = sf_db.get_sf(extractant, "Y_Dy")
            assert got < 1.0
            assert got == pytest.approx(expected, rel=0.02)


class TestAuthoredOverrides:
    def test_a_mapping_is_taken_as_given(self, tmp_path):
        override = tmp_path / "sf.yaml"
        override.write_text(
            "separation_factors:\n"
            "  D2EHPA:\n"
            "    conditions: {pH: 3.0, temperature_K: 298, concentration_M: 0.5}\n"
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
            float(get_separation_factor("Dy", "Nd", "D2EHPA", pH=3.0, T=298.0,
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
    """Changing the declared pH changes the factors, which is the point."""
    data = sf_db.get("D2EHPA")
    at_declared = data.adjacent_pairs["Nd_Pr"]
    elsewhere = float(get_separation_factor("Nd", "Pr", "D2EHPA", pH=1.5,
                                            T=298.0, concentration=0.5))
    assert at_declared != pytest.approx(elsewhere, rel=1e-6)
