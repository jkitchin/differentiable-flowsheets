"""SeparationFactorDatabase: authored overrides and malformed pair names (#265)."""

import pytest

from difflow_ree.database import SeparationFactorDatabase


def test_a_mapping_is_taken_as_given(tmp_path):
    override = tmp_path / "sf.yaml"
    override.write_text(
        "separation_factors:\n"
        "  D2EHPA:\n"
        "    conditions: {pH: 1.0, temperature_K: 298, concentration_M: 0.5}\n"
        "    adjacent_pairs: {Nd_Pr: 1.23}\n"
        "    group_pairs: [Dy_Nd]\n"
    )
    data = SeparationFactorDatabase(override).get("D2EHPA")
    assert data.adjacent_pairs["Nd_Pr"] == 1.23
    assert "Nd_Pr" not in data.derived and "Dy_Nd" in data.derived


def test_a_malformed_pair_name_says_so(tmp_path):
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
