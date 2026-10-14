"""Antoine database contracts: every species has a source, list_species API."""

import pytest

from difflow.database import (
    ANTOINE_SOURCES, _IDEAL_THERMO_DATA, get_critical_props, get_species_data,
    list_species,
)


def test_sources_cover_exactly_the_ideal_species():
    assert set(ANTOINE_SOURCES) == set(_IDEAL_THERMO_DATA)


class TestListSpecies:
    def test_default_lists_only_names_get_species_data_accepts(self):
        for s in list_species():
            get_species_data(s)  # must not raise

    def test_critical_names_all_have_critical_props(self):
        for s in list_species("critical"):
            get_critical_props(s)

    def test_any_is_the_union(self):
        assert set(list_species("any")) == (set(list_species())
                                            | set(list_species("critical")))

    def test_bad_kind(self):
        with pytest.raises(ValueError):
            list_species("bogus")
