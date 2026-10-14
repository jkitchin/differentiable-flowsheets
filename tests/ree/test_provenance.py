"""The data provenance registry: every field tagged, every source registered."""

import pytest

from difflow_ree.provenance import (
    DATASETS,
    coverage,
    explain,
    load_rules,
    load_sources,
    untagged,
)


def test_every_rule_cites_a_registered_source_and_nothing_is_untagged():
    sources = load_sources()
    for dataset in DATASETS:
        for pattern, rule in load_rules(dataset):
            assert rule["source"] in sources, (dataset, pattern)
        assert not untagged(dataset), dataset


def test_explain_and_coverage_smoke():
    p = explain("elements", "Dy.ionic_radius_pm")
    assert p.source in load_sources()
    assert sum(coverage().values()) > 0


def test_unknown_path_raises():
    with pytest.raises(KeyError):
        explain("elements", "elements.Nd.no_such_field")
