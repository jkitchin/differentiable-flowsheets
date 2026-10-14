"""Refinery C4-C8 olefins, naphthenes and branched paraffins in the database (#308-#310)."""

import jax.numpy as jnp

from difflow.database import (
    SOURCE_CITATIONS,
    get_critical_props,
    get_species_data,
    list_species,
    resolve_alias,
)
from difflow.eos import PengRobinson

NAMES = [
    "cis_2_butene", "trans_2_butene", "isobutylene", "1_pentene", "2_methyl_2_butene",
    "cyclohexane", "methylcyclopentane", "2_methylpentane", "3_methylpentane",
    "2_2_dimethylbutane", "2_3_dimethylbutane", "2_2_4_trimethylpentane",
    "2_3_4_trimethylpentane", "2_5_dimethylhexane",
]


def test_loads_from_both_tables():
    listed = set(list_species())
    for name in NAMES:
        crit = get_critical_props(name)
        data = get_species_data(name)
        assert crit.name == name and data.name == name
        assert name in listed and name in SOURCE_CITATIONS


def test_mixture_eos_builds_and_is_finite():
    eos = PengRobinson({n: get_critical_props(n) for n in NAMES})
    y = jnp.ones(len(NAMES)) / len(NAMES)
    assert bool(jnp.isfinite(eos.solve_Z(jnp.asarray(400.0), jnp.asarray(2.0e5), y, "vapor")))


def test_aliases():
    for alias, name in [("isooctane", "2_2_4_trimethylpentane"), ("isobutene", "isobutylene"),
                        ("neohexane", "2_2_dimethylbutane")]:
        assert resolve_alias(alias) == name
        assert get_species_data(alias).name == name
