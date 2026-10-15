"""Smoke tests for the hydrotreater-product adapters to the reformer and the gas plant (#327)."""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import BlendCharacterization
from difflow_refinery.composition import CompositionRangeWarning
from difflow_refinery.gasplant.hydroprocessed import hydroprocessed_feed, resolve_product
from difflow_refinery.reforming import NaphthaFeed

jax.config.update("jax_enable_x64", True)

C = 273.15


@pytest.fixture(scope="module")
def grid():
    """A treated-like grid and a stream on it."""
    assay = dr.Assay([0, 10, 30, 50, 70, 90, 100], [t + C for t in [30, 60, 95, 120, 145, 170, 190]], sg=0.74,
                     light_ends={"isopentane": 2.0, "n_pentane": 2.5}, sulfur_wt=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositionRangeWarning)
        char = dr.characterize(assay, [C + t for t in (60, 100, 140, 180)], composition=True)
    g = BlendCharacterization.from_characterization(char)
    F = jnp.asarray(char.mole_fraction) * 200.0
    stream = {f"F_{n}": F[i] for i, n in enumerate(char.names)}
    stream.update(T=jnp.asarray(320.0), P=jnp.asarray(4e5))
    return g, stream


def test_naphtha_feed_from_hydrotreater(grid):
    g, stream = grid
    f = NaphthaFeed.from_hydrotreater(stream, char=g)
    assert np.all(np.isfinite(np.asarray(f.flows)))
    assert float(f.sulfur_wppm) > 0.0


def test_dissolved_gases_are_refused(grid):
    g, stream = grid
    with pytest.raises(ValueError, match="drop_gases"):
        NaphthaFeed.from_hydrotreater(dict(stream, F_hydrogen=jnp.asarray(0.5)), char=g)


def test_a_grid_without_types_is_refused(grid):
    g, stream = grid
    bare = dataclasses.replace(g, qualities={"S_ppm": g.qualities["S_ppm"]})
    with pytest.raises(ValueError, match="hydrocarbon types"):
        NaphthaFeed.from_hydrotreater(stream, char=bare)


def test_hydroprocessed_feed(grid):
    g, stream = grid
    wild = dict(stream, F_hydrogen=jnp.asarray(0.05), F_ammonia=jnp.asarray(0.001))
    with pytest.raises(ValueError, match="ammonia"):
        hydroprocessed_feed(wild, char=g)
    feed = hydroprocessed_feed(wild, char=g, unsupported="drop")
    assert set(feed.dropped) == {"ammonia"}
    assert np.isfinite(float(feed.mass_flow))


def test_resolve_refuses_what_it_cannot_read():
    with pytest.raises(ValueError, match="char="):
        resolve_product({"F_x": 1.0})
