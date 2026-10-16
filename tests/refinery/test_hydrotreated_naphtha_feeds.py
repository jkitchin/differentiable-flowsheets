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


@pytest.fixture(scope="module")
def grid_with_hydrogen(grid):
    """The fixture grid with each component's hydrogen (wt%), as a hydrotreater grid carries it."""
    g, stream = grid
    assay = dr.Assay([0, 10, 30, 50, 70, 90, 100], [t + C for t in [30, 60, 95, 120, 145, 170, 190]], sg=0.74,
                     light_ends={"isopentane": 2.0, "n_pentane": 2.5}, sulfur_wt=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositionRangeWarning)
        char = dr.characterize(assay, [C + t for t in (60, 100, 140, 180)], composition=True)
    H = 100.0 * jnp.asarray(char.composition.hydrogen)
    return dataclasses.replace(g, qualities=dict(g.qualities, H_wt=H)), stream


def _feed_elements(f):
    from difflow_refinery.reforming import species as sp
    return (float(f.flows @ jnp.asarray(sp.ELEMENTS[0])), float(f.flows @ jnp.asarray(sp.ELEMENTS[1])))


def test_from_hydrotreater_conserves_mass_and_hydrogen(grid_with_hydrogen):
    g, stream = grid_with_hydrogen
    m = jnp.stack([jnp.asarray(stream[f"F_{n}"]) for n in g.names]) * jnp.asarray(g.MW) / 1000.0
    H_src = float(jnp.sum(m * g.qualities["H_wt"] / 100.0)) * 1000.0 / 1.00794          # mol/s of H
    S_mass = float(jnp.sum(m * g.qualities["S_ppm"])) * 1e-6
    f = NaphthaFeed.from_hydrotreater(stream, char=g)
    C_f, H_f = _feed_elements(f)
    assert float(f.mass_flow) == pytest.approx(float(jnp.sum(m)), rel=1e-12)
    # cuts exactly; the fixture's light ends (characterization molar masses and hydrogen against
    # the reformer species') agree to the sixth figure
    assert H_f == pytest.approx(H_src, rel=2e-6)
    # carbon takes the rest of the mass, less the sulfur still on the cuts
    C_src = (float(jnp.sum(m)) - S_mass - H_src * 1.00794e-3) * 1000.0 / 12.0107
    assert C_f == pytest.approx(C_src, rel=2 * S_mass / float(jnp.sum(m)) + 1e-6)
    # without the match the model compounds' hydrogen differs from the grid's
    _, H_0 = _feed_elements(NaphthaFeed.from_hydrotreater(stream, char=g, match_hydrogen=False))
    assert abs(H_0 / H_src - 1.0) > 1e-3


def test_match_hydrogen_needs_the_quality(grid):
    g, stream = grid
    with pytest.raises(ValueError, match="H_wt"):
        NaphthaFeed.from_hydrotreater(stream, char=g, match_hydrogen=True)


def test_hydrogen_matched_feed_gradients_are_finite(grid_with_hydrogen):
    g, stream = grid_with_hydrogen
    names = [f"F_{n}" for n in g.names]
    x0 = jnp.stack([jnp.asarray(stream[k]) for k in names])

    def n2a(x, dH):
        s = dict(stream, **{k: x[i] for i, k in enumerate(names)})
        gg = dataclasses.replace(g, qualities=dict(g.qualities, H_wt=g.qualities["H_wt"] + dH))
        return NaphthaFeed.from_hydrotreater(s, char=gg).n_plus_2a()

    gx, gh = jax.grad(n2a, argnums=(0, 1))(x0, 0.0)
    _, jv = jax.jvp(lambda x: n2a(x, 0.0), (x0,), (jnp.ones_like(x0),))
    assert np.all(np.isfinite(np.asarray(gx))) and np.isfinite(float(gh)) and np.isfinite(float(jv))
    assert float(gh) < 0.0          # more hydrogen, fewer rings
