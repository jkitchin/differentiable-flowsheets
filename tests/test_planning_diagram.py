"""Smoke tests for the planning drawings."""

import matplotlib
import pytest

matplotlib.use("Agg")           # no display in CI; must precede pyplot

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from difflow.planning import (
    Block, DeltaBasePlanner, Network, draw_chain, draw_delta_vectors,
    draw_planning_network, draw_taylor_model, draw_trust_region, linearize_block,
)
from difflow.planning import chain as chain_mod


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_draw_chain_and_network():
    problem = chain_mod.two_plant_chain()
    state = problem.network.evaluate(problem.network.decision_start()).as_dict()
    assert draw_chain(state, prices=problem.prices, specs=problem.specs).texts
    assert draw_chain().texts
    ax = draw_planning_network(problem.network, prices=problem.prices,
                               specs=problem.specs, values=state)
    assert ax.texts


def test_draw_delta_vectors_and_taylor_model():
    block = chain_mod.ngl_block()
    lin = linearize_block(block)
    ax = draw_delta_vectors(lin, block=block)
    assert len(ax.texts) == lin.J.shape[0] * lin.J.shape[1]
    assert draw_taylor_model(block, "T_coldbox", "residue_F", radius=0.25).lines
    with pytest.raises(KeyError):
        draw_taylor_model(block, "nope", "residue_F")


def test_draw_trust_region_smoke():
    bowl = Block(name="q", fn=lambda u: jnp.array([-(u[0] - 0.3) ** 2 - (u[1] - 0.7) ** 2]),
                 u_names=["x", "y"], y_names=["f"], lb=[0.0, 0.0], ub=[1.0, 1.0])
    result = DeltaBasePlanner(Network([bowl]), prices={"q.f": 1.0},
                              radius=0.3, vertex_seeding=False).solve()
    ax = draw_trust_region(result, grid=5, max_cycles=2)
    assert ax.get_xlabel() == "q.x" and ax.get_ylabel() == "q.y"
