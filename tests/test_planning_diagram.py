"""Smoke tests for the planning drawings.

A drawing cannot be asserted correct, so these check the things that
can be: that every unit, decision, link and priced stream the caller
passed in reaches the figure as text, that the annotations track the
state they are given, and that the trust-region picture shows the
cycles the run actually took.
"""

import matplotlib
import pytest

matplotlib.use("Agg")           # no display in CI; must precede pyplot

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from difflow.planning import (
    Block, DeltaBasePlanner, Network, PhaseBoundaryWarning, draw_chain,
    draw_delta_vectors, draw_planning_network, draw_taylor_model,
    draw_trust_region, linearize_block,
)
from difflow.planning import chain as chain_mod


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def problem():
    return chain_mod.two_plant_chain()


def _texts(ax):
    return [t.get_text() for t in ax.texts]


class TestDrawChain:

    def test_labels_every_unit_lever_and_priced_stream(self, problem):
        state = problem.network.evaluate(problem.network.decision_start())
        ax = draw_chain(state.as_dict(), prices=problem.prices,
                        specs=problem.specs)
        text = "\n".join(_texts(ax))
        for unit in ("cold box", "reflux\ncontactor", "deethanizer",
                     "gas turbine", "turbo-\nexpander"):
            assert unit in text, f"unit {unit!r} unlabelled"
        for decision in problem.network.decision_names:
            assert decision.split(".", 1)[1] in text
        for priced in problem.prices:
            assert priced.split(".", 1)[1] in text
        assert "residue_F" in text            # the link between the plants
        assert "236" in text                  # the spec, from problem.specs

    def test_annotations_follow_the_state(self, problem):
        net = problem.network
        hot = net.evaluate(net.decision_array(
            {"ngl.ethane_recovery": 0.95, "ngl.T_coldbox": 240.0,
             "ngl.split": 0.2, "ngl.P_expander": 3.0e6,
             "power.alloc": 1.0}))
        ax = draw_chain(hot.as_dict())
        text = "\n".join(_texts(ax))
        assert "240" in text and "0.95" in text
        assert f"{hot['ngl.residue_F']:.3g}" in text

    def test_draws_without_a_state(self, problem):
        ax = draw_chain()
        assert "=" not in "".join(t for t in _texts(ax) if "->" not in t)


class TestDrawPlanningNetwork:

    def test_labels_blocks_decisions_links_and_specs(self, problem):
        state = problem.network.evaluate(problem.network.decision_start())
        ax = draw_planning_network(problem.network, prices=problem.prices,
                                   specs=problem.specs,
                                   values=state.as_dict())
        text = "\n".join(_texts(ax))
        for block in problem.network.blocks:
            assert block.name in text
            for name in block.u_names + block.y_names:
                assert name in text, f"{block.name}.{name} unlabelled"
        assert "link" in text
        assert "spec <= 236" in text

    def test_handles_a_single_block(self):
        ax = draw_planning_network(Network([chain_mod.power_block()]))
        assert "power" in "\n".join(_texts(ax))


class TestDrawDeltaVectors:

    def test_prints_every_entry_of_J(self):
        block = chain_mod.ngl_block()
        lin = linearize_block(block)
        ax = draw_delta_vectors(lin, block=block)
        assert [t.get_text() for t in ax.get_xticklabels()] == block.u_names
        assert [t.get_text() for t in ax.get_yticklabels()] == block.y_names
        assert len(_texts(ax)) == lin.J.shape[0] * lin.J.shape[1]
        title = ax.get_title(loc="left")
        assert lin.block in title and lin.mode in title


class TestDrawTaylorModel:

    def test_model_and_block_agree_at_the_linearisation_point(self):
        block = chain_mod.ngl_block()
        ax = draw_taylor_model(block, "T_coldbox", "residue_F", radius=0.25)
        grid = ax.lines[0].get_xdata()
        truth, model = ax.lines[0].get_ydata(), ax.lines[1].get_ydata()
        centre = int(abs(grid - float(block.u0[block.u_index("T_coldbox")]))
                     .argmin())
        assert abs(truth[centre] - model[centre]) < 1e-8
        # ... and part company away from it, which is the point of the figure
        assert abs(truth[0] - model[0]) > 1e-3
        assert "trust region" in "\n".join(_texts(ax))

    def test_rejects_an_unknown_variable(self):
        with pytest.raises(KeyError):
            draw_taylor_model(chain_mod.ngl_block(), "nope", "residue_F")


class TestDrawTrustRegion:

    def test_shows_the_cycles_the_run_took(self):
        bowl = Block(name="q",
                     fn=lambda u: jnp.array([-(u[0] - 0.3) ** 2
                                             - (u[1] - 0.7) ** 2]),
                     u_names=["x", "y"], y_names=["f"],
                     lb=[0.0, 0.0], ub=[1.0, 1.0])
        result = DeltaBasePlanner(Network([bowl]), prices={"q.f": 1.0},
                                  radius=0.3, vertex_seeding=False).solve()
        ax = draw_trust_region(result, grid=9, max_cycles=4)
        assert ax.get_xlabel() == "q.x" and ax.get_ylabel() == "q.y"
        assert len(ax.patches) <= 4, "more boxes drawn than cycles requested"
        assert "first 4 drawn" in ax.get_title(loc="left")
        rhos = [f"rho = {it.rho:.2f}" for it in result.history[:4]]
        text = "\n".join(_texts(ax))
        for rho in rhos:
            assert rho in text

    def test_accepts_decisions_by_name(self, problem):
        with pytest.warns(PhaseBoundaryWarning):
            result = problem.planner(radius=0.25).solve()
        ax = draw_trust_region(result, grid=5,
                               decisions=("power.alloc",
                                          "ngl.ethane_recovery"),
                               max_cycles=2)
        assert ax.get_xlabel() == "power.alloc"
