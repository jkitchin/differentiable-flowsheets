"""Tests for difflow_power.plotting.

A drawing cannot be asserted on, so these tests check what can be:
that every network draws without error, that annotations reach the
axes, that a missing position is a clear error rather than a silent
omission, and that importing the package does not require matplotlib.
"""

import importlib

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")           # no display in CI; must precede pyplot
import matplotlib.pyplot as plt

import difflow_power as dp


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("case", sorted(dp.cases.CASES))
def test_every_case_draws(case):
    net = dp.cases.load_case(case)
    ax = dp.draw_network(net)
    assert ax.get_title() == net.name
    # One marker line per bus, one line per branch.
    assert len(ax.lines) >= net.n_bus + net.n_branch


def test_annotations_reach_the_axes():
    net = dp.cases.case9()
    result = dp.solve_power_flow(net)
    ax = dp.draw_network(
        net,
        pos=dp.circular_positions(net.bus_ids),
        voltages=result.vm,
        flows={a: v[0] for a, v in result.branch_mw.items()},
        loading=result.branch_loading,
    )
    texts = [t.get_text() for t in ax.texts]
    assert any("pu" in t for t in texts)
    assert any("MW" in t for t in texts)
    assert all(bus in texts for bus in net.bus_ids)


def test_prices_and_highlights_are_drawn():
    net = dp.cases.case5()
    result = dp.solve_acopf(net)
    ax = dp.draw_network(
        net, prices=result.lmp_mw, highlight=["4", "br6"]
    )
    assert any("$" in t.get_text() for t in ax.texts)


def test_an_overloaded_branch_is_flagged():
    net = dp.cases.case9()
    ax = dp.draw_network(net, loading={"br3": 1.4})
    colours = {tuple(line.get_color()) if not isinstance(
        line.get_color(), str) else line.get_color() for line in ax.lines}
    assert "#d03b3b" in colours


def test_missing_positions_are_a_clear_error():
    net = dp.cases.case9()
    with pytest.raises(KeyError, match="no position given"):
        dp.draw_network(net, pos={"1": (0.0, 0.0)})


def test_tree_layout_places_the_root_at_the_left():
    net = dp.cases.radial_feeder()
    pos = dp.tree_positions(net)
    assert pos["s"][0] == 0.0
    assert pos["n4"][0] > pos["n1"][0]
    # A meshed network has no tree, so it falls back to a circle.
    circle = dp.tree_positions(dp.cases.case9())
    assert len(circle) == 9


def test_legend_draws():
    ax = dp.draw_network(dp.cases.case3())
    dp.draw_legend(ax)
    assert ax.get_legend() is not None


def test_importing_the_plugin_does_not_import_matplotlib():
    """matplotlib is imported inside the drawing functions, not at
    module scope, so the plugin is usable without it."""
    source = (
        importlib.import_module("difflow_power.plotting").__file__
    )
    with open(source) as handle:
        head = handle.read().split("def circular_positions")[0]
    assert "import matplotlib" not in head
