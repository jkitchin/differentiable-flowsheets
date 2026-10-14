"""Smoke tests for difflow_power.plotting."""

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


def test_draw_network_smoke():
    net = dp.cases.case9()
    ax = dp.draw_network(net, loading={"br3": 1.4})
    dp.draw_legend(ax)
    assert ax.get_title() == net.name
    assert ax.get_legend() is not None
    assert len(dp.tree_positions(dp.cases.radial_feeder())) > 0


def test_missing_positions_are_a_clear_error():
    net = dp.cases.case9()
    with pytest.raises(KeyError, match="no position given"):
        dp.draw_network(net, pos={"1": (0.0, 0.0)})


def test_importing_the_plugin_does_not_import_matplotlib():
    source = importlib.import_module("difflow_power.plotting").__file__
    with open(source) as handle:
        head = handle.read().split("def circular_positions")[0]
    assert "import matplotlib" not in head
