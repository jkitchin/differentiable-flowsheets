"""Smoke tests for the network schematic."""

import matplotlib
import pytest

matplotlib.use("Agg")           # no display in CI; must precede pyplot

import matplotlib.pyplot as plt

from difflow_gas.plotting import circular_positions, draw_network
from tests.gas.test_network import mixed_network, triangle


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_draw_network_smoke():
    net = mixed_network()
    ax = draw_network(net, pos=circular_positions(net.nodes))
    assert ax.collections


def test_missing_position_raises():
    with pytest.raises(ValueError, match="no position given"):
        draw_network(triangle(), pos={"n0": (0.0, 0.0)})
