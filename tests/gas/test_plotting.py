"""Smoke tests for the network schematic.

A drawing cannot be asserted correct, so these check the things that
can be: that it runs on every arc kind, that it puts something on the
axes, that the annotations it is given reach the figure as text, and
that a bad layout fails loudly rather than silently dropping a node.
"""

import matplotlib
import pytest

matplotlib.use("Agg")           # no display in CI; must precede pyplot

import matplotlib.pyplot as plt

import difflow_gas as dg
from difflow_gas.plotting import circular_positions, draw_network
from tests.gas.test_network import mixed_network, triangle


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _texts(ax):
    return {t.get_text() for t in ax.texts}


class TestDrawNetwork:
    def test_draws_every_arc_kind(self):
        """mixed_network has pipe, resistor, valve, short pipe, cv, compressor."""
        net = mixed_network()
        ax = draw_network(net)
        assert ax.collections, "no node markers drawn"
        for node in net.nodes:
            assert node in _texts(ax), f"node {node} unlabelled"
        for arc in net.arcs:
            assert any(arc in t for t in _texts(ax)), f"arc {arc} unlabelled"

    def test_annotations_reach_the_figure(self):
        net = triangle()
        ax = draw_network(
            net,
            pos={"n0": (0.0, 0.0), "n1": (1.0, 0.0), "n2": (0.5, 1.0)},
            pressures={"n0": 50.0, "n1": 49.96, "n2": 49.95},
            flows={"p01": 18.76, "p12": 8.76, "p02": 11.24},
            title="triangle",
        )
        joined = " ".join(_texts(ax))
        assert "50.0 bar" in joined
        assert "18.8 kg/s" in joined
        # the title is set with loc="left", which matplotlib stores
        # separately from the centred one
        assert ax.get_title(loc="left") == "triangle"

    def test_highlight_is_labelled_not_colour_alone(self):
        """A flagged arc keeps its text label, so colour is never the only cue."""
        net = triangle()
        ax = draw_network(net, highlight=["p01"])
        assert any("p01" in t for t in _texts(ax))

    def test_missing_position_raises(self):
        net = triangle()
        with pytest.raises(ValueError, match="no position given"):
            draw_network(net, pos={"n0": (0.0, 0.0)})

    def test_supplies_override(self):
        """Measured nominations can be shown instead of the network's."""
        net = triangle()
        ax = draw_network(net, supplies={"n0": 31.5, "n1": -9.8, "n2": -20.4})
        joined = " ".join(_texts(ax))
        assert "+32 kg/s" in joined or "+31 kg/s" in joined

    def test_accepts_an_existing_axes(self):
        fig, axes = plt.subplots(1, 2)
        out = draw_network(triangle(), ax=axes[1])
        assert out is axes[1]
        assert not axes[0].collections, "drew on the wrong axes"

    def test_legend_can_be_suppressed(self):
        ax = draw_network(triangle(), legend=False)
        assert ax.get_legend() is None


class TestCircularPositions:
    def test_one_position_per_node_on_the_unit_circle(self):
        nodes = ["a", "b", "c", "d"]
        pos = circular_positions(nodes)
        assert set(pos) == set(nodes)
        for x, y in pos.values():
            assert (x * x + y * y) == pytest.approx(1.0)

    def test_empty_is_not_a_division_by_zero(self):
        assert circular_positions([]) == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
