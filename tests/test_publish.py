"""Tests for difflow.publish.

A published page is precomputed, so the thing to check is that the grid
it carries is the flowsheet's real answer -- the page can only be as
right as the numbers baked into it.
"""

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (
    CSTR,
    CSTRParams,
    Flowsheet,
    SweepAxis,
    Unit,
    get_flows,
    make_stream,
    mass_action_kinetics,
)
from difflow.publish import publish, sweep, to_html

SPECIES = ["A", "B"]


def payload_of(page: str) -> dict:
    """What the page hands its front end, read back out of the file."""
    body = page.split("window.DIFFLOW = ", 1)[1].split(";</script>", 1)[0]
    return json.loads(body)


@pytest.fixture(scope="module")
def flowsheet():
    kin = mass_action_kinetics([{
        "equation": "A -> B",
        "reactants": {"A": 1.0}, "products": {"B": 1.0},
        "rate_params": {"A": 1.0e6, "Ea": 50_000.0, "n": 0.0},
    }], SPECIES)
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0))
    fs.add_unit(Unit("reactor", CSTR(CSTRParams(
        V=1.0, molar_density=1000.0, **kin.params_kwargs()
    )), ["feed"], ["out"]))
    return fs


def product(streams):
    return get_flows(streams["out"])["B"]


# =============================================================================
# The sweep
# =============================================================================


class TestSweep:
    def test_grid_matches_direct_evaluation(self, flowsheet):
        """The page can only be as right as the numbers baked into it."""
        axis = SweepAxis("reactor.V", 0.5, 3.0, n=6)
        result = sweep(flowsheet, [axis], {"product": product})

        objective = flowsheet.make_objective_fn(product)
        for i, v in enumerate(axis.values()):
            expected = float(objective({"reactor.V": float(v)}))
            assert float(result.values["product"][i]) == pytest.approx(
                expected, rel=1e-12
            )

    def test_gradients_are_the_real_derivatives(self, flowsheet):
        axis = SweepAxis("reactor.V", 0.5, 3.0, n=4)
        result = sweep(flowsheet, [axis], {"product": product})

        objective = flowsheet.make_objective_fn(product)
        for i, v in enumerate(axis.values()):
            expected = float(
                jax.grad(objective)({"reactor.V": float(v)})["reactor.V"]
            )
            got = float(result.gradients["product"]["reactor.V"][i])
            assert got == pytest.approx(expected, rel=1e-9)

    def test_two_axes_give_a_two_dimensional_grid(self, flowsheet):
        result = sweep(
            flowsheet,
            [SweepAxis("reactor.V", 0.5, 3.0, n=5),
             SweepAxis("reactor.T_damping", 0.2, 0.6, n=3)],
            {"product": product},
        )
        assert result.shape == (5, 3)
        assert result.n_points == 15
        assert jnp.asarray(result.values["product"]).shape == (5, 3)

    def test_the_loop_fallback_agrees_with_vmap(self, flowsheet):
        axis = SweepAxis("reactor.V", 0.5, 3.0, n=4)
        batched = sweep(flowsheet, [axis], {"product": product}, batch=True)
        looped = sweep(flowsheet, [axis], {"product": product}, batch=False)
        np.testing.assert_allclose(
            np.asarray(batched.values["product"]),
            np.asarray(looped.values["product"]),
            rtol=1e-12,
        )

    def test_gradients_can_be_skipped(self, flowsheet):
        result = sweep(
            flowsheet, [SweepAxis("reactor.V", 0.5, 3.0, n=3)],
            {"product": product}, gradients=False,
        )
        assert result.gradients == {}

    def test_multiple_outputs(self, flowsheet):
        result = sweep(
            flowsheet, [SweepAxis("reactor.V", 0.5, 3.0, n=3)],
            {"B": product, "A": lambda s: get_flows(s["out"])["A"]},
        )
        assert set(result.values) == {"A", "B"}
        # a first-order conversion: what leaves as B did not leave as A
        total = (jnp.asarray(result.values["A"])
                 + jnp.asarray(result.values["B"]))
        np.testing.assert_allclose(np.asarray(total), 1.0, rtol=1e-9)

    def test_a_bigger_volume_converts_more(self, flowsheet):
        result = sweep(
            flowsheet, [SweepAxis("reactor.V", 0.5, 5.0, n=8)],
            {"product": product},
        )
        values = np.asarray(result.values["product"])
        assert np.all(np.diff(values) > 0), "conversion must rise with volume"


class TestSweepValidation:
    def test_no_axes_is_refused(self, flowsheet):
        with pytest.raises(ValueError, match="at least one axis"):
            sweep(flowsheet, [], {"product": product})

    def test_no_outputs_is_refused(self, flowsheet):
        with pytest.raises(ValueError, match="at least one output"):
            sweep(flowsheet, [SweepAxis("reactor.V", 1.0, 2.0)], {})

    def test_a_degenerate_axis_is_refused(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            SweepAxis("reactor.V", 1.0, 2.0, n=1)
        with pytest.raises(ValueError, match="hi must exceed lo"):
            SweepAxis("reactor.V", 2.0, 1.0)


# =============================================================================
# The page
# =============================================================================


@pytest.fixture(scope="module")
def sweep_result(flowsheet):
    return sweep(
        flowsheet,
        [SweepAxis("reactor.V", 0.5, 3.0, n=5, label="Volume", units="m^3")],
        {"Product": product}, units={"Product": "mol/s"},
    )


@pytest.fixture(scope="module")
def page(sweep_result):
    return to_html(sweep_result, title="Test model",
                   description="A description.")


@pytest.fixture(scope="module")
def drawn_page(sweep_result, flowsheet):
    """The same page, published with the flowsheet it came from."""
    return to_html(sweep_result, title="Test model", flowsheet=flowsheet)


#: URLs the page carries as text and never fetches. XML namespaces are
#: names that happen to look like addresses, and the rest are the
#: documentation links Svelte and @xyflow put in their error messages.
#: Anything outside this list is a page that reaches out.
INERT_URLS = (
    "http://www.w3.org/",
    "https://svelte.dev/e/",
    "flow.dev/",
    "https://svelteflow.dev",
)


class TestPage:
    def test_loads_nothing_at_run_time(self, drawn_page):
        """No network at run time: nothing to rot next to a paper."""
        for pattern in ("<script src", "<link ", "@import url(", "url(http",
                        ' src="http', ' href="http'):
            assert pattern not in drawn_page, f"page fetches via {pattern!r}"

    def test_every_url_it_names_is_inert(self, drawn_page):
        """The bundle carries URLs; none of them is a resource.

        Stricter than the absence of ``http``, which the front-end
        libraries make impossible: every URL in the file is accounted
        for by name, so a new one has to be looked at.
        """
        for match in re.finditer(r"""https?://[^\s"'`)\\]*""", drawn_page):
            url = match.group(0)
            assert any(known in url for known in INERT_URLS), \
                f"page names {url!r}"

    def test_carries_the_grid(self, page):
        assert '"Product"' in page
        assert "Volume" in page
        assert "m^3" in page

    def test_carries_the_topology(self, drawn_page, page):
        """The flowsheet, and what its units are, travel with the sweep."""
        data = payload_of(drawn_page)
        assert [u["name"] for u in data["topology"]["units"]] == ["reactor"]
        assert data["topology"]["units"][0]["operation"] == "CSTR"
        assert data["topology"]["feeds"]["feed"]["T"] == 350.0
        # Placed, so the canvas opens on a flowsheet rather than a pile.
        assert "reactor" in data["topology"]["view"]["nodes"]
        # And described, from the same catalog the editor reads.
        assert "CSTR" in data["catalog"]
        assert data["catalog"]["CSTR"]["description"]
        assert any(p["name"] == "V" for p in data["catalog"]["CSTR"]["parameters"])
        # Without a flowsheet the page is the sliders alone, as before.
        assert payload_of(page)["topology"] is None

    def test_a_parameter_it_cannot_show_is_named_rather_than_dropped(
        self, drawn_page
    ):
        """A CSTR holds more than numbers, and the page says so.

        A parameter that vanished would read as a parameter the unit
        does not have, which is worse than one shown as its own type.
        """
        params = payload_of(drawn_page)["topology"]["units"][0]["params"]
        assert params["V"] == 1.0
        assert params["rate_fn"] == "<function>"
        assert params["stoich"] == [[-1.0], [1.0]]
        # `mass_action_kinetics` writes an infinite equilibrium constant for
        # every irreversible reaction, and `JSON.parse` cannot read the word
        # Python spells it with.
        assert params["rate_params"]["K_eq"] == ["Infinity"]

    def test_has_the_interactive_pieces(self, page):
        for marker in ('id="sliders"', 'id="chart"', 'id="readout"',
                       'id="output-pick"'):
            assert marker in page, f"page has no {marker}"
        assert "window.DIFFLOW" in page

    def test_title_and_description_are_escaped(self, flowsheet):
        result = sweep(
            flowsheet, [SweepAxis("reactor.V", 0.5, 3.0, n=3)],
            {"p": product},
        )
        page = to_html(result, title="<script>x</script>", description="a & b")
        assert "<script>x</script>" not in page.split("<style>")[0]
        assert "&amp;" in page

    def test_a_name_cannot_end_the_script_element(self, flowsheet):
        """The payload carries names the user chose, not just numbers."""
        result = sweep(
            flowsheet, [SweepAxis("reactor.V", 0.5, 3.0, n=3)],
            {"</script><b>x": product},
        )
        page = to_html(result, title="t")
        assert "</script><b>x" not in page
        assert payload_of(page)["sweep"]["outputs"][0]["name"] == "</script><b>x"

    def test_a_title_cannot_impersonate_a_placeholder(self, flowsheet):
        """The template is filled in one pass, so a title stays a title."""
        result = sweep(
            flowsheet, [SweepAxis("reactor.V", 0.5, 3.0, n=3)], {"p": product},
        )
        page = to_html(result, title="__SCRIPT__", description="__STYLE__")
        assert page.count("<h1>__SCRIPT__</h1>") == 1
        assert '<p class="lede">__STYLE__</p>' in page

    def test_paints_its_own_background(self, page):
        """A standalone page cannot inherit a host's colours."""
        assert "background:var(--surface)" in page
        assert "--surface:" in page

    def test_records_provenance(self, page):
        assert "Precomputed with difflow" in page
        assert "solved operating" in page   # wraps in the template


class TestBundle:
    def test_a_missing_bundle_is_said_out_loud(self, sweep_result, monkeypatch):
        """Better than a blank page from a checkout that never built."""
        import difflow.gui.server as server

        monkeypatch.setattr(server, "STATIC", Path("/nonexistent"))
        with pytest.raises(FileNotFoundError, match="gui-build"):
            to_html(sweep_result)


class TestPublish:
    def test_writes_a_file(self, flowsheet, tmp_path):
        path = publish(
            flowsheet,
            [SweepAxis("reactor.V", 0.5, 3.0, n=4)],
            {"Product": product},
            tmp_path / "model.html",
            title="Reactor",
        )
        assert path.exists()
        text = path.read_text()
        assert text.startswith("<!doctype html>")
        assert "Reactor" in text

    def test_the_payload_is_valid_json(self, flowsheet):
        result = sweep(
            flowsheet, [SweepAxis("reactor.V", 0.5, 3.0, n=4)],
            {"Product": product},
        )
        payload = result.to_dict()
        restored = json.loads(json.dumps(payload))
        assert restored["axes"][0]["n"] == 4
        assert len(restored["outputs"][0]["values"]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
