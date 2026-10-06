"""difflow_bio cost of goods (#358): costs that follow the process.

The coarse model priced every chromatography step as Protein A, ran one
resin cycle per batch whatever the load, and charged fixed labor, so cost
per gram moved with titer and yield only through its denominator. These
tests pin what replaced it: per-step resins, cycles from load / (DBC x CV),
labor from batches and steps, buffers, QC and failed batches, all loaded
from data, all differentiable.

No published cost-of-goods breakdown is reproduced here yet; the numbers in
the shipped reference are placeholders and are tested only for internal
consistency and for agreement with the resin database they quote.
"""

import dataclasses
import json
import re
import warnings

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from difflow_bio.database import get_resin
from difflow_bio.economics import (
    CATEGORIES,
    REFERENCE,
    ChromatographyStep,
    CostBasis,
    ProcessSpec,
    cogs_breakdown,
    cycles_per_batch,
    estimate_total_opex,
    load_cost_model,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def model():
    return load_cost_model()


def with_step(process, name, **changes):
    """``process`` with one step's fields replaced."""
    steps = tuple(dataclasses.replace(s, **changes) if s.name == name else s
                  for s in process.steps)
    return dataclasses.replace(process, steps=steps)


def cost(process, basis, key="cost_per_g"):
    return cogs_breakdown(process, basis)[key]


# =============================================================================
# The issue's four complaints
# =============================================================================


class TestEachStepHasItsOwnResin:
    def test_polishing_is_not_priced_as_protein_a(self, model):
        process, basis = model
        steps = cogs_breakdown(process, basis)["steps"]
        resin = {n: float(e["resin"]) for n, e in steps.items() if "resin" in e}
        assert resin["protein_a_capture"] > 5 * resin["cation_exchange"]
        assert resin["protein_a_capture"] > 5 * resin["anion_exchange_flow_through"]

    def test_step_costs_add_up_to_the_category(self, model):
        process, basis = model
        out = cogs_breakdown(process, basis)
        assert float(out["resin"]) == pytest.approx(
            sum(float(e.get("resin", 0.0)) for e in out["steps"].values()))
        assert float(out["buffers"]) == pytest.approx(
            sum(float(e["buffers"]) for e in out["steps"].values()))

    def test_the_coarse_function_prices_steps_by_their_resins(self):
        """It charged three Protein A columns; capture is now the only one."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            three = estimate_total_opex(24, 2000.0, column_volume_L=50.0)
        assert any(w.category is DeprecationWarning for w in caught)
        proa = 50.0 * get_resin("MabSelect_SuRe").cost_usd_L * 24 / 200
        assert three["consumables_resin"] < 1.5 * proa


class TestResinFollowsTheLoad:
    def test_cycles_are_load_over_capacity(self):
        n = cycles_per_batch(jnp.asarray(9500.0), 28.0, 100.0)
        assert float(n) == pytest.approx(9500.0 / 2800.0, rel=1e-9)

    def test_the_floor_is_one_cycle(self):
        assert float(cycles_per_batch(jnp.asarray(10.0), 28.0, 100.0)) == \
            pytest.approx(1.0, abs=1e-6)

    def test_a_higher_titer_uses_more_resin(self, model):
        process, basis = model
        d = jax.grad(lambda t: cost(dataclasses.replace(process, titer_g_L=t),
                                    basis, "resin"))(5.0)
        assert d > 0

    def test_a_higher_binding_capacity_lowers_cost_per_gram(self, model):
        process, basis = model
        d = jax.grad(lambda q: cost(with_step(
            process, "protein_a_capture", dynamic_binding_capacity_g_L=q), basis))(28.0)
        assert d < 0


class TestLaborFollowsTheProcess:
    def test_labor_grows_with_batches(self, model):
        process, basis = model
        d = jax.grad(lambda n: cost(dataclasses.replace(process, batches_per_year=n),
                                    basis, "labor"))(24.0)
        expected = basis.labor_rate_usd_h * (
            basis.operator_hours_per_bioreactor_batch
            + basis.operator_hours_per_step_batch * len(process.steps))
        assert float(d) == pytest.approx(expected)

    def test_labor_grows_with_steps(self, model):
        process, basis = model
        fewer = dataclasses.replace(process, steps=process.steps[:-1])
        assert float(cost(fewer, basis, "labor")) < float(cost(process, basis, "labor"))

    def test_labor_is_an_input_not_a_hidden_default(self, model):
        process, basis = model
        doubled = dataclasses.replace(basis, labor_rate_usd_h=2 * basis.labor_rate_usd_h)
        assert float(cost(process, doubled, "labor")) == pytest.approx(
            2 * float(cost(process, basis, "labor")))


class TestTheSmallerItems:
    def test_media_includes_the_feed(self, model):
        process, basis = model
        d = jax.grad(lambda f: cost(dataclasses.replace(process, feed_L_per_L=f),
                                    basis, "media"))(0.3)
        assert float(d) == pytest.approx(
            process.working_volume_L * basis.media_usd_L["feed"] * process.batches_per_year)

    def test_buffers_are_priced_per_column_volume(self, model):
        process, basis = model
        out = cogs_breakdown(process, basis)["steps"]["protein_a_capture"]
        step = next(s for s in process.steps if s.name == "protein_a_capture")
        per_cycle = step.column_volume_L * sum(
            n * basis.buffer_usd_L[k] for k, n in step.buffer_cv.items())
        assert float(out["buffers"]) == pytest.approx(
            per_cycle * float(out["cycles"]) * process.batches_per_year)

    def test_failed_batches_cost_and_release_nothing(self, model):
        process, basis = model
        perfect = dataclasses.replace(process, batch_success_rate=1.0)
        assert float(cost(perfect, basis, "failed_batch_cost")) == 0.0
        assert float(cost(process, basis, "failed_batch_cost")) > 0.0
        # Same spend, fewer grams.
        assert float(cost(perfect, basis, "total")) == pytest.approx(
            float(cost(process, basis, "total")))
        assert float(cost(perfect, basis)) < float(cost(process, basis))

    def test_qc_and_single_use_are_counted(self, model):
        out = cogs_breakdown(*model)
        assert float(out["qc"]) > 0 and float(out["single_use"]) > 0

    def test_categories_sum_to_the_total(self, model):
        out = cogs_breakdown(*model, capex=1e8)
        assert float(out["total"]) == pytest.approx(
            sum(float(out[k]) for k in CATEGORIES))
        assert float(out["depreciation"]) == pytest.approx(1e7)


# =============================================================================
# Gradients
# =============================================================================


class TestGradients:
    @pytest.mark.parametrize("field, value", [
        ("titer_g_L", 5.0), ("batch_success_rate", 0.95), ("working_volume_L", 2000.0),
    ])
    def test_check_grads_on_process_fields(self, model, field, value):
        process, basis = model
        check_grads(lambda v: cost(dataclasses.replace(process, **{field: v}), basis),
                    (value,), order=1, modes=["fwd", "rev"])

    def test_check_grads_on_a_binding_capacity(self, model):
        process, basis = model
        check_grads(lambda q: cost(with_step(
            process, "cation_exchange", dynamic_binding_capacity_g_L=q), basis),
            (68.0,), order=1, modes=["fwd", "rev"])

    @pytest.mark.parametrize("field, value, sign", [
        ("titer_g_L", 5.0, -1),
        ("batch_success_rate", 0.95, -1),
    ])
    def test_signs(self, model, field, value, sign):
        process, basis = model
        d = jax.grad(lambda v: cost(dataclasses.replace(process, **{field: v}), basis))(value)
        assert jnp.sign(d) == sign

    def test_a_step_yield_lowers_cost(self, model):
        process, basis = model
        d = jax.grad(lambda y: cost(with_step(process, "cation_exchange",
                                              step_yield=y), basis))(0.92)
        assert d < 0

    def test_titer_is_not_only_the_denominator(self, model):
        """The issue: d(cost/g)/d(titer) was exactly -(cost/g)/titer."""
        process, basis = model
        d = jax.grad(lambda t: cost(dataclasses.replace(process, titer_g_L=t), basis))(5.0)
        trivial = -float(cost(process, basis)) / 5.0
        assert abs(float(d) - trivial) > 1e-3 * abs(trivial)


# =============================================================================
# Data in, data out
# =============================================================================


class TestData:
    def test_json_round_trip(self, model, tmp_path):
        process, basis = model
        path = tmp_path / "plant.json"
        path.write_text(json.dumps({"process": process.to_dict(),
                                    "basis": basis.to_dict()}))
        again = load_cost_model(path)
        assert float(cost(*again)) == pytest.approx(float(cost(process, basis)))

    def test_an_unknown_field_is_named(self):
        with pytest.raises(ValueError, match="no field titre"):
            ProcessSpec.from_dict({"working_volume_L": 1.0, "titre": 5.0,
                                   "batches_per_year": 1.0, "titer_g_L": 1.0})

    def test_a_step_without_a_type_is_named(self):
        with pytest.raises(ValueError, match="needs a type"):
            ProcessSpec.from_dict({"working_volume_L": 1.0, "titer_g_L": 1.0,
                                   "batches_per_year": 1.0,
                                   "steps": [{"name": "x", "step_yield": 0.9}]})

    def test_an_unpriced_buffer_is_named(self, model):
        process, basis = model
        thin = dataclasses.replace(basis, buffer_usd_L={})
        with pytest.raises(KeyError, match="no price for buffer"):
            cogs_breakdown(process, thin)

    def test_a_missing_section_is_named(self, tmp_path):
        path = tmp_path / "half.json"
        path.write_text(json.dumps({"basis": {}}))
        with pytest.raises(ValueError, match="no process"):
            load_cost_model(path)

    def test_from_resin_reads_the_database(self):
        step = ChromatographyStep.from_resin("capture", "MabSelect_SuRe", 100.0, 0.95)
        resin = get_resin("MabSelect_SuRe")
        assert step.dynamic_binding_capacity_g_L == pytest.approx(0.8 * resin.q_max)
        assert step.resin_cost_usd_L == resin.cost_usd_L

    def test_basis_from_dict(self):
        basis = CostBasis.from_dict({"media_usd_L": {"basal": 1.0, "feed": 1.0,
                                                     "seed": 1.0},
                                     "buffer_usd_L": {}, "labor_rate_usd_h": 50.0})
        assert basis.labor_rate_usd_h == 50.0


class TestTheShippedReference:
    def test_every_number_says_where_it_came_from(self):
        """A value with no source tag on its line or its parent's is refused."""
        tag = re.compile(r"\[(database|carried over|placeholder)\]")
        number = re.compile(r"^\s*-?\s*[\w.]+:\s*-?[0-9.]+(e-?\d+)?\s*(#.*)?$")
        lines = REFERENCE.read_text().splitlines()
        untagged, checked = [], 0
        for i, line in enumerate(lines):
            if not number.match(line):
                continue
            checked += 1
            if tag.search(line):
                continue
            indent = len(line) - len(line.lstrip())
            parent = next((p for p in reversed(lines[:i])
                           if p.strip() and len(p) - len(p.lstrip()) < indent), "")
            if not tag.search(parent):
                untagged.append(line.strip())
        assert checked > 50, "the pattern found too few values to be checking"
        assert not untagged, untagged

    def test_database_values_match_the_database(self, model):
        """What the file says it copied from the resin database, it did."""
        process, _ = model
        for step in process.steps:
            if not isinstance(step, ChromatographyStep):
                continue
            resin = get_resin(step.resin)
            assert step.resin_cost_usd_L == resin.cost_usd_L, step.name
            assert step.resin_lifetime_cycles == resin.cycles, step.name
        capture = next(s for s in process.steps if s.name == "protein_a_capture")
        assert capture.dynamic_binding_capacity_g_L == pytest.approx(
            0.8 * get_resin("MabSelect_SuRe").q_max)
