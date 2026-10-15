"""difflow_bio cost of goods: smoke, gradient, data I/O and validation."""

import dataclasses
import json
import re

import jax
import jax.numpy as jnp
import pytest

from difflow_bio.economics import (
    CATEGORIES,
    REFERENCE,
    ProcessSpec,
    cogs_breakdown,
    estimate_total_opex,
    load_cost_model,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def model():
    return load_cost_model()


def test_breakdown_runs_and_sums(model):
    out = cogs_breakdown(*model)
    assert jnp.isfinite(out["cost_per_g"])
    assert float(out["total"]) == pytest.approx(sum(float(out[k]) for k in CATEGORIES))


def test_grad_is_finite(model):
    process, basis = model
    g = jax.grad(lambda t: cogs_breakdown(
        dataclasses.replace(process, titer_g_L=t), basis)["cost_per_g"])(5.0)
    assert jnp.isfinite(g)


def test_coarse_function_is_deprecated():
    with pytest.warns(DeprecationWarning):
        estimate_total_opex(24, 2000.0, column_volume_L=50.0)


def test_json_round_trip(model, tmp_path):
    process, basis = model
    path = tmp_path / "plant.json"
    path.write_text(json.dumps({"process": process.to_dict(), "basis": basis.to_dict()}))
    again = load_cost_model(path)
    assert float(cogs_breakdown(*again)["cost_per_g"]) == pytest.approx(
        float(cogs_breakdown(process, basis)["cost_per_g"]))


def test_an_unknown_field_is_named():
    with pytest.raises(ValueError, match="no field titre"):
        ProcessSpec.from_dict({"working_volume_L": 1.0, "titre": 5.0,
                               "batches_per_year": 1.0, "titer_g_L": 1.0})


def test_a_step_without_a_type_is_named():
    with pytest.raises(ValueError, match="needs a type"):
        ProcessSpec.from_dict({"working_volume_L": 1.0, "titer_g_L": 1.0,
                               "batches_per_year": 1.0,
                               "steps": [{"name": "x", "step_yield": 0.9}]})


def test_an_unpriced_buffer_is_named(model):
    process, basis = model
    with pytest.raises(KeyError, match="no price for buffer"):
        cogs_breakdown(process, dataclasses.replace(basis, buffer_usd_L={}))


def test_a_missing_section_is_named(tmp_path):
    path = tmp_path / "half.json"
    path.write_text(json.dumps({"basis": {}}))
    with pytest.raises(ValueError, match="no process"):
        load_cost_model(path)


def test_every_number_in_the_reference_is_tagged():
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
    assert checked > 50
    assert not untagged, untagged
