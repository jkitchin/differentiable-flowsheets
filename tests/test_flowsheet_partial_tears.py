"""Regression: tearing a stream that omits a species used to raise KeyError."""

import jax

jax.config.update("jax_enable_x64", True)

from difflow import Flowsheet, make_stream  # noqa: E402
from difflow.flowsheet import Unit  # noqa: E402

SPECIES = ["A", "B", "S"]


def mix(feed, rec):
    return make_stream(
        {s: feed.get(f"F_{s}", 0.0) + rec.get(f"F_{s}", 0.0) for s in SPECIES},
        feed["T"], feed["P"],
    )


def split_without_solvent(mixed):
    back = make_stream({"A": 0.5 * mixed["F_A"], "B": 0.5 * mixed["F_B"]},
                       mixed["T"], mixed["P"])
    product = make_stream({s: mixed[f"F_{s}"] - back.get(f"F_{s}", 0.0)
                           for s in SPECIES}, mixed["T"], mixed["P"])
    return product, back


def test_a_tear_missing_a_species_solves():
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 1.0, "S": 2.0}, 300.0, 1e5))
    fs.add_unit(Unit("mix", mix, ["feed", "rec"], ["mixed"]))
    fs.add_unit(Unit("split", split_without_solvent, ["mixed"], ["product", "back"]))
    fs.add_recycle("back", "rec")
    streams = fs.solve(tol=1e-10, max_iter=200)
    assert "F_S" not in streams["back"]
    assert fs.last_solve_converged
