"""A recycle loop with two stable steady states, x = 0 and x = 1 (0.5 is unstable)."""
from difflow import Flowsheet, Unit, make_stream


def two_state_loop(feed, tear):
    x = tear["F_A"]
    return make_stream({"A": 3 * x**2 - 2 * x**3}, feed["T"], feed["P"])


def bistable():
    fs = Flowsheet(species_order=["A"])
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 101325.0))
    fs.add_unit(Unit("loop", two_state_loop, ["feed", "tear"], ["out"]))
    fs.add_recycle("out", "tear")
    return fs
