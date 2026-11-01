"""The same loop with a preheater whose correlation is NaN below 400 K."""
import jax.numpy as jnp
from difflow import Unit, make_stream
from plant import fs, species


def preheat(stream):
    T_out = stream["T"] + 10.0 * jnp.log(stream["T"] - 400.0)     # bug: NaN when T < 400 K
    return make_stream({s: stream[f"F_{s}"] for s in species}, T_out, stream["P"])


fs.units.insert(1, Unit("preheat", preheat, ["reactor_feed"], ["hot_feed"]))
fs.units[2].inlet_names = ["hot_feed"]
