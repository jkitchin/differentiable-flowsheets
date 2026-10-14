"""How a flowsheet reads the value a unit operation returns.

A unit returns a single stream, ``(S1, ..., Sn)``, ``(S1, ..., Sn, info)``
or the nested ``((S1, ..., Sn), info)``.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.eo_solver import _parse_unit_result
from difflow.flowsheet import Flowsheet, Unit, _is_stream, _split_result
from difflow.streams import make_stream


def _s(a=1.0, b=0.0, T=300.0):
    return make_stream({"A": a, "B": b}, T, 101325.0)


class NestedSplit:
    def __call__(self, inlet):
        half = {k: (v * 0.5 if k.startswith("F_") else v) for k, v in inlet.items()}
        return (half, dict(half)), {"fraction": jnp.asarray(0.5)}


def test_is_stream():
    assert _is_stream(_s())
    assert not _is_stream({"T": 300.0, "P": 1e5, "duty": 2.0})


def test_split_result_shapes():
    s1, s2, info = _s(1.0), _s(2.0), {"k": 1}
    assert _split_result(((s1, s2), info), ["a", "b"]) == ((s1, s2), info)
    assert _split_result((s1, s2), ["a", "b"]) == ((s1, s2), None)
    assert _split_result((s1, s2, info), ["a", "b"]) == ((s1, s2), info)
    streams, _ = _split_result((s1, s2, info), ["a", "b", "a_info"])
    assert streams == (s1, s2, info)
    assert _parse_unit_result(((s1, s2), info), ["a", "b"]) == {"a": s1, "b": s2}


def test_split_result_errors():
    with pytest.raises(ValueError, match="2 stream"):
        _split_result((_s(), _s()), ["only"])
    with pytest.raises(ValueError, match="3 outlets"):
        _split_result((_s(), _s()), ["a", "b", "c"])
    with pytest.raises(ValueError, match="results dict is not a stream"):
        _split_result({"product": _s(), "recovery": 0.9}, ["product"])


def test_nested_outlets_in_a_flowsheet():
    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", _s(4.0, 2.0))
    fs.add_unit(Unit("u", NestedSplit(), ["feed"], ["o1", "o2"]))
    streams = fs.solve()
    assert _is_stream(streams["o1"]) and _is_stream(streams["o2"])
    assert "fraction" in fs.last_solve_unit_info["u"]
