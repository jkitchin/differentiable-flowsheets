"""How a flowsheet reads the value a unit operation returns.

A unit returns a single stream, ``(S1, ..., Sn)``, ``(S1, ..., Sn, info)``
or the nested ``((S1, ..., Sn), info)``. Four copies of the reader used
to disagree about those shapes (the sequential solve, the optimistix
fixed-point path, the info recorder and the equation-oriented solver),
and each told a stream from an info dict with ``isinstance(x, dict)``,
which a Stream also satisfies. The audit of the core outlets found:

- the nested shape was checked AFTER ``len(result) == len(outlets)``,
  so a two-outlet Ultrafiltration got its ``(retentate, permeate)``
  tuple as the first outlet and its info dict as the second, and a
  one-outlet wiring silently dropped a stream;
- a results dict handed back by a circuit went downstream as a stream
  and failed later with ``KeyError: 'T'``.
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
    """A unit in the shape the bio and power splits return."""

    def __call__(self, inlet):
        half = {k: (v * 0.5 if k.startswith("F_") else v)
                for k, v in inlet.items()}
        return (half, dict(half)), {"fraction": jnp.asarray(0.5)}


class TwoStreamsNoInfo:
    """``(S, S)`` and no info, as Desalter, FlowSplit and TearSplit return."""

    def __call__(self, inlet):
        return _s(1.0), _s(2.0)


class ResultsDict:
    """A circuit that returns one results dict, as the REE circuits do."""

    def __call__(self, inlet):
        return {"product": inlet, "recovery": 0.9}


class TestIsStream:
    def test_a_stream_is_a_stream(self):
        assert _is_stream(_s())
        assert _is_stream(make_stream({"A": 1.0}, 300.0, 1e5, phase="liquid"))

    def test_an_info_dict_reporting_T_and_P_is_not(self):
        """The old ``"T" in x`` test took any info reporting a T for a stream."""
        assert not _is_stream({"T": 300.0, "P": 1e5, "duty": 2.0})
        assert not _is_stream({"fraction": 0.5})
        assert not _is_stream({"F_A": 1.0, "T": 300.0})    # no P
        assert not _is_stream((1.0,))


class TestSplitResult:
    def test_the_nested_shape_is_read_before_the_length_match(self):
        s1, s2, info = _s(1.0), _s(2.0), {"k": 1}
        streams, got = _split_result(((s1, s2), info), ["a", "b"])
        assert streams == (s1, s2) and got is info

    def test_streams_with_no_info_are_all_streams(self):
        """The second stream of ``(S, S)`` is not an info dict."""
        s1, s2 = _s(1.0), _s(2.0)
        streams, info = _split_result((s1, s2), ["a", "b"])
        assert streams == (s1, s2) and info is None

    def test_a_trailing_info_dict(self):
        s1, s2 = _s(1.0), _s(2.0)
        info = {"T": 1.0, "P": 2.0, "duty": 3.0}    # reports T and P
        streams, got = _split_result((s1, s2, info), ["a", "b"])
        assert streams == (s1, s2) and got is info

    def test_too_few_outlets_named_raises(self):
        """With one outlet named, the second stream used to vanish."""
        with pytest.raises(ValueError, match="2 stream"):
            _split_result(((_s(), _s()), {}), ["only"])
        with pytest.raises(ValueError, match="2 stream"):
            _split_result((_s(), _s()), ["only"])

    def test_too_many_outlets_named_raises(self):
        with pytest.raises(ValueError, match="3 outlets"):
            _split_result((_s(), _s()), ["a", "b", "c"])

    def test_a_results_dict_raises_and_names_the_way_out(self):
        with pytest.raises(ValueError, match="results dict is not a stream"):
            _split_result({"product": _s(), "recovery": 0.9}, ["product"])

    def test_the_eo_solver_reads_the_same_shapes(self):
        s1, s2 = _s(1.0), _s(2.0)
        assert _parse_unit_result(((s1, s2), {"k": 1}), ["a", "b"]) == {
            "a": s1, "b": s2}
        with pytest.raises(ValueError):
            _parse_unit_result(((s1, s2), {"k": 1}), ["a"])


class TestInAFlowsheet:
    def _fs(self, op, outlets):
        fs = Flowsheet(species_order=["A", "B"])
        fs.add_feed("feed", _s(4.0, 2.0))
        fs.add_unit(Unit("u", op, ["feed"], outlets))
        return fs

    def test_nested_outlets_are_streams_and_info_is_recorded(self):
        fs = self._fs(NestedSplit(), ["o1", "o2"])
        streams = fs.solve()
        assert _is_stream(streams["o1"]) and _is_stream(streams["o2"])
        assert float(streams["o2"]["F_A"]) == pytest.approx(2.0)
        assert "fraction" in fs.last_solve_unit_info["u"]

    def test_nested_outlets_on_the_optimistix_recycle_path(self):
        """``acceleration="none"`` had its own inline copy of the reader."""
        class Mix:
            def __call__(self, a, b):
                return {k: (a[k] + b[k] if k.startswith("F_") else a[k])
                        for k in a}

        fs = Flowsheet(species_order=["A", "B"])
        fs.add_feed("feed", _s(4.0, 2.0))
        fs.add_unit(Unit("mix", Mix(), ["feed", "recycle"], ["mixed"]))
        fs.add_unit(Unit("split", NestedSplit(), ["mixed"], ["out", "back"]))
        fs.add_recycle("back", "recycle")
        streams = fs.solve(tear_initial={"recycle": _s(0.0, 0.0)},
                           acceleration="none")
        # out = back, and feed = out, at the fixed point
        assert float(streams["out"]["F_A"]) == pytest.approx(4.0, rel=1e-6)
        assert _is_stream(streams["back"])

    def test_two_streams_with_one_outlet_named_raises(self):
        with pytest.raises(ValueError, match="1 outlets"):
            self._fs(TwoStreamsNoInfo(), ["only"]).solve()

    def test_a_results_dict_raises_where_it_is_returned(self):
        """It used to go downstream and fail there with KeyError 'T'."""
        with pytest.raises(ValueError, match="results dict"):
            self._fs(ResultsDict(), ["product"]).solve()


class TestRealUnits:
    def test_ultrafiltration_wires_two_outlets(self):
        from difflow_bio import Ultrafiltration, UltrafiltrationParams

        uf = Ultrafiltration(UltrafiltrationParams(
            membrane_area=1.0, rejection={"mAb": 0.99}))
        fs = Flowsheet(species_order=["mAb", "buffer"])
        fs.add_feed("feed", make_stream({"mAb": 2.0, "buffer": 998.0},
                                        298.0, 101325.0))
        fs.add_unit(Unit("uf", uf, ["feed"], ["retentate", "permeate"],
                         {"concentration_factor": 5.0}))
        streams = fs.solve()
        assert _is_stream(streams["retentate"])
        assert _is_stream(streams["permeate"])
        assert float(streams["retentate"]["F_mAb"]) > float(
            streams["permeate"]["F_mAb"])
        assert "flux" in fs.last_solve_unit_info["uf"]

    def test_power_split_wires_two_outlets(self):
        from difflow_power.streams import power_stream
        from difflow_power.units.nodes import PowerSplit, SplitParams

        fs = Flowsheet(species_order=["P", "Q"])
        fs.add_feed("bus", power_stream(1.0, 0.2, 1.0, 0.0))
        fs.add_unit(Unit("sp", PowerSplit(SplitParams(fraction=0.25)),
                         ["bus"], ["a", "b"]))
        streams = fs.solve()
        assert float(streams["a"]["F_P"]) == pytest.approx(0.25)
        assert float(streams["b"]["F_P"]) == pytest.approx(0.75)
