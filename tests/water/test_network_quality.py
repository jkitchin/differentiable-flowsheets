"""Water quality (Lagrangian transport): runs and finite for each mode; inputs validated."""

import numpy as np
import pytest

from difflow_water.network import example_inp, read_inp, simulate
from difflow_water.network.quality import water_quality


@pytest.mark.parametrize("mode", ["CHEMICAL", "AGE"])
def test_net1_quality_runs(mode):
    net, sch = read_inp(example_inp("Net1"))
    sch.quality_options["mode"] = mode
    wq = water_quality(net, sch, simulate(net, sch, duration=6 * 3600))
    assert wq["quality"].shape == (7, len(wq["nodes"]))
    assert np.all(np.isfinite(wq["quality"])) and np.all(wq["quality"] >= 0)


def test_net3_trace_runs():
    net, sch = read_inp(example_inp("Net3"))
    wq = water_quality(net, sch, simulate(net, sch, duration=3 * 3600))
    assert wq["mode"] == "TRACE"
    assert np.all((wq["quality"] >= -1e-9) & (wq["quality"] <= 100 + 1e-9))


def test_no_quality_analysis_is_refused():
    net, sch = read_inp(example_inp("Net1"))
    sch.quality_options["mode"] = "NONE"
    with pytest.raises(ValueError):
        water_quality(net, sch, simulate(net, sch, duration=3600))


@pytest.mark.parametrize("section", ["[SOURCES]\n 10 CONCEN 1.0\n", "[MIXING]\n 2 FIFO\n",
                                     "[REACTIONS]\n Roughness Correlation 1.0\n"])
def test_unsupported_quality_input_is_refused(tmp_path, section):
    text = example_inp("Net1").read_text().replace("[END]", section + "[END]")
    p = tmp_path / "net1.inp"
    p.write_text(text)
    with pytest.raises(ValueError):
        read_inp(p)
