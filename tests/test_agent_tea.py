"""Cost bases declared by units, and the tea tool (MCP phase 5)."""

from difflow.agent import Workbench
from difflow.catalog import catalog
from difflow.economics import capital


def test_declared_bases_name_real_correlations():
    declared = {n: s.cost_basis for n, s in catalog().items() if s.cost_basis}
    assert {"CSTR", "PFR", "GasPFR", "FedBatchReactor", "SemiBatchReactor"} <= set(declared)
    for name, basis in declared.items():
        table = getattr(capital, basis["table"])
        assert basis["type"] in table, name


def test_tea_smoke():
    bench = Workbench()
    bench.open_example("01_flash_drum")
    assert bench.solve()["converged"]
    answer = bench.tea()
    assert answer["ok"] and "units" in answer
