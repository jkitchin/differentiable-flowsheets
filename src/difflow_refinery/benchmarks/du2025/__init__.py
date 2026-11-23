"""The Du et al. (2025) refinery planning benchmark, implemented as published.

W. Du, C. Wang, C. Fan, et al., "A production planning benchmark for
real-world refinery-petrochemical complexes," arXiv:2503.22057 (2025).
Data and GAMS model: https://github.com/EMRPS/refinery-planning-benchmark

The paper is CC BY-NC-SA and the data repository has no license file, so
nothing of the benchmark is vendored: :func:`fetch` downloads it,
:func:`load_case` reads it, :func:`build_model` assembles the Case 1
superstructure as one differentiable JAX model, and :func:`solve_local`
maximizes profit locally with pounce. The generic blocks (fixed-yield,
delta-base, swing-cut CDU, mixer, splitter, blender) are in
:mod:`~difflow_refinery.benchmarks.du2025.blocks`; the property map in
:mod:`~difflow_refinery.benchmarks.du2025.properties`; reconstructed crude
assays in :mod:`~difflow_refinery.benchmarks.du2025.assays` (#532); calibrating
difflow units into it in :mod:`~difflow_refinery.benchmarks.du2025.calibration`
and :mod:`~difflow_refinery.benchmarks.du2025.units` (#533).

A library, not a palette operation (#538 tracks the reference refinery).

>>> from difflow_refinery.benchmarks import du2025          # doctest: +SKIP
>>> root = du2025.fetch()                                  # doctest: +SKIP
>>> model = du2025.build_model(du2025.load_case(root / "case1"))  # doctest: +SKIP
>>> sol = du2025.load_solution(root / "case1" / "solution_baron.gdx")  # doctest: +SKIP
>>> float(model.profit(model.x_from_solution(sol)))        # doctest: +SKIP
34167967.96...
"""

from difflow_refinery.benchmarks.du2025 import assays, blocks, calibration, properties, units
from difflow_refinery.benchmarks.du2025.data import (
    CASE1_PUBLISHED_PROFIT,
    Du2025Data,
    default_data_dir,
    export_solution_json,
    fetch,
    find_case,
    load_case,
    load_solution,
    parse_gams_text,
)
from difflow_refinery.benchmarks.du2025.model import (
    Du2025Model,
    PolySystem,
    WARM_START_OPTIONS,
    build_model,
    solve_local,
)

__all__ = [
    "CASE1_PUBLISHED_PROFIT", "Du2025Data", "assays", "calibration", "properties", "units", "Du2025Model", "PolySystem", "WARM_START_OPTIONS", "blocks",
    "build_model", "default_data_dir", "export_solution_json", "fetch", "find_case",
    "load_case", "load_solution", "parse_gams_text", "solve_local",
]
