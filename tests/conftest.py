"""Fixtures every test file gets.

Why this file exists: JAX keeps each executable it compiles in a
process-global cache, keyed by the traced callable and its *static*
arguments, and nothing evicts it -- the cache outlives the test that filled
it. difflow reaches that cache from two directions at once. Cores like the
ones in ``difflow/units/eos_units.py`` take a thermo object as a static
argument, so a fresh thermo is a fresh key; and every closure a test builds
over a fresh network, flowsheet or residual function is a fresh key too.

The modules that fill it fastest are the dense-linear-algebra ones, because
their executables are the largest: over one ``pytest tests/`` run,
``tests/gas/test_reconciliation.py`` (a ``jax.jacobian`` Jacobian into a
dense KKT solve, per test) adds ~1.7 GB by itself, ``tests/power/test_dc.py``
(a dense ``inv`` of the bus matrix, PTDF/LODF) ~0.7 GB, and
``tests/ree/test_mass_action.py`` ~0.5 GB. The run climbs past 5 GB and
tens of thousands of mapped code sections.

What that produces is not a clean ``MemoryError``. LLVM's JIT asks the
kernel for one more section mapping, the kernel refuses, and the process
aborts where it stands::

    allocateMappedMemory failed with error: Cannot allocate memory
    Failed to satisfy suballocation request for 40
    LLVM ERROR: Unable to allocate section memory!
    Fatal Python error: Aborted

The request that fails is 40 bytes, with gigabytes of RAM free: what has run
out is address-space mappings, not memory. Whichever test happens to be
compiling at that moment takes the abort, which is why the crash lands
somewhere different every run -- ``tests/ree`` once, ``tests/test_flash_gradients.py``
the next -- and why every one of those tests passes on its own.

Clearing holds it flat: measured on a synthetic stand-in, 120 cached
executables cost 665 MB and 720 mappings uncleared, versus 340 MB and ~20
mappings cleared per block.

Two fixtures do it, because two different things overflow. Across files it
is the sum that grows, and clearing at each module boundary costs nothing:
no test module builds on another's compiled objects. Inside one file it can
be that file alone -- ``tests/test_distillation.py`` reaches three quarters
of the kernel's mapping ceiling within a single test class -- so a budget
check after each test drops the caches mid-module once they have grown past
``MAPPING_BUDGET``. That check is deliberately a budget rather than an
unconditional per-test clear: a file whose tests share a thermo so its
compiled core is reused (``tests/test_dynamic_eos.py`` says so in as many
words) stays under the budget and keeps its reuse.
"""

import gc
import os

import jax
import pytest

#: The kernel's ceiling on mapped sections per process, ``vm.max_map_count``.
#: 65530 is the kernel default and what this budget was first measured
#: against, but distributions and CI images raise it: GitHub's
#: ``ubuntu-latest`` runner has 262144.
_DEFAULT_MAX_MAP_COUNT = 65_530


def _max_map_count():
    """``vm.max_map_count``, or the kernel default where it is not readable."""
    try:
        with open("/proc/sys/vm/max_map_count") as fh:
            return int(fh.read())
    except (OSError, ValueError):
        return _DEFAULT_MAX_MAP_COUNT


#: Clear once the process holds more mapped sections than this. Against the
#: default ceiling of 65530 it is 15000, which leaves room for a module
#: heavier than any here today. Measured on ``tests/test_distillation.py``,
#: the worst offender: 49480 mappings and 8.2 GB peak with only the
#: module-boundary clear below, versus 13496 and 6.0 GB with this budget, for
#: 15 seconds on a 10-minute module -- those tests build a fresh column per
#: test and so recompile either way, which is why dropping the caches under
#: them costs so little.
#:
#: It scales with the real ceiling, because the ceiling is what it protects.
#: A fixed 15000 on a 262144-mapping CI runner cleared the caches 19 times
#: in shard 2 (#315), each at a small fraction of the headroom the kernel
#: actually had. And not every module recompiles regardless: the crude-unit
#: planning tests reuse compiled column cores across tests, and clearing
#: them mid-module took ``tests/refinery/test_planning.py`` from 929 s to
#: 1150 s on the runner.
_MAPPING_BUDGET = 15_000 * _max_map_count() // _DEFAULT_MAX_MAP_COUNT

#: Under ``pytest -n`` every worker is a separate process holding its own
#: caches, so the budget above is spent once per worker and what they add up
#: to is what the machine has to hold. Measured on the whole suite at
#: ``-n 4``: 12.1 GB peak against the 16 GB a GitHub runner has, which is
#: close enough to the edge that a slightly heavier module would be the one
#: to find it. Dividing the budget by the worker count holds the *sum* at
#: what one serial run costs: the same suite peaks at 7.4 GB with this in
#: place. The floor keeps a hypothetical ``-n 32`` from clearing the caches
#: after every test.
#:
#: On CI (``-n auto`` is 2 workers there: the runner's 4 vCPUs are 2
#: physical cores) this comes to 30000 a worker. Memory there was measured
#: rather than assumed (#315). With the mid-module clear switched off
#: altogether the heaviest per-commit shard peaked at 10.6 GB used of 16 GB,
#: against 8.7 GB with the old 7500 budget; with this one, 10.8 GB on the
#: per-commit tier and 11.3 GB on the whole suite (the release tier
#: included). A clear does not hand memory back -- resident size stays where
#: it was -- it only stops it climbing, so a larger budget costs headroom
#: slowly, and that headroom is what to re-measure if a module gets heavier.
MAPPING_BUDGET = max(
    _MAPPING_BUDGET // int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", 1)),
    2_000,
)


def _mappings():
    """Mapped regions held by this process, or -1 where that is not readable."""
    try:
        with open("/proc/self/maps") as fh:
            return sum(1 for _ in fh)
    except OSError:          # not Linux; the module-boundary clear still runs
        return -1


@pytest.fixture(autouse=True)
def _bound_jax_compilation_caches():
    """Drop the caches mid-module once they have grown past the budget.

    The module-boundary clear below is not enough on its own: one file can
    exhaust the mappings by itself, and ``tests/test_distillation.py`` gets
    three quarters of the way there inside a single test class.
    """
    yield
    if _mappings() > MAPPING_BUDGET:
        jax.clear_caches()
        gc.collect()


@pytest.fixture(autouse=True, scope="module")
def _release_jax_compilation_caches():
    """Drop JAX's compilation caches when a test module finishes."""
    yield
    jax.clear_caches()
    gc.collect()
