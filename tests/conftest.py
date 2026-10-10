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
import json
import os

import jax
import pytest

# Load difflow in whichever process imports this file, the xdist controller
# included. When a worker reports a warning raised by a difflow class
# (CSTRDensityWarning, ...), the controller imports the warning's module to
# rebuild it; two such reports at once import ``difflow`` from two threads
# and deadlock on the module lock. Having it loaded first avoids the import.
import difflow  # noqa: F401
import difflow_bio  # noqa: F401
import difflow_cc  # noqa: F401
import difflow_gas  # noqa: F401
import difflow_power  # noqa: F401
import difflow_ree  # noqa: F401
import difflow_refinery  # noqa: F401

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



_DURATIONS = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".test_durations")

#: Files that must not run at the same time as each other (#315). On the CI
#: runner (2 physical cores, 4 vCPUs) each of these runs 2-2.5x slower when
#: another of them is running on the other worker -- measured: the
#: non-convergence file 470s alone against 1140s overlapped, the full grid
#: 210s against 510s, the planning file 420s against 630s -- so overlapping
#: them costs more time than it saves. ``--dist loadfile`` hands them out as
#: ONE work unit, which puts them on one worker in series, while the other
#: worker takes everything else. (Pinning XLA to one thread did not help:
#: measured, same wall time.) A file belongs here only on that evidence.
_SERIAL_GROUP = "serial-heavy"
_SERIAL_FILES = frozenset({
    "tests/refinery/test_planning.py",
    "tests/refinery/test_planning_nonconvergence.py",
    "tests/test_convergence_full_grid.py",
})


#: Files that must not be compiling at the same time as each other because
#: together they exhaust the runner's memory. Full suite shard 6/12 holds all
#: three and on Python 3.11 lost its runner at 79 min ("The hosted runner lost
#: communication with the server", the out-of-memory signature); 3.12 passed
#: the same shard in 42 min against a recorded 24 min of tests, so the two
#: workers were contending there as well. In series on one worker they take
#: about 23 min.
_MEMORY_GROUP = "isomerization"
_MEMORY_FILES = frozenset({
    "tests/refinery/test_isomerization.py",
    "tests/refinery/test_isomerization_dih.py",
    "tests/refinery/test_isomerization_dih_gradients.py",
})


#: Files whose tests are independent and long enough that one worker running
#: the whole file sets the shard's length: each test is its own work unit.
#: tests/test_doc_examples.py runs every document's code blocks, ~23 runner
#: minutes in all (the refinery page and README alone ~5 each), and as one
#: unit it was the tail that pushed per-commit shard 4 past its 40 min cap.
_SPLIT_FILES = frozenset({
    "tests/test_doc_examples.py",
})


def _work_unit(nodeid):
    """The unit ``--dist loadfile`` hands out: the file, its group, or (for
    :data:`_SPLIT_FILES`) the test."""
    path = nodeid.split("::", 1)[0]
    if path in _SPLIT_FILES:
        return nodeid
    if path in _SERIAL_FILES:
        return _SERIAL_GROUP
    if path in _MEMORY_FILES:
        return _MEMORY_GROUP
    return path


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config, log):
    """``--dist loadfile``, with :data:`_SERIAL_FILES` and
    :data:`_MEMORY_FILES` each as one work unit and :data:`_SPLIT_FILES`
    one unit per test."""
    if config.getvalue("dist") != "loadfile":
        return None
    from xdist.scheduler import LoadFileScheduling

    class _Scheduling(LoadFileScheduling):
        def _split_scope(self, nodeid):
            return _work_unit(nodeid)

    return _Scheduling(config, log)


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(config, items):
    """Under xdist, hand the work units out longest first (#315).

    xdist's own order is by test COUNT, descending, and each worker keeps
    its next unit queued -- so a unit with a few tests and many minutes of
    setup is handed out late, queued behind a running one, and ends the
    shard alone while the other worker sits idle. Measured on shard 2: the
    refinery planning files started at 552s and 879s and ran to 1089s, one
    worker idle from 775s.

    Sorting whole units by their recorded duration (``.test_durations``,
    the same numbers pytest-split splits on) is the longest-processing-time
    rule; CI passes ``--no-loadscope-reorder`` so xdist keeps this order.
    The order inside a file is untouched. This is a wrapper so it runs after
    pytest-split has chosen the group, and only on xdist workers (where the
    collection that xdist schedules from happens), so a serial run is
    unchanged.
    """
    result = yield
    if not os.environ.get("PYTEST_XDIST_WORKER") or not items:
        return result
    try:
        with open(_DURATIONS) as fh:
            durations = json.load(fh)
    except (OSError, ValueError):
        return result
    known = [durations[i.nodeid] for i in items if i.nodeid in durations]
    default = sum(known) / len(known) if known else 0.0
    totals = {}
    for item in items:
        unit = _work_unit(item.nodeid)
        totals[unit] = totals.get(unit, 0.0) + durations.get(item.nodeid, default)
    # sorted() is stable: in-file order survives, ties keep collection order.
    items[:] = sorted(items, key=lambda i: -totals[_work_unit(i.nodeid)])
    return result
