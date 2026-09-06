"""Tests for exporting delta vectors to LP-style planning systems.

Two things are being verified here, and they are different in kind.

The *export* tests check that :mod:`difflow.planning.export` loses nothing:
the JSON manifest reconstructs ``J``, ``u0`` and ``y0`` exactly, the CSV
matrix reloads with ``numpy.loadtxt``, and the ``.lp``/``.mps`` files carry the
sanitised names the manifest promises.

The *bridge* tests check that :meth:`difflow.planning.Block.from_flowsheet`
produces delta vectors that are actually right.  The decisive one linearises a
flowsheet with a recycle and compares AD against finite differences, because
that path differentiates implicitly through the tear solve --- the place where
a plausible-looking Jacobian is most likely to be wrong.
"""

import csv
import json
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    Flowsheet,
    IdealThermo,
    Mixer,
    SpeciesData,
    Splitter,
    Unit,
    make_stream,
)
from difflow.planning import (
    Block,
    DeltaVectorSet,
    chain,
    check_delta_vectors,
    linearize_block,
    sanitize_name,
    write_csv,
    write_iterations_csv,
    write_json,
    write_lp,
    write_mps,
)
from difflow.planning.linearize import PhaseBoundaryWarning

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plan_result():
    """A solved reference plan: two blocks, one link, one spec."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PhaseBoundaryWarning)
        return chain.two_plant_chain().planner(radius=0.3).solve()


@pytest.fixture(scope="module")
def dvs(plan_result):
    return DeltaVectorSet.from_result(plan_result)


@pytest.fixture
def recycle_flowsheet():
    """Feed + recycle -> mixer -> CSTR -> splitter, half the product recycled.

    Small, but it has the property that matters: the solve is a fixed-point
    iteration, so any Jacobian through it is an implicit derivative.
    """
    species = {
        s: SpeciesData(
            s,
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 500.0),
            antoine_coeffs=(10.0, 3000.0, -50.0),
        )
        for s in ("A", "B")
    }
    thermo = IdealThermo(species)

    def rate_fn(C, T, params):
        k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
        return jnp.array([k * C["A"]])

    cstr = CSTR(
        CSTRParams(
            V=jnp.array(1.5),
            rate_fn=rate_fn,
            stoich=jnp.array([[-1.0], [+1.0]]),
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        ),
        thermo=thermo,
        mode="isothermal",
    )

    fs = Flowsheet(species_order=["A", "B"], default_flow=1.0)
    fs.add_feed("feed", make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(Unit("mix", Mixer(["A", "B"]), ["feed", "recycle"], ["mixed"]))
    fs.add_unit(Unit("reactor", cstr, ["mixed"], ["product"],
                     params={"T_spec": 350.0}))
    fs.add_unit(Unit("split", Splitter(["A", "B"]), ["product"],
                     ["purge", "recycled"], params={"split_frac": 0.5}))
    fs.add_recycle("recycled", "recycle")
    return fs


@pytest.fixture
def recycle_block(recycle_flowsheet):
    return Block.from_flowsheet(
        recycle_flowsheet,
        u=["reactor.V", "feed:feed.total_flow"],
        y=["purge.F_B", "purge.total_flow"],
        name="plant",
        lb=[0.5, 5.0],
        ub=[5.0, 20.0],
        solve_kwargs={"tol": 1e-10, "max_iter": 200},
    )


def _read_matrix(path):
    """Return (header, u0 row, {y_name: row}) from a jacobian CSV."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    u0 = [float(v) for v in rows[1][3:]]
    body = {r[0]: [float(v) for v in r[3:]] for r in rows[3:]}
    return header, u0, body


# ---------------------------------------------------------------------------
# The IR
# ---------------------------------------------------------------------------


class TestDeltaVectorSet:
    def test_carries_every_block(self, dvs, plan_result):
        assert [v.block for v in dvs.vectors] == list(plan_result.linearizations)

    def test_joins_names_bounds_and_numbers(self, dvs, plan_result):
        v = dvs.vector("ngl")
        block = plan_result.network.block("ngl")

        assert v.u_names == block.qualified_u()
        assert v.y_names == block.qualified_y()
        assert np.allclose(v.J, np.asarray(plan_result.linearizations["ngl"].J))
        assert v.lb == [float(x) for x in block.lb]
        assert v.ub == [float(x) for x in block.ub]

    def test_carries_units_from_block_metadata(self, dvs):
        """The reference chain's UNITS table reaches the export."""
        v = dvs.vector("ngl")
        assert v.u_units[v.u_names.index("ngl.T_coldbox")] == "K"
        assert v.y_units[v.y_names.index("ngl.NGL_C2")] == "mol/s"

    def test_records_the_validity_radius(self, dvs, plan_result):
        v = dvs.vector("ngl")
        assert v.radius == pytest.approx(plan_result.radius)
        for lo, hi, u0 in zip(v.tr_lo, v.tr_hi, v.u0):
            assert lo <= u0 + 1e-12
            assert hi >= u0 - 1e-12

    def test_scaled_jacobian_is_dimensionless(self, dvs):
        v = dvs.vector("ngl")
        assert v.scaled_J is not None
        assert np.shape(v.scaled_J) == np.shape(v.J)
        # Pressure in Pa vs a split fraction: raw entries differ by decades,
        # scaled ones do not.
        assert np.max(np.abs(v.scaled_J)) < 1e3

    def test_carries_links_prices_specs_and_duals(self, dvs, plan_result):
        assert ["ngl.residue_F", "power.fuel_F"] in dvs.links
        assert dvs.prices == {k: float(x)
                              for k, x in plan_result.planner.prices.items()}
        assert dvs.specs and "coeffs" in dvs.specs[0]
        assert dvs.specs[0]["coeffs"], "Spec.coeffs must survive the export"
        assert dvs.duals is not None and "eq" in dvs.duals

    def test_carries_the_trust_region_history(self, dvs, plan_result):
        assert len(dvs.history) == len(plan_result.history)
        assert {"index", "radius", "rho", "accepted"} <= set(dvs.history[0])

    def test_meta_states_provenance(self, dvs):
        assert dvs.meta["format"] == "delta-vectors/1"
        assert dvs.meta["converged"] is True
        assert dvs.meta["lp_sense"] in (1.0, -1.0)

    def test_predict_reproduces_the_first_order_model(self, dvs):
        v = dvs.vector("ngl")
        assert v.predict(v.u0) == pytest.approx(v.y0)
        step = list(v.u0)
        step[1] += 1.0
        expected = np.asarray(v.y0) + np.asarray(v.J)[:, 1]
        assert v.predict(step) == pytest.approx(expected.tolist())

    def test_unknown_block_raises(self, dvs):
        with pytest.raises(KeyError, match="no block"):
            dvs.vector("nope")

    def test_unbounded_levers_still_get_a_scaled_form(self):
        """Without bounds the scale falls back to max(|u0|, 1), not infinity."""
        blk = Block(name="free", fn=lambda u: jnp.array([u[0] * 2.0]),
                    u_names=["a"], y_names=["f"], u0=[1.0])
        one = DeltaVectorSet.from_block(blk)

        assert one.vectors[0].J == [[2.0]]
        assert one.vectors[0].scaled_J == [[1.0]]   # 2 * 1 / |y0| = 2/2

    def test_from_block_needs_no_planner(self, recycle_block):
        one = DeltaVectorSet.from_block(recycle_block, radius=0.2)

        assert len(one.vectors) == 1
        assert one.vectors[0].block == "plant"
        assert one.vectors[0].radius == pytest.approx(0.2)
        assert one.links == [] and one.specs == []


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class TestWriteJson:
    def test_round_trips_the_numbers_exactly(self, dvs, tmp_path):
        path = write_json(dvs, tmp_path / "plan.json")
        back = json.load(open(path, encoding="utf-8"))

        for v in dvs.vectors:
            got = next(b for b in back["vectors"] if b["block"] == v.block)
            assert got["J"] == v.J          # bit-for-bit, not approx
            assert got["u0"] == v.u0
            assert got["y0"] == v.y0
            assert got["u_names"] == v.u_names

    def test_is_plain_json(self, dvs, tmp_path):
        """No JAX or numpy scalars survive; the file loads anywhere."""
        back = json.load(open(write_json(dvs, tmp_path / "p.json"),
                              encoding="utf-8"))
        leaves = json.dumps(back)          # would raise on a JAX scalar
        assert "DeviceArray" not in leaves and "array(" not in leaves


class TestWriteCsv:
    def test_writes_one_matrix_per_block_plus_tables(self, dvs, tmp_path):
        written = write_csv(dvs, tmp_path / "tables")
        names = {os.path.basename(p) for p in written}

        assert {"ngl_jacobian.csv", "power_jacobian.csv", "base.csv",
                "bounds.csv", "links.csv", "specs.csv", "prices.csv",
                "duals.csv", "health.csv", "names.csv"} <= names
        for path in written:
            assert os.path.exists(path)

    def test_matrix_reloads_and_matches(self, dvs, tmp_path):
        write_csv(dvs, tmp_path / "tables")
        v = dvs.vector("ngl")
        path = tmp_path / "tables" / "ngl_jacobian.csv"

        header, u0, body = _read_matrix(path)
        assert header[3:] == v.u_names
        assert u0 == pytest.approx(v.u0)
        for i, y in enumerate(v.y_names):
            assert body[y] == pytest.approx(v.J[i])

        # And with numpy, which is how a planning engineer will actually
        # reload it.
        J = np.loadtxt(path, delimiter=",", skiprows=3,
                       usecols=range(3, 3 + len(v.u_names)))
        assert np.allclose(J, np.asarray(v.J))

    def test_scaled_flag_writes_the_dimensionless_form(self, dvs, tmp_path):
        write_csv(dvs, tmp_path / "raw")
        write_csv(dvs, tmp_path / "scaled", scaled=True)
        v = dvs.vector("ngl")

        _, _, raw = _read_matrix(tmp_path / "raw" / "ngl_jacobian.csv")
        _, _, sc = _read_matrix(tmp_path / "scaled" / "ngl_jacobian.csv")

        assert raw["ngl.NGL_C2"] == pytest.approx(v.J[0])
        assert sc["ngl.NGL_C2"] == pytest.approx(v.scaled_J[0])

    def test_bounds_table_carries_the_trust_region(self, dvs, tmp_path):
        write_csv(dvs, tmp_path / "t")
        rows = list(csv.DictReader(open(tmp_path / "t" / "bounds.csv",
                                        encoding="utf-8")))
        row = next(r for r in rows if r["variable"] == "ngl.T_coldbox")

        assert float(row["lb"]) <= float(row["tr_lo"])
        assert float(row["tr_hi"]) <= float(row["ub"])

    def test_names_table_records_the_lp_mapping(self, dvs, tmp_path):
        write_csv(dvs, tmp_path / "t")
        mapping = {r["name"]: r["lp_name"] for r in
                   csv.DictReader(open(tmp_path / "t" / "names.csv",
                                       encoding="utf-8"))}

        assert mapping["ngl.T_coldbox"] == "ngl_T_coldbox"


class TestWriteIterationsCsv:
    def test_writes_the_audit_trail(self, plan_result, tmp_path):
        path = write_iterations_csv(plan_result, tmp_path / "iters.csv")
        rows = list(csv.DictReader(open(path, encoding="utf-8")))

        assert len(rows) == len(plan_result.history)
        assert int(rows[0]["index"]) == plan_result.history[0].index
        assert float(rows[0]["rho"]) == pytest.approx(
            plan_result.history[0].rho)


class TestSanitizeName:
    @pytest.mark.parametrize("raw, safe", [
        ("ngl.residue_F", "ngl_residue_F"),
        ("slack[cap]", "slack_cap_"),
        ("feed:F1.total_flow", "feed_F1_total_flow"),
        ("2nd_stage", "v2nd_stage"),
    ])
    def test_makes_lp_safe_names(self, raw, safe):
        assert sanitize_name(raw) == safe


class TestWriteLpAndMps:
    def test_lp_uses_the_manifest_symbols(self, plan_result, dvs, tmp_path):
        pytest.importorskip("pyomo")
        text = open(write_lp(plan_result.lp_model, tmp_path / "p.lp"),
                    encoding="utf-8").read()

        for original, symbol in dvs.meta["lp_symbols"].items():
            assert symbol in text, f"{original} -> {symbol} missing from the LP"
            if original != symbol:
                assert original not in text

    def test_mps_is_written(self, plan_result, dvs, tmp_path):
        pytest.importorskip("pyomo")
        text = open(write_mps(plan_result.lp_model, tmp_path / "p.mps"),
                    encoding="utf-8").read()

        assert "ROWS" in text and "COLUMNS" in text
        assert dvs.meta["lp_symbols"]["ngl.T_coldbox"] in text

    def test_re_solving_the_written_model_matches(self, plan_result, tmp_path):
        """The file is the same problem difflow solved, not a lookalike."""
        pyo = pytest.importorskip("pyomo.environ")
        from difflow.planning.export import _sanitized_model

        solver = next((s for s in ("appsi_highs", "glpk", "cbc", "cplex")
                       if pyo.SolverFactory(s).available(False)), None)
        if solver is None:
            pytest.skip("no LP solver available to Pyomo")

        renamed, symbols = _sanitized_model(plan_result.lp_model)
        model = renamed.to_pyomo()
        pyo.SolverFactory(solver).solve(model)
        ours = plan_result.lp_model.solve()

        # Pyomo minimises the same row difflow's own solver does, so the
        # objectives must agree before the sense and offset are applied.
        assert pyo.value(model.obj) == pytest.approx(
            float(plan_result.lp_model.c @ ours.x), rel=1e-6)
        for name, value in ours.values().items():
            assert pyo.value(model.x[symbols[name]]) == pytest.approx(
                value, abs=1e-6, rel=1e-6)


# ---------------------------------------------------------------------------
# Block.from_flowsheet -- the bridge
# ---------------------------------------------------------------------------


class TestBlockFromFlowsheet:
    def test_defaults_u0_to_the_current_flowsheet(self, recycle_block):
        assert recycle_block.u0 == pytest.approx([1.5, 10.0])

    def test_derives_names_and_units(self, recycle_block):
        assert recycle_block.u_names == ["reactor_V", "feed_feed_total_flow"]
        assert recycle_block.y_names == ["purge_F_B", "purge_total_flow"]
        assert recycle_block.metadata["y_units"] == ["mol/s", "mol/s"]
        assert recycle_block.metadata["source"] == "flowsheet"

    def test_evaluates_to_the_flowsheet_solution(self, recycle_block,
                                                 recycle_flowsheet):
        streams = recycle_flowsheet.solve(tol=1e-10, max_iter=200)
        y = np.asarray(recycle_block.fn(jnp.asarray(recycle_block.u0)))

        assert y[0] == pytest.approx(float(streams["purge"]["F_B"]), rel=1e-6)

    def test_delta_vectors_match_finite_differences(self, recycle_block):
        """The decisive test: AD through a tear solve, checked numerically."""
        report = check_delta_vectors(recycle_block, rtol=1e-4)

        assert report["passed"], report

    def test_the_feed_lever_column_is_nonzero(self, recycle_block):
        lin = linearize_block(recycle_block)
        J = np.asarray(lin.J)
        j = recycle_block.u_names.index("feed_feed_total_flow")

        assert np.all(np.abs(J[:, j]) > 0.0)
        # More feed at a fixed reactor -> more of everything leaving.
        assert J[1, j] > 0.0

    def test_the_volume_column_has_the_right_sign(self, recycle_block):
        lin = linearize_block(recycle_block)
        J = np.asarray(lin.J)
        j = recycle_block.u_names.index("reactor_V")

        assert J[0, j] > 0.0, "a bigger reactor must make more B"

    def test_exports_through_the_full_stack(self, recycle_block, tmp_path):
        one = DeltaVectorSet.from_block(recycle_block, radius=0.1)
        path = write_json(one, tmp_path / "block.json")
        back = json.load(open(path, encoding="utf-8"))["vectors"][0]

        assert back["u_names"] == ["plant.reactor_V",
                                   "plant.feed_feed_total_flow"]
        # Units resolve exactly as the catalog resolves them: the field's
        # own metadata first, then the operation's `parameter_units`.
        # `CSTRParams.V` carries none of its own and the table has it, so
        # reading only the metadata left the axis unlabelled while the
        # editor's own picker showed `m^3`.
        assert back["u_units"] == ["m^3", "mol/s"]
        assert back["u_keys"] == ["reactor.V", "feed:feed.total_flow"]

    def test_rejects_an_empty_lever_list(self, recycle_flowsheet):
        with pytest.raises(ValueError, match="at least one lever"):
            Block.from_flowsheet(recycle_flowsheet, u=[], y=["purge.F_B"])

    def test_rejects_an_output_key_without_a_quantity(self, recycle_flowsheet):
        with pytest.raises(ValueError, match="<stream>.<quantity>"):
            Block.from_flowsheet(recycle_flowsheet, u=["reactor.V"],
                                 y=["purge"])

    def test_rejects_an_unknown_lever(self, recycle_flowsheet):
        with pytest.raises(KeyError):
            Block.from_flowsheet(recycle_flowsheet, u=["nope.V"],
                                 y=["purge.F_B"])


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


_PLAN_SCRIPT = '''
import warnings
from difflow.planning import chain
from difflow.planning.linearize import PhaseBoundaryWarning

warnings.simplefilter("ignore", PhaseBoundaryWarning)
result = chain.two_plant_chain().planner(radius=0.3).solve()
'''

_FLOWSHEET_SCRIPT = '''
import jax.numpy as jnp
from difflow import (CSTR, CSTRParams, Flowsheet, IdealThermo, SpeciesData,
                     Unit, make_stream)


def rate_fn(C, T, params):
    return jnp.array([params["A"] * jnp.exp(-params["Ea"] / (8.314 * T)) * C["A"]])


thermo = IdealThermo({
    s: SpeciesData(s, MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                   Hvap_coeffs=(35000.0, 0.38, 500.0),
                   antoine_coeffs=(10.0, 3000.0, -50.0))
    for s in ("A", "B")})

fs = Flowsheet(species_order=["A", "B"])
fs.add_feed("feed", make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0))
fs.add_unit(Unit("reactor", CSTR(CSTRParams(
    V=jnp.array(1.0), rate_fn=rate_fn, stoich=jnp.array([[-1.0], [1.0]]),
    rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
    species_order=["A", "B"]), thermo=thermo, mode="isothermal"),
    ["feed"], ["product"], params={"T_spec": 350.0}))
'''


@pytest.fixture
def plan_script(tmp_path):
    path = tmp_path / "plan_source.py"
    path.write_text(_PLAN_SCRIPT)
    return str(path)


@pytest.fixture
def flowsheet_script(tmp_path):
    path = tmp_path / "flowsheet_source.py"
    path.write_text(_FLOWSHEET_SCRIPT)
    return str(path)


class TestPlanExportCli:
    def test_json_goes_to_stdout(self, plan_script, capsys):
        from difflow.planning.cli import main

        assert main([plan_script]) == 0
        payload = json.loads(capsys.readouterr().out)

        assert {v["block"] for v in payload["vectors"]} == {"ngl", "power"}

    def test_dispatched_from_the_difflow_entry_point(self, plan_script,
                                                     tmp_path):
        from difflow.report.cli import main

        out = tmp_path / "plan.json"
        assert main(["plan-export", plan_script, "-o", str(out)]) == 0
        assert json.loads(out.read_text())["meta"]["converged"] is True

    def test_csv_writes_a_directory_of_tables(self, plan_script, tmp_path,
                                              capsys):
        from difflow.planning.cli import main

        out = tmp_path / "tables"
        assert main([plan_script, "--format", "csv", "-o", str(out)]) == 0
        assert (out / "ngl_jacobian.csv").exists()

    def test_mps_needs_the_planner_and_gets_it(self, plan_script, tmp_path):
        pytest.importorskip("pyomo")
        from difflow.planning.cli import main

        out = tmp_path / "plan.mps"
        assert main([plan_script, "--format", "mps", "-o", str(out)]) == 0
        assert "COLUMNS" in out.read_text()

    def test_a_bare_flowsheet_needs_levers(self, flowsheet_script, capsys):
        from difflow.planning.cli import main

        with pytest.raises(SystemExit, match="needs levers and outputs"):
            main([flowsheet_script])

    def test_a_flowsheet_with_levers_exports(self, flowsheet_script, tmp_path):
        from difflow.planning.cli import main

        out = tmp_path / "block.json"
        rc = main([flowsheet_script,
                   "-u", "reactor.V", "-u", "feed:feed.total_flow",
                   "-y", "product.F_B",
                   "--lb", "0.5,5", "--ub", "5,20",
                   "--radius", "0.2", "--name", "plant", "-o", str(out)])

        assert rc == 0
        payload = json.loads(out.read_text())
        vector = payload["vectors"][0]
        assert vector["block"] == "plant"
        assert vector["u_keys"] == ["reactor.V", "feed:feed.total_flow"]
        assert vector["radius"] == pytest.approx(0.2)
        assert np.shape(vector["J"]) == (1, 2)

    def test_lp_formats_refuse_a_source_with_no_planner(self, flowsheet_script,
                                                        tmp_path, capsys):
        from difflow.planning.cli import main

        rc = main([flowsheet_script, "-u", "reactor.V", "-y", "product.F_B",
                   "--format", "lp", "-o", str(tmp_path / "x.lp")])

        assert rc == 2
        assert "must run the planner" in capsys.readouterr().err

    def test_csv_without_an_output_path_is_an_error(self, plan_script, capsys):
        from difflow.planning.cli import main

        assert main([plan_script, "--format", "csv"]) == 2
        assert "needs -o/--output" in capsys.readouterr().err

    def test_an_empty_script_is_reported(self, tmp_path):
        from difflow.planning.cli import main

        path = tmp_path / "empty.py"
        path.write_text("x = 1\n")
        with pytest.raises(SystemExit, match="no PlanResult"):
            main([str(path)])
