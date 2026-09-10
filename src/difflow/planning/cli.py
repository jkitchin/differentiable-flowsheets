"""``difflow plan-export`` -- write delta vectors out to a planning system.

The point of this command is that the person who owns the planning model is
usually not the person who owns the simulation.  They want a table of
coefficients and a base case, not a JAX install.  This turns any of the
objects difflow's planning layer produces into one.

Usage::

    # A script that builds a Flowsheet; name the levers and the outputs.
    difflow plan-export model.py -u reactor.V -u 'feed:F1.total_flow' \\
        -y product.F_B -y product.total_flow --lb 0.5,5 --ub 5,20 \\
        --format csv -o tables/

    # A serialized flowsheet, with the levers taken from its own view.planning.
    difflow plan-export flowsheet.json --format json -o plan.json

    # A script that builds a DeltaBasePlanner or holds a solved PlanResult.
    difflow plan-export plan.py --format mps -o plan.mps

The source may be a ``.py`` script -- run, then searched for the first
:class:`~difflow.planning.planner.PlanResult`,
:class:`~difflow.planning.planner.DeltaBasePlanner`,
:class:`~difflow.planning.block.Block` or
:class:`~difflow.flowsheet.Flowsheet`, in that order -- or a ``.json``
flowsheet written by :func:`difflow.serialize.save`.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from typing import Any, Sequence

__all__ = ["main"]


def _floats(text: str | None) -> list[float] | None:
    """Parse a comma-separated bound list."""
    if text is None:
        return None
    return [float(part) for part in text.split(",") if part.strip() != ""]


def _find(namespace: dict, kinds: Sequence[type]) -> Any:
    """First value in ``namespace`` that is an instance of any of ``kinds``."""
    for kind in kinds:
        for value in namespace.values():
            if isinstance(value, kind):
                return value
    return None


def _block_from_flowsheet(flowsheet, args, planning: dict | None):
    """Build the planning block from CLI flags, falling back to the file."""
    from difflow.planning.block import Block

    planning = planning or {}
    u = list(args.u) or list(planning.get("u", []))
    y = list(args.y) or list(planning.get("y", []))
    if not u or not y:
        raise SystemExit(
            "a flowsheet needs levers and outputs: pass -u/-y, or save the "
            "flowsheet with a view.planning selection from the GUI")

    bounds = planning.get("bounds") or {}
    lb = _floats(args.lb) or [bounds.get(k, [None, None])[0] for k in u]
    ub = _floats(args.ub) or [bounds.get(k, [None, None])[1] for k in u]
    lb = None if any(v is None for v in lb) else lb
    ub = None if any(v is None for v in ub) else ub

    return Block.from_flowsheet(flowsheet, u=u, y=y, name=args.name,
                                lb=lb, ub=ub)


def _load_source(args):
    """Resolve the source into ``(delta_vector_set, plan_result_or_None)``."""
    from difflow.flowsheet import Flowsheet
    from difflow.planning.block import Block
    from difflow.planning.export import DeltaVectorSet
    from difflow.planning.planner import DeltaBasePlanner, PlanResult

    radius = float(args.radius)

    if args.source.endswith(".json"):
        from difflow import serialize

        with open(args.source, encoding="utf-8") as fh:
            data = json.load(fh)
        flowsheet = serialize.from_dict(data)
        planning = (data.get("view") or {}).get("planning")
        block = _block_from_flowsheet(flowsheet, args, planning)
        return DeltaVectorSet.from_block(block, radius=radius,
                                         source_file=args.source), None

    ns = runpy.run_path(args.source, run_name="__difflow_plan_export__")

    result = _find(ns, [PlanResult])
    if result is not None:
        return DeltaVectorSet.from_result(result), result

    planner = _find(ns, [DeltaBasePlanner])
    if planner is not None:
        result = planner.solve()
        return DeltaVectorSet.from_result(result), result

    block = _find(ns, [Block])
    if block is not None:
        return DeltaVectorSet.from_block(block, radius=radius,
                                         source_file=args.source), None

    flowsheet = _find(ns, [Flowsheet])
    if flowsheet is not None:
        built = _block_from_flowsheet(flowsheet, args, None)
        return DeltaVectorSet.from_block(built, radius=radius,
                                         source_file=args.source), None

    raise SystemExit(
        f"no PlanResult, DeltaBasePlanner, Block or Flowsheet found in "
        f"{args.source}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="difflow plan-export",
        description="Export delta vectors for an LP-style planning system.")
    parser.add_argument(
        "source",
        help="A .py script that builds the model, or a serialized "
             "flowsheet .json")
    parser.add_argument(
        "--format", choices=("json", "csv", "lp", "mps", "iterations"),
        default="json",
        help="json: one manifest. csv: a directory of tables. lp/mps: the "
             "assembled LP. iterations: the trust-region audit trail.")
    parser.add_argument(
        "-o", "--output",
        help="Output path (a directory for --format csv). Defaults to stdout "
             "for json, and is required otherwise.")
    parser.add_argument(
        "-u", action="append", default=[], metavar="KEY",
        help="A lever, '<unit>.<param>' or 'feed:<stream>.<field>'. Repeat.")
    parser.add_argument(
        "-y", action="append", default=[], metavar="KEY",
        help="An output, '<stream>.<quantity>'. Repeat.")
    parser.add_argument("--lb", help="Comma-separated lever lower bounds")
    parser.add_argument("--ub", help="Comma-separated lever upper bounds")
    parser.add_argument(
        "--radius", type=float, default=0.0,
        help="Trust-region radius to record, as a fraction of each lever's "
             "bound range (default: 0, meaning none declared)")
    parser.add_argument("--name", default="flowsheet", help="Block name")
    parser.add_argument(
        "--scaled", action="store_true",
        help="Write the dimensionless Jacobian in the CSV tables")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command.  Returns a process exit code."""
    from difflow.planning.export import (
        write_csv, write_iterations_csv, write_json, write_lp, write_mps,
    )

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "plan-export":
        argv = argv[1:]
    args = _parser().parse_args(argv)

    dvs, result = _load_source(args)

    if args.format == "json":
        if args.output:
            write_json(dvs, args.output)
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            json.dump(dvs.to_dict(), sys.stdout, indent=2)
            sys.stdout.write("\n")
        return 0

    if not args.output:
        print(f"error: --format {args.format} needs -o/--output",
              file=sys.stderr)
        return 2

    if args.format == "csv":
        for path in write_csv(dvs, args.output, scaled=args.scaled):
            print(path, file=sys.stderr)
        return 0

    if args.format == "iterations":
        if result is None:
            print("error: --format iterations needs a source that runs the "
                  "planner (a PlanResult or a DeltaBasePlanner)",
                  file=sys.stderr)
            return 2
        write_iterations_csv(result, args.output)
        print(f"wrote {args.output}", file=sys.stderr)
        return 0

    if result is None:
        print(f"error: --format {args.format} needs an assembled LP, so the "
              "source must run the planner (a PlanResult or a "
              "DeltaBasePlanner)", file=sys.stderr)
        return 2

    writer = write_lp if args.format == "lp" else write_mps
    writer(result.lp_model, args.output)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
