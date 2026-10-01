"""The crude unit: an assay in, a furnace, an atmospheric column, products out.

:class:`CrudeUnit` is the assembly a refinery planner means by "the CDU":
characterise the crude, heat it in the furnace and fractionate it, and
report the products as yields, gravities and boiling ranges. Everything it
does is one of the pieces -- :func:`~difflow_refinery.assay.characterize`,
:class:`~difflow_refinery.thermo.ColumnThermo`,
:class:`~difflow_refinery.column.CrudeColumn` with a
:class:`~difflow_refinery.column.Furnace`, and
:func:`~difflow_refinery.products.product_properties` -- so use those
directly for anything it does not cover.

It is differentiable with respect to the crude rate and inlet temperature,
every spec value, and the assay itself: :meth:`CrudeUnit.solve` takes an
``assay=`` that replaces the one the unit was built on (same light ends,
same cut points -- the cut points fix the number of pseudo-components, and
with it the shape of the column's equations).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import jax
from jax import Array

from difflow_refinery import products as prod
from difflow_refinery.assay import Assay, Characterization, characterize, default_cut_points
from difflow_refinery.column import BARREL, CrudeColumn, CrudeColumnParams, CrudeColumnResult, Furnace
from difflow_refinery.thermo import RHO_WATER_60F, ColumnThermo


@dataclass(frozen=True)
class CrudeUnitResult:
    """A solved crude unit.

    Attributes:
        column: The column's full :class:`CrudeColumnResult`.
        properties: :class:`~difflow_refinery.products.ProductProperties`
            for every hydrocarbon product, yields relative to the crude.
        feed: The crude feed stream (at the furnace inlet).
    """

    column: CrudeColumnResult
    properties: dict
    feed: dict

    @property
    def converged(self) -> Array:
        return self.column.converged

    @property
    def products(self) -> dict:
        return self.column.products

    def gaps(self, order=None) -> dict:
        """5-95 gaps (K) between neighbouring products (see :func:`~difflow_refinery.products.gaps`).

        ``order`` defaults to the products from lightest to heaviest by their
        50 % point, with any offgas left out.
        """
        if order is None:
            names = [n for n in self.properties if n != "offgas"]
            order = sorted(names, key=lambda n: float(self.properties[n].tbp_at(50)))
        return prod.gaps(self.properties, order)

    def table(self) -> str:
        """Yield table, then furnace and column duties."""
        c = self.column
        lines = [prod.table(self.properties), ""]
        lines.append(f"coil outlet {float(c.coil_outlet_T) - 273.15:.1f} C, "
                     f"{100 * float(c.feed_vaporized):.1f} mol% vaporised; "
                     f"furnace {float(c.furnace_duty) / 1e6:.1f} MW absorbed, "
                     f"{float(c.furnace_fired_duty) / 1e6:.1f} MW fired")
        pa = ", ".join(f"{float(q) / 1e6:.1f}" for q in c.pumparound_duty)
        lines.append(f"condenser {float(c.condenser_duty) / 1e6:.1f} MW"
                     + (f", pumparounds {pa} MW" if pa else ""))
        return "\n".join(lines)


jax.tree_util.register_dataclass(CrudeUnitResult, data_fields=["column", "properties", "feed"], meta_fields=[])


class CrudeUnit:
    """Furnace and atmospheric column, built on a crude assay.

    Args:
        assay: The crude's :class:`~difflow_refinery.assay.Assay`. Must be
            concrete (it fixes the cut points).
        params: The column's :class:`CrudeColumnParams`. A furnace is added
            (with default :class:`Furnace` settings) if it has none, so its
            specs must include one that closes the furnace: an overflash, a
            coil outlet temperature or a furnace duty.
        cut_points: Interior cut boundaries (K). Default:
            :func:`~difflow_refinery.assay.default_cut_points`.
        method: Critical-property correlation for the characterisation.

    Example:
        >>> from difflow_refinery import Assay, CrudeUnit, column as cc
        >>> assay = Assay([0, 30, 70, 100], [320., 480., 640., 900.], sg=0.85)
        >>> unit = CrudeUnit(assay, cc.CrudeColumnParams(
        ...     n_stages=8, feed_stage=7, bottom_steam=0.2,
        ...     specs=(cc.product_rate("naphtha", 0.25, basis="mole"),
        ...            cc.coil_outlet_temperature(620.0))))
        >>> res = unit.solve(1.0, T=500.0, P=4e5, basis="mole")
        >>> bool(res.converged)
        True
    """

    def __init__(self, assay: Assay, params: CrudeColumnParams,
                 cut_points=None, method: str = "twu"):
        self.assay = assay
        self.cut_points = tuple(default_cut_points(assay) if cut_points is None else cut_points)
        self.method = method
        if params.furnace is None:
            params = replace(params, furnace=Furnace())
        self.params = params
        self.crude = characterize(assay, self.cut_points, method)
        self.thermo = ColumnThermo.from_characterization(self.crude)
        self.column = CrudeColumn(params, self.thermo)

    def feed(self, rate, T, P, basis: Literal["volume", "mass", "mole", "bpd"] = "bpd",
             crude: Characterization | None = None) -> dict:
        """The crude feed stream at the furnace inlet.

        Args:
            rate: Crude rate: barrels per day (``"bpd"``), standard m^3/s
                (``"volume"``), kg/s or mol/s.
            T, P: Furnace inlet temperature (K) and pressure (Pa) -- the
                preheat train's outlet.
            crude: A characterisation to use instead of the unit's own.
        """
        crude = self.crude if crude is None else crude
        if basis == "bpd":
            rate, basis = rate * BARREL / 86400.0, "volume"
        if basis == "volume":
            rate, basis = rate * crude.bulk_sg * RHO_WATER_60F, "mass"
        if basis not in ("mass", "mole"):
            raise ValueError(f"basis must be 'bpd', 'volume', 'mass' or 'mole', not {basis!r}")
        return crude.stream(rate, T=T, P=P, basis=basis)

    def solve(self, rate, T, P, basis: Literal["volume", "mass", "mole", "bpd"] = "bpd",
              assay: Assay | None = None, params: CrudeColumnParams | None = None) -> CrudeUnitResult:
        """Solve the unit for a crude rate and furnace inlet condition.

        Args:
            rate, T, P, basis: As for :meth:`feed`.
            assay: A different assay to run (for a gradient with respect to
                the assay data). Characterised with this unit's cut points and
                method, so it needs the same light ends.
            params: Different column params (spec values, steam, pressures)
                with the same layout.
        """
        if assay is None:
            crude, thermo = self.crude, self.thermo
        else:
            crude = characterize(assay, self.cut_points, self.method)
            if crude.names != self.crude.names:
                raise ValueError("the assay characterises into different components "
                                 f"({len(crude.names)} vs {len(self.crude.names)}); "
                                 "it needs the unit's light ends")
            thermo = ColumnThermo.from_characterization(crude)
        column = self.column
        if params is not None:
            if params.furnace is None:
                params = replace(params, furnace=self.params.furnace)
            column = CrudeColumn(params, thermo)
        elif assay is not None:
            column = CrudeColumn(self.params, thermo)
        feed = self.feed(rate, T, P, basis, crude=crude)
        result = column.solve(feed)
        return CrudeUnitResult(column=result,
                               properties=prod.product_properties(result.products, thermo, feed),
                               feed=feed)


__all__ = ["CrudeUnit", "CrudeUnitResult"]
