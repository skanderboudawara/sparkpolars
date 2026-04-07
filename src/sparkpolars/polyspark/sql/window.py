"""PySpark Window API compatibility layer backed by Polars."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import polars.datatypes as _dt
from polars.expr import Expr

if TYPE_CHECKING:
    pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_expr(c: Any) -> Expr:
    if isinstance(c, str):
        return pl.col(c)
    return c


def _parse_order_cols(*cols: Any) -> tuple[list[Expr], list[bool]]:
    """Return (exprs, descending_flags) for orderBy columns."""
    exprs: list[Expr] = []
    desc: list[bool] = []
    for c in cols:
        if isinstance(c, (list, tuple)):
            e2, d2 = _parse_order_cols(*c)
            exprs.extend(e2)
            desc.extend(d2)
        else:
            e = _to_expr(c)
            exprs.append(e)
            desc.append(bool(getattr(e, "_desc_status", False)))
    return exprs, desc


def _to_pyexprs(exprs: list[Expr]) -> list[Any]:
    """Convert Python Expr objects to internal PyExpr objects."""
    return [e._pyexpr for e in exprs]


# ── WindowSpec ────────────────────────────────────────────────────────────────

class WindowSpec:
    """Immutable window specification: partition + order + frame."""

    __slots__ = ("_partition_by", "_order_by", "_order_by_desc")

    def __init__(
        self,
        partition_by: list[Expr] | None = None,
        order_by: list[Expr] | None = None,
        order_by_desc: list[bool] | None = None,
    ) -> None:
        self._partition_by: list[Expr] = partition_by or []
        self._order_by: list[Expr] = order_by or []
        self._order_by_desc: list[bool] = order_by_desc or []

    def partitionBy(self, *cols: Any) -> "WindowSpec":
        pb = [_to_expr(c) for c in cols]
        return WindowSpec(pb, self._order_by, self._order_by_desc)

    def orderBy(self, *cols: Any) -> "WindowSpec":
        ob, desc = _parse_order_cols(*cols)
        return WindowSpec(self._partition_by, ob, desc)

    # frame bounds — accepted for API compatibility, no-op in Polars
    def rowsBetween(self, start: int, end: int) -> "WindowSpec":  # noqa: ARG002
        return self

    def rangeBetween(self, start: int, end: int) -> "WindowSpec":  # noqa: ARG002
        return self


# ── Window ────────────────────────────────────────────────────────────────────

class Window:
    """
    PySpark-compatible Window factory.

    Usage::

        Window.partitionBy("dept").orderBy("salary")
        Window.partitionBy("dept").orderBy(col("salary").desc())
        Window.orderBy("salary")
    """

    unboundedPreceding: int = -(2 ** 31)
    unboundedFollowing: int = 2 ** 31 - 1
    currentRow: int = 0

    @staticmethod
    def partitionBy(*cols: Any) -> WindowSpec:
        return WindowSpec().partitionBy(*cols)

    @staticmethod
    def orderBy(*cols: Any) -> WindowSpec:
        return WindowSpec().orderBy(*cols)

    @staticmethod
    def rowsBetween(start: int, end: int) -> WindowSpec:
        return WindowSpec()

    @staticmethod
    def rangeBetween(start: int, end: int) -> WindowSpec:
        return WindowSpec()


# ── _WindowFuncExpr ───────────────────────────────────────────────────────────

class _WindowFuncExpr:
    """
    Deferred window function expression.

    Returned by ``rank()``, ``dense_rank()``, ``row_number()``, ``lag()``,
    ``lead()``, ``ntile()``, ``cume_dist()``, ``percent_rank()``.

    Call ``.over(window_spec)`` to materialise into a real Polars Expr.
    Optionally call ``.alias(name)`` before or after ``.over()``.
    """

    def __init__(self, fn: str, *args: Any) -> None:
        self._fn = fn
        self._args = args
        self._alias_name: str | None = None

    # ── fluent helpers ────────────────────────────────────────────────────────

    def alias(self, name: str) -> "_WindowFuncExpr":
        self._alias_name = name
        return self

    # ── materialise ───────────────────────────────────────────────────────────

    def over(self, window_spec: WindowSpec) -> Expr:
        """Build the real Polars expression using *window_spec*."""
        ws: WindowSpec
        if isinstance(window_spec, WindowSpec):
            ws = window_spec
        else:
            # Legacy: plain string / Expr list passed directly
            pb = [_to_expr(window_spec)]
            ws = WindowSpec(partition_by=pb)

        pb = ws._partition_by
        ob = ws._order_by
        desc = ws._order_by_desc

        expr = self._build(pb, ob, desc)
        if self._alias_name:
            expr = expr.alias(self._alias_name)
        return expr

    def _build(
        self,
        partition_by: list[Expr],
        order_by: list[Expr],
        descending: list[bool],
    ) -> Expr:
        fn = self._fn

        def _over(base: Expr, pb: list[Expr], ob: list[Expr], desc: list[bool]) -> Expr:
            """Apply .over() using the real Polars API"""
            if not pb:
                return base
            # _polars_over is the original Polars Expr.over() saved before our patch.
            # Polars 1.x signature: over(partition_by, order_by=, descending=bool, ...)
            if ob:
                return base._polars_over(
                    pb,
                    order_by=ob,
                    descending=desc[0] if desc else False,
                    mapping_strategy="group_to_rows",
                )
            return base._polars_over(pb, mapping_strategy="group_to_rows")

        # ── row_number ────────────────────────────────────────────────────────
        if fn == "row_number":
            base = pl.int_range(pl.len(), dtype=_dt.Int64())
            if partition_by:
                base = _over(base, partition_by, order_by, descending)
            elif order_by:
                base = base.sort_by(order_by, descending=descending)
            return base + 1

        # ── rank / dense_rank ─────────────────────────────────────────────────
        if fn in ("rank", "dense_rank"):
            method = "min" if fn == "rank" else "dense"
            if order_by:
                col_to_rank = order_by[0] if len(order_by) == 1 else pl.struct(order_by)
                base = col_to_rank.rank(method).cast(_dt.Int64())
            else:
                base = pl.lit(1).cast(_dt.Int64())
            return _over(base, partition_by, [], [])

        # ── lag ───────────────────────────────────────────────────────────────
        if fn == "lag":
            col_expr, offset, default = self._args
            base = col_expr.shift(offset, fill_value=default)
            return _over(base, partition_by, [], [])

        # ── lead ──────────────────────────────────────────────────────────────
        if fn == "lead":
            col_expr, offset, default = self._args
            base = col_expr.shift(-offset, fill_value=default)
            return _over(base, partition_by, [], [])

        # ── ntile ─────────────────────────────────────────────────────────────
        if fn == "ntile":
            (n,) = self._args
            length = pl.len()
            rank_e = pl.int_range(start=0, end=length, dtype=_dt.Int64())
            base = (rank_e * n // length + 1).cast(_dt.Int32())
            return _over(base, partition_by, order_by, descending)

        # ── cume_dist ─────────────────────────────────────────────────────────
        if fn == "cume_dist":
            length = pl.len()
            rank_e = pl.int_range(start=1, end=length + 1, dtype=_dt.Int64())
            base = rank_e.cast(_dt.Float64()) / length.cast(_dt.Float64())
            return _over(base, partition_by, [], [])

        # ── percent_rank ──────────────────────────────────────────────────────
        if fn == "percent_rank":
            length = pl.len()
            rank_e = pl.int_range(start=0, end=length, dtype=_dt.Int64())
            base = pl.when(length <= 1).then(pl.lit(0.0)).otherwise(
                rank_e.cast(_dt.Float64()) / (length - 1).cast(_dt.Float64())
            )
            return _over(base, partition_by, [], [])

        raise NotImplementedError(f"Window function '{fn}' is not implemented")
