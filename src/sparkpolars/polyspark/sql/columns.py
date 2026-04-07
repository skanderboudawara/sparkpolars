"""Polars Expr (Column) monkey-patches for PySpark API compatibility."""

import re as _re
from collections.abc import Callable
from typing import Any

import polars as pl
import polars.functions as polars_functions
from polars.expr import Expr

# ── cast ──────────────────────────────────────────────────────────────────────

Expr._original_cast = Expr.cast


def _cast_strict(self: Expr, dtype: Any, strict: bool = False) -> Expr:
    return self._original_cast(dtype, strict=strict)


Expr.cast = _cast_strict


# ── helpers ───────────────────────────────────────────────────────────────────

def _str_to_col(name: str | Expr) -> Expr:
    if isinstance(name, str):
        return pl.col(name)
    return name


# ── null checks ───────────────────────────────────────────────────────────────

def isNull(self: Expr) -> Expr:
    return self.is_null()


def isNotNull(self: Expr) -> Expr:
    return self.is_not_null()


Expr.isNull = isNull
Expr.isNotNull = isNotNull


# ── isin ──────────────────────────────────────────────────────────────────────

def isin(self: Expr, *cols: Any) -> Expr:
    if len(cols) == 1:
        return self.is_in(cols[0])
    return self.is_in(cols)


Expr.isin = isin


# ── between ───────────────────────────────────────────────────────────────────

def between(self: Expr, lowerBound: Any, upperBound: Any) -> Expr:
    return self.ge(lowerBound) & self.le(upperBound)


Expr.between = between


# ── eqNullSafe ────────────────────────────────────────────────────────────────

def eqNullSafe(self: Expr, other: str | Expr) -> Expr:
    return self.eq_missing(_str_to_col(other))


Expr.eqNullSafe = eqNullSafe


# ── string methods ────────────────────────────────────────────────────────────

def rlike(self: Expr, regexp: str) -> Expr:
    return self.str.contains(regexp, literal=False)


Expr.rlike = rlike


def startswith(self: Expr, prefix: str) -> Expr:
    return self.str.starts_with(prefix)


Expr.startswith = startswith


def endswith(self: Expr, suffix: str) -> Expr:
    return self.str.ends_with(suffix)


Expr.endswith = endswith


def substr(self: Expr, pos: int, length: int | None = None) -> Expr:
    if length is not None:
        return self.str.slice(pos - 1, length)
    return self.str.slice(pos - 1)


Expr.substr = substr


def contains(self: Expr, right: str) -> Expr:
    return self.str.contains(right)


Expr.contains = contains


# ── LIKE / ILIKE ──────────────────────────────────────────────────────────────

def _sql_like_to_regex(pattern: str) -> str:
    """Convert a SQL LIKE pattern (% = any chars, _ = one char) to a regex."""
    parts = _re.split(r"([%_])", pattern)
    regex_parts = ["^"]
    for part in parts:
        if part == "%":
            regex_parts.append(".*")
        elif part == "_":
            regex_parts.append(".")
        else:
            regex_parts.append(_re.escape(part))
    regex_parts.append("$")
    return "".join(regex_parts)


def like(self: Expr, pattern: str) -> Expr:
    return self.str.contains(_sql_like_to_regex(pattern), literal=False)


def ilike(self: Expr, pattern: str) -> Expr:
    return self.str.contains("(?i)" + _sql_like_to_regex(pattern), literal=False)


Expr.like = like
Expr.ilike = ilike


# ── isNaN ─────────────────────────────────────────────────────────────────────

def isNaN(self: Expr) -> Expr:
    return self.is_nan()


Expr.isNaN = isNaN


# ── astype ────────────────────────────────────────────────────────────────────

Expr.astype = Expr.cast


# ── bitwise ───────────────────────────────────────────────────────────────────

def bitwiseAND(self: Expr, other: Any) -> Expr:
    return self & other


def bitwiseOR(self: Expr, other: Any) -> Expr:
    return self | other


def bitwiseXOR(self: Expr, other: Any) -> Expr:
    return self ^ other


Expr.bitwiseAND = bitwiseAND
Expr.bitwiseOR = bitwiseOR
Expr.bitwiseXOR = bitwiseXOR


# ── sort direction ────────────────────────────────────────────────────────────
# These functions are defined in functions.py and imported here so that
# col("x").asc() / col("x").desc() work in addition to the standalone form.

def _get_desc_status(self: Expr) -> bool:
    return getattr(self, "_desc_status_value", False)


def _set_desc_status(self: Expr, status: bool) -> None:
    self._desc_status_value = status


Expr._desc_status = property(_get_desc_status, _set_desc_status)


def asc(self: Expr) -> Expr:
    self._desc_status = False
    return self.sort(descending=False, nulls_last=False)


def asc_nulls_first(self: Expr) -> Expr:
    self._desc_status = False
    return self.sort(descending=False, nulls_last=False)


def asc_nulls_last(self: Expr) -> Expr:
    self._desc_status = False
    return self.sort(descending=False, nulls_last=True)


def desc(self: Expr) -> Expr:
    self._desc_status = True
    return self.sort(descending=True, nulls_last=False)


def desc_nulls_first(self: Expr) -> Expr:
    self._desc_status = True
    return self.sort(descending=True, nulls_last=False)


def desc_nulls_last(self: Expr) -> Expr:
    self._desc_status = True
    return self.sort(descending=True, nulls_last=True)


Expr.asc = asc
Expr.asc_nulls_first = asc_nulls_first
Expr.asc_nulls_last = asc_nulls_last
Expr.desc = desc
Expr.desc_nulls_first = desc_nulls_first
Expr.desc_nulls_last = desc_nulls_last


# ── struct operations ─────────────────────────────────────────────────────────

def getField(self: Expr, name: str) -> Expr:
    return self.struct.field(name)


Expr.getField = getField


def withField(self: Expr, fieldName: str, col: Expr | str) -> Expr:
    return self.struct.with_fields(_str_to_col(col).alias(fieldName))


Expr.withField = withField


def dropFields(self: Expr, *fieldNames: str) -> Expr:
    msg = (
        "dropFields cannot be expressed as a pure Expr in Polars — "
        "use df.with_columns(pl.struct([pl.col('s').struct.field(f) "
        "for f in keep_fields]).alias('s')) instead."
    )
    raise NotImplementedError(msg)


Expr.dropFields = dropFields


# ── try_cast ──────────────────────────────────────────────────────────────────

def try_cast(self: Expr, dtype: Any) -> Expr:
    return self._original_cast(dtype, strict=False)


Expr.try_cast = try_cast


# ── transform (list columns) ──────────────────────────────────────────────────

def transform_list(self: Expr, f: Callable) -> Expr:
    return self.list.eval(f(pl.element()))


Expr.transform = transform_list


# ── getItem (JSON map) ────────────────────────────────────────────────────────

def getItem(self: Expr, key: str | Expr) -> Expr:
    """Extract a value from a JSON-encoded map column using a key."""
    import json

    import polars.datatypes as polars_datatypes

    key = _str_to_col(key)
    return polars_functions.struct([self.alias("json"), key.alias("key")]).map_elements(
        lambda x: json.loads(x["json"]).get(x["key"]),
        return_dtype=polars_datatypes.String,
    )


Expr.getItem = getItem


# ── over (window) ─────────────────────────────────────────────────────────────

from polars._utils.parse import parse_into_list_of_expressions  # noqa: E402


def over(
    self: Expr,
    partition_by: Any = None,
    *more_exprs: Any,
    order_by: Any = None,
    sort_by: Any = None,
) -> Expr:
    if hasattr(partition_by, "_partition_by"):
        window_partition_by = partition_by._partition_by
        window_order_by = partition_by._order_by
        window_sort_by = partition_by._sort_by
        if partition_by._partition_by is not None:
            partition_by = window_partition_by
        if order_by is None and window_order_by is not None:
            order_by = window_order_by
        if sort_by is None and window_sort_by is not None:
            sort_by = window_sort_by

    if partition_by is None:
        if order_by is not None and sort_by is not None:
            return self.sort_by(order_by, descending=sort_by)
        return self

    partition_by = parse_into_list_of_expressions(partition_by, *more_exprs)
    result = self._from_pyexpr(
        self._pyexpr.over(
            partition_by,
            order_by=None,
            order_by_descending=False,
            order_by_nulls_last=False,
            mapping_strategy="group_to_rows",
        ),
    )

    if order_by is not None and sort_by is not None:
        result = result.sort_by(order_by, descending=sort_by)

    return result


Expr.over = over


# ── outer (correlated subquery no-op) ─────────────────────────────────────────

def outer(self: Expr) -> Expr:
    """No-op marker for PySpark correlated-subquery syntax.

    PySpark uses ``col("x").outer()`` to reference an outer-query column inside
    a correlated subquery.  Polars has no correlated-subquery concept at the
    DataFrame API level — rewrite as a semi/anti join instead.
    """
    return self


Expr.outer = outer


# ── when / otherwise on Expr ──────────────────────────────────────────────────

class SparkWhen:
    def __init__(self, condition: Expr, value: Any) -> None:
        self._when_expr = polars_functions.when(condition).then(value)

    def when(self, condition: Expr, value: Any) -> "SparkWhen":
        self._when_expr = self._when_expr.when(condition).then(value)
        return self

    def otherwise(self, value: Any) -> Expr:
        return self._when_expr.otherwise(value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._when_expr, name)


def when_on_expr(self: Expr, condition: Expr, value: Any) -> SparkWhen:
    """Start a CASE WHEN chain.  Equivalent to sf.when(condition, value)."""
    return SparkWhen(condition, value)


def otherwise_on_expr(self: Expr, value: Any) -> Expr:
    """Return lit(value) for null rows, self for non-null rows."""
    return pl.when(self.is_not_null()).then(self).otherwise(pl.lit(value))


Expr.when = when_on_expr
Expr.otherwise = otherwise_on_expr


# ── alias / name ──────────────────────────────────────────────────────────────
# `alias` and `name` already exist natively on Expr; no patching needed.
