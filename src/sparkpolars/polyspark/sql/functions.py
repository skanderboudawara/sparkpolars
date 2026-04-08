"""Polars SQL functions for Polyspark."""

import datetime
from collections.abc import Callable
from functools import reduce
from typing import Any

import polars as pl
import polars.datatypes as polars_datatypes
import polars.functions as polars_functions
from polars import lit
from polars.expr import Expr

# Apply all Expr (Column) monkey-patches first so every function in this
# module can rely on them being in place.
from . import columns as _columns  # noqa: F401 — side-effects only
from .columns import SparkWhen, _str_to_col


def broadcast(df: Any) -> Any:
    return df


col = polars_functions.col
column = col
Column = col


def trim(col: str | Expr, trim_chars: str | None = None) -> Expr:
    return _str_to_col(col).trim(trim_chars)


def when(condition: Expr, value: Any) -> SparkWhen:
    return SparkWhen(condition, value)


def concat_ws(separator: str, *cols: Any) -> Expr:
    exprs = []
    for c in cols:
        if hasattr(c, "_expr"):
            exprs.append(c)
        else:
            c_ = _str_to_col(c)
            exprs.append(c_)
    if len(exprs) == 1:
        return exprs[0].list.join(separator)
    return polars_functions.concat_str(*exprs, separator=separator)


def expr(str: str) -> Expr:
    return polars_functions.sql_expr(str)


def upper(col: str | Expr) -> Expr:
    return _str_to_col(col).upper()


ucase = upper


def lower(col: str | Expr) -> Expr:
    return _str_to_col(col).lower()


lcase = lower


def regexp_count(col: str | Expr, pattern: str) -> Expr:
    return _str_to_col(col).regexp_count(pattern)


def regexp_extract(col: str | Expr, pattern: str, idx: int) -> Expr:
    return _str_to_col(col).regexp_extract(pattern, idx)


def regexp_extract_all(col: str | Expr, pattern: str, idx: int | None = None) -> Expr:
    if idx is not None:
        msg = "idx parameter is not supported in Polars for regexp_extract_all"
        raise NotImplementedError(msg)
    return _str_to_col(col).regexp_extract_all(pattern)


def regexp_replace(col: str | Expr, pattern: str, replacement: str) -> Expr:
    return _str_to_col(col).regexp_replace(pattern, replacement)


def replace(col: str | Expr, search: str, replacement: str | None = None) -> Expr:
    return _str_to_col(col).str_replace(search, replacement)


def rtrim(col: str | Expr, trim_chars: str | None = None) -> Expr:
    return _str_to_col(col).rtrim(trim_chars)


def ltrim(col: str | Expr, trim_chars: str | None = None) -> Expr:
    return _str_to_col(col).ltrim(trim_chars)


def rpad(col: str | Expr, length: int, pad: str = " ") -> Expr:
    return _str_to_col(col).rpad(length, pad)


def lpad(col: str | Expr, length: int, pad: str = " ") -> Expr:
    return _str_to_col(col).lpad(length, pad)


def base64(col: str | Expr) -> Expr:
    return _str_to_col(col).base64()


def btrim(col: str | Expr, trim_chars: str | None = None) -> Expr:
    return _str_to_col(col).btrim(trim_chars)


def contains(col: str | Expr, substr: str) -> Expr:
    return _str_to_col(col).contains(substr)


def encode(col: str | Expr, charset: str) -> Expr:
    return _str_to_col(col).encode(charset)


def decode(col: str | Expr, charset: str) -> Expr:
    return _str_to_col(col).decode(charset)


def left(col: str | Expr, n: int) -> Expr:
    return _str_to_col(col).left(n)


def right(col: str | Expr, n: int) -> Expr:
    return _str_to_col(col).right(n)


def length(col: str | Expr) -> Expr:
    return _str_to_col(col).length()


def locate(substr: str, col: str | Expr, _pos: int | None = None) -> Expr:
    return _str_to_col(col).locate(substr)


position = locate


def repeat(col: str | Expr, n: int) -> Expr:
    return _str_to_col(col).repeat(n)


def concat(*cols: str | Expr) -> Expr:
    return reduce(lambda acc, col: acc + col.cast(polars_datatypes.String(), strict=False), cols)


def round(col: str | Expr, scale: int = 0) -> Expr:
    return col.round(scale)


def sqrt(col: str | Expr) -> Expr:
    return col.sqrt()


def pow(col: str | Expr, exponent: int | float) -> Expr:
    return col.pow(exponent)


power = pow


def negate(col: str | Expr) -> Expr:
    return _str_to_col(col).negate()


negative = negate


def translate(col: str | Expr, matching: str, replace: str) -> Expr:
    return _str_to_col(col).translate(matching, replace)


def collect_list(col: str | Expr) -> Expr:
    return _str_to_col(col).collect_list()


array_agg = collect_list
listagg = collect_list


def collect_set(col: str | Expr) -> Expr:
    return _str_to_col(col).collect_set()


array_agg_distinct = collect_set
listagg_distinct = collect_set


def _col_name(col: str | Expr) -> str:
    """Extract a column name string for building PySpark-style agg aliases."""
    if isinstance(col, str):
        return col
    try:
        return col.meta.output_name()
    except Exception:
        return "col"


def sum(col: str | Expr) -> Expr:
    return _str_to_col(col).sum().alias(f"sum({_col_name(col)})")


def min(col: str | Expr) -> Expr:
    return _str_to_col(col).min().alias(f"min({_col_name(col)})")


def max(col: str | Expr) -> Expr:
    return _str_to_col(col).max().alias(f"max({_col_name(col)})")


def abs(col: str | Expr) -> Expr:
    """Absolute value — scalar transform, keeps input column name."""
    return _str_to_col(col).abs()


def avg(col: str | Expr) -> Expr:
    return _str_to_col(col).mean().alias(f"avg({_col_name(col)})")


def first(col: str | Expr, ignorenulls: bool = False) -> Expr:
    e = _str_to_col(col)
    base = e.drop_nulls().first() if ignorenulls else e.first()
    return base.alias(f"first({_col_name(col)})")


first_value = first


def last(col: str | Expr, ignorenulls: bool = False) -> Expr:
    e = _str_to_col(col)
    base = e.drop_nulls().last() if ignorenulls else e.last()
    return base.alias(f"last({_col_name(col)})")


last_value = last


def array_distinct(col: str | Expr) -> Expr:
    return _str_to_col(col).array_distinct()


def greatest(*cols: str | Expr) -> Expr:
    cols = [_str_to_col(col) for col in cols]
    return polars_functions.max_horizontal(*cols)


def least(*cols: str | Expr) -> Expr:
    cols = [_str_to_col(col) for col in cols]
    return polars_functions.min_horizontal(*cols)


# Sort-direction standalone helpers — used by orderBy/_desc_status logic in dataframe.py.
# The Expr method variants (col("a").asc() etc.) are patched in columns.py.
from .columns import (  # noqa: E402
    asc,
    asc_nulls_first,
    asc_nulls_last,
    desc,
    desc_nulls_first,
    desc_nulls_last,
)


def array(*cols: str | Expr) -> Expr:
    if not cols:
        return polars_functions.concat_list(polars_functions.lit([]))
    cols = [_str_to_col(col) for col in cols]
    return polars_functions.concat_list(*cols)


def array_append(col: str | Expr, value: Any) -> Expr:
    return _str_to_col(col).array_append(value)


def array_compact(col: str | Expr) -> Expr:
    return _str_to_col(col).array_compact()


def array_contains(col: str | Expr, value: Any) -> Expr:
    return _str_to_col(col).array_contains(value)


def array_except(col1: str | Expr, col2: str | Expr) -> Expr:
    return _str_to_col(col1).array_except(_str_to_col(col2))


def array_intersect(col1: str | Expr, col2: str | Expr) -> Expr:
    return _str_to_col(col1).array_intersect(_str_to_col(col2))


def array_join(col: str | Expr, delimiter: str, null_replacement: str | None = None) -> Expr:
    return _str_to_col(col).array_join(delimiter, null_replacement)


def array_max(col: str | Expr) -> Expr:
    return _str_to_col(col).array_max()


def array_min(col: str | Expr) -> Expr:
    return _str_to_col(col).array_min()


def array_size(col: str | Expr) -> Expr:
    return _str_to_col(col).array_size()


size = array_size


def array_union(col1: str | Expr, col2: str | Expr) -> Expr:
    return _str_to_col(col1).array_union(_str_to_col(col2))


def array_sort(col: str | Expr, comparator: Any = None) -> Expr:
    if comparator is not None:
        msg = "Custom comparator for array_sort is not supported in Polars."
        raise NotImplementedError(msg)
    return _str_to_col(col).array_sort(asc=True)


def sort_array(col: str | Expr, asc: bool = True) -> Expr:
    return _str_to_col(col).array_sort(asc=asc)


def slice(col: str | Expr, start: int, length: int | None = None) -> Expr:
    return _str_to_col(col).array_slice(start, length)


def array_remove(col: str | Expr, element: Any) -> Expr:
    return _str_to_col(col).array_remove(element)


def flatten(col: str | Expr) -> Expr:
    # Native Expr.flatten() flattens List[List[T]] → List[T] per row
    c = _str_to_col(col)
    return c.flatten()


def split(col: str | Expr, pattern: str, limit: int = -1) -> Expr:
    return _str_to_col(col).split(pattern, limit)


def explode(col: str | Expr) -> Expr:
    """Mark an expression as needing explode treatment."""
    col_expr = _str_to_col(col)
    # Add a special marker to indicate this should use DataFrame.explode()
    col_expr._explode_marker = True
    col_expr._explode_outer = False
    col_expr._explode_column = col if isinstance(col, str) else col.name
    return col_expr


def explode_outer(col: str | Expr) -> Expr:
    """Mark an expression as needing explode treatment with outer join."""
    col_expr = _str_to_col(col)
    # Add a special marker to indicate this should use DataFrame.explode()
    col_expr._explode_marker = True
    col_expr._explode_outer = True
    col_expr._explode_column = col if isinstance(col, str) else col.name
    return col_expr


def count(col: str | Expr = "*") -> Expr:
    name = col if isinstance(col, str) else _col_name(col)
    if col == "*" or col == "1":
        return pl.len().alias("count(1)")
    return _str_to_col(col).count().alias(f"count({name})")


coalesce = polars_functions.coalesce


def monotonically_increasing_id() -> Expr:
    return polars_functions.int_range(pl.len(), dtype=pl.UInt64)


def product(col: str | Expr) -> Expr:
    return _str_to_col(col).product()


def year(col: str | Expr) -> Expr:
    return _str_to_col(col).year()


def month(col: str | Expr) -> Expr:
    return _str_to_col(col).month()


def hour(col: str | Expr) -> Expr:
    return _str_to_col(col).hour()


def last_day(col: str | Expr) -> Expr:
    return _str_to_col(col).last_day()


def dayofmonth(col: str | Expr) -> Expr:
    return _str_to_col(col).dayofmonth()


def dayofweek(col: str | Expr) -> Expr:
    return _str_to_col(col).dayofweek()


def dayofyear(col: str | Expr) -> Expr:
    return _str_to_col(col).dayofyear()


def current_date() -> Expr:
    return polars_functions.lit(datetime.datetime.now().date(), dtype=polars_datatypes.Date)  # noqa: DTZ005


now = current_date
curdate = current_date


def current_timestamp() -> Expr:
    return polars_functions.lit(datetime.datetime.now(), dtype=polars_datatypes.Datetime)  # noqa: DTZ005


localtimestamp = current_timestamp


def date_sub(col: str | Expr, days: int) -> Expr:
    return _str_to_col(col).date_sub(days)


def date_add(col: str | Expr, days: int) -> Expr:
    return _str_to_col(col).date_add(days)


def datediff(end: str | Expr, start: str | Expr) -> Expr:
    end = _str_to_col(end)
    start = _str_to_col(start)
    return end - start


def add_months(col: str | Expr, months: int) -> Expr:
    return _str_to_col(col).add_months(months)


def sequence(start: int | Expr, stop: int | Expr, step: int | None = None) -> Expr:
    start = _str_to_col(start)
    stop = _str_to_col(stop)
    if step is None:
        return polars_functions.int_ranges(start, stop + 1)
    return polars_functions.int_ranges(start, stop + 1, step=step)


def create_map(dict: dict) -> Expr:
    return polars_functions.struct(**{k: polars_functions.lit(v) for k, v in dict.items()}).struct.json_encode()


def md5(col: str | Expr) -> Expr:
    return _str_to_col(col).md5()


def sha1(col: str | Expr) -> Expr:
    return _str_to_col(col).sha1()


def sha256(col: str | Expr) -> Expr:
    return _str_to_col(col).sha256()


def transform(col: str | Expr, f: Callable) -> Expr:
    """Transform each element in a list column using a function."""
    return _str_to_col(col).transform(f)


def filter(col: str | Expr, f: Callable) -> Expr:
    """Filter elements in a list column based on a predicate."""
    return _str_to_col(col).list_filter(f)


def forall(col: str | Expr, f: Callable) -> Expr:
    """Check if all elements in a list column satisfy a predicate."""
    return _str_to_col(col).forall(f)


def reverse(col: str | Expr) -> Expr:
    return _str_to_col(col).reverse()


# ── missing standalone aliases ────────────────────────────────────────────────

def substr(col: str | Expr, pos: int, length: int | None = None) -> Expr:
    return _str_to_col(col).substr(pos, length)


substring = substr


def struct(*cols: str | Expr) -> Expr:
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.struct(exprs)


# ── math functions ────────────────────────────────────────────────────────────

def ceil(col: str | Expr) -> Expr:
    return _str_to_col(col).ceil()


def floor(col: str | Expr) -> Expr:
    return _str_to_col(col).floor()


def log(col: str | Expr, base: float = 2.718281828459045) -> Expr:
    """Natural log by default; pass base= for other bases."""
    return _str_to_col(col).log(base)


def log2(col: str | Expr) -> Expr:
    return _str_to_col(col).log(2.0)


def log10(col: str | Expr) -> Expr:
    return _str_to_col(col).log(10.0)


def exp(col: str | Expr) -> Expr:
    return _str_to_col(col).exp()


def signum(col: str | Expr) -> Expr:
    return _str_to_col(col).sign()


sign = signum


def cbrt(col: str | Expr) -> Expr:
    return _str_to_col(col) ** (1.0 / 3.0)


# ── null-handling functions ───────────────────────────────────────────────────

def nvl(col: str | Expr, replacement: Any) -> Expr:
    """Return replacement when col is null (alias: ifnull)."""
    c = _str_to_col(col)
    r = replacement if isinstance(replacement, Expr) else lit(replacement)
    return polars_functions.coalesce(c, r)


ifnull = nvl


def nullif(col: str | Expr, value: Any) -> Expr:
    """Return null when col equals value, else col."""
    return _str_to_col(col).nullif(value)


# ── date / time functions ─────────────────────────────────────────────────────

def to_date(col: str | Expr, fmt: str = "%Y-%m-%d") -> Expr:
    return _str_to_col(col).str.to_date(fmt)


def to_timestamp(col: str | Expr, fmt: str = "%Y-%m-%d %H:%M:%S") -> Expr:
    return _str_to_col(col).str.to_datetime(fmt)


def date_format(col: str | Expr, fmt: str) -> Expr:
    return _str_to_col(col).dt.strftime(fmt)


def unix_timestamp(col: str | Expr) -> Expr:
    """Seconds since Unix epoch (1970-01-01 00:00:00 UTC)."""
    return _str_to_col(col).dt.epoch(time_unit="s")


def from_unixtime(col: str | Expr) -> Expr:
    """Convert integer seconds since epoch to a Datetime."""
    return polars_functions.from_epoch(_str_to_col(col), time_unit="s")


# ── string helper ─────────────────────────────────────────────────────────────

def format_string(fmt: str, *cols: str | Expr) -> Expr:
    """Equivalent to PySpark format_string / printf-style interpolation."""
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.format(fmt.replace("%s", "{}").replace("%d", "{}"), *exprs)


# ── array helpers ─────────────────────────────────────────────────────────────

def array_position(col: str | Expr, value: Any) -> Expr:
    """Return the 1-based index of value in the array, or 0 if not found."""
    c = _str_to_col(col)
    v = value if isinstance(value, Expr) else lit(value)
    # Polars list.eval gives 0-based index via arg_where; +1 for Spark compatibility
    idx = c.list.eval(pl.arg_where(pl.element() == v).first()).list.first()
    return polars_functions.when(c.list.contains(v)).then(idx + 1).otherwise(lit(0))


def array_prepend(col: str | Expr, value: Any) -> Expr:
    return _str_to_col(col).array_prepend(value)


# ── trig functions ────────────────────────────────────────────────────────────

def sin(col: str | Expr) -> Expr:
    return _str_to_col(col).sin()


def cos(col: str | Expr) -> Expr:
    return _str_to_col(col).cos()


def tan(col: str | Expr) -> Expr:
    return _str_to_col(col).tan()


def asin(col: str | Expr) -> Expr:
    return _str_to_col(col).arcsin()


def acos(col: str | Expr) -> Expr:
    return _str_to_col(col).arccos()


def atan(col: str | Expr) -> Expr:
    return _str_to_col(col).arctan()


def atan2(y: str | Expr, x: str | Expr) -> Expr:
    import math
    y_expr = _str_to_col(y).cast(polars_datatypes.Float64(), strict=False)
    x_expr = _str_to_col(x).cast(polars_datatypes.Float64(), strict=False)
    return polars_functions.struct([y_expr.alias("_y"), x_expr.alias("_x")]).map_elements(
        lambda row: math.atan2(row["_y"], row["_x"]),
        return_dtype=polars_datatypes.Float64(),
    )


def sinh(col: str | Expr) -> Expr:
    return _str_to_col(col).sinh()


def cosh(col: str | Expr) -> Expr:
    return _str_to_col(col).cosh()


def tanh(col: str | Expr) -> Expr:
    return _str_to_col(col).tanh()


def asinh(col: str | Expr) -> Expr:
    return _str_to_col(col).arcsinh()


def acosh(col: str | Expr) -> Expr:
    return _str_to_col(col).arccosh()


def atanh(col: str | Expr) -> Expr:
    return _str_to_col(col).arctanh()


def degrees(col: str | Expr) -> Expr:
    return _str_to_col(col).degrees()


def radians(col: str | Expr) -> Expr:
    return _str_to_col(col).radians()


# ── string extras ─────────────────────────────────────────────────────────────

def initcap(col: str | Expr) -> Expr:
    return _str_to_col(col).initcap()


def ascii(col: str | Expr) -> Expr:
    return _str_to_col(col).ascii_code()


def instr(col: str | Expr, substr: str) -> Expr:
    return _str_to_col(col).instr(substr)


def split_part(col: str | Expr, delimiter: str, part_num: int) -> Expr:
    return _str_to_col(col).split_part(delimiter, part_num)


def substring_index(col: str | Expr, delimiter: str, count: int) -> Expr:
    return _str_to_col(col).substring_index(delimiter, count)


# ── date extras ───────────────────────────────────────────────────────────────

def quarter(col: str | Expr) -> Expr:
    return _str_to_col(col).quarter()


def minute(col: str | Expr) -> Expr:
    return _str_to_col(col).minute()


def second(col: str | Expr) -> Expr:
    return _str_to_col(col).second()


def weekofyear(col: str | Expr) -> Expr:
    return _str_to_col(col).weekofyear()


def weekday(col: str | Expr) -> Expr:
    return _str_to_col(col).weekday()


def date_trunc(unit: str, col: str | Expr) -> Expr:
    """Truncate date/timestamp to the given unit (Spark arg order: unit first)."""
    return _str_to_col(col).date_trunc(unit)


def make_date(year: Any, month: Any, day: Any) -> Expr:
    """Construct a Date from year, month, day (each may be int or Expr)."""
    y = _str_to_col(year) if isinstance(year, (str, Expr)) else polars_functions.lit(year)
    m = _str_to_col(month) if isinstance(month, (str, Expr)) else polars_functions.lit(month)
    d = _str_to_col(day) if isinstance(day, (str, Expr)) else polars_functions.lit(day)
    return polars_functions.date(y, m, d)


# ── aggregate extras ──────────────────────────────────────────────────────────

def stddev(col: str | Expr) -> Expr:
    return _str_to_col(col).std().alias(f"stddev({_col_name(col)})")


std = stddev
stddev_samp = stddev


def stddev_pop(col: str | Expr) -> Expr:
    return _str_to_col(col).std(ddof=0).alias(f"stddev_pop({_col_name(col)})")


def variance(col: str | Expr) -> Expr:
    return _str_to_col(col).var().alias(f"variance({_col_name(col)})")


var = variance
var_samp = variance


def var_pop(col: str | Expr) -> Expr:
    return _str_to_col(col).var(ddof=0).alias(f"var_pop({_col_name(col)})")


def median(col: str | Expr) -> Expr:
    return _str_to_col(col).median().alias(f"median({_col_name(col)})")


def count_distinct(*cols: str | Expr) -> Expr:
    if len(cols) == 1:
        return _str_to_col(cols[0]).n_unique()
    # multi-column distinct count via struct
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.struct(exprs).n_unique()


def count_if(condition: Expr) -> Expr:
    """Count rows where condition is true."""
    return polars_functions.when(condition).then(polars_functions.lit(1)).otherwise(polars_functions.lit(None)).count()


def bool_and(col: str | Expr) -> Expr:
    return _str_to_col(col).all()


every = bool_and


def bool_or(col: str | Expr) -> Expr:
    return _str_to_col(col).any()


some = bool_or


# ── null extras ───────────────────────────────────────────────────────────────

def nanvl(col: str | Expr, replacement: Any) -> Expr:
    return _str_to_col(col).nanvl(replacement)


def nvl2(col: str | Expr, not_null_val: Any, null_val: Any) -> Expr:
    return _str_to_col(col).nvl2(not_null_val, null_val)


# ── bitwise extras ────────────────────────────────────────────────────────────

def bitwise_not(col: str | Expr) -> Expr:
    return _str_to_col(col).bitwiseNOT()


def shiftleft(col: str | Expr, n: int) -> Expr:
    return _str_to_col(col).shiftLeft(n)


def shiftright(col: str | Expr, n: int) -> Expr:
    return _str_to_col(col).shiftRight(n)


# ── array extras ─────────────────────────────────────────────────────────────

def element_at(col: str | Expr, index: int) -> Expr:
    """Return element at 1-based index (negative counts from end)."""
    return _str_to_col(col).element_at(index)


def arrays_overlap(col1: str | Expr, col2: str | Expr) -> Expr:
    return _str_to_col(col1).arrays_overlap(_str_to_col(col2))


def array_repeat(element: Any, count: int) -> Expr:
    """Create an array by repeating element count times."""
    e = element if isinstance(element, Expr) else polars_functions.lit(element)
    if count <= 0:
        return polars_functions.lit(pl.Series(values=[[]], dtype=pl.List(pl.Null)))
    return polars_functions.concat_list([e] * count)


# ── window / ranking functions ────────────────────────────────────────────────

from .window import _WindowFuncExpr  # noqa: E402


def row_number() -> _WindowFuncExpr:
    """Row number (1-based) within the window partition, ordered by orderBy."""
    return _WindowFuncExpr("row_number")


def rank() -> _WindowFuncExpr:
    """Min rank within the window partition (ties share the lowest rank)."""
    return _WindowFuncExpr("rank")


def dense_rank() -> _WindowFuncExpr:
    """Dense rank within the window partition (no gaps after ties)."""
    return _WindowFuncExpr("dense_rank")


def lag(col: str | Expr, offset: int = 1, default: Any = None) -> _WindowFuncExpr:
    """Value of *col* offset rows before the current row within the partition."""
    return _WindowFuncExpr("lag", _str_to_col(col), offset, default)


def lead(col: str | Expr, offset: int = 1, default: Any = None) -> _WindowFuncExpr:
    """Value of *col* offset rows after the current row within the partition."""
    return _WindowFuncExpr("lead", _str_to_col(col), offset, default)


# ── math extras ───────────────────────────────────────────────────────────────

import math as _math  # noqa: E402


def pi() -> Expr:
    return polars_functions.lit(_math.pi)


def e() -> Expr:
    return polars_functions.lit(_math.e)


def rand(seed: int | None = None) -> Expr:
    """Random double in [0, 1) for each row."""
    import random as _random
    rng = _random.Random(seed)
    return polars_functions.int_range(pl.len(), dtype=polars_datatypes.UInt64()).map_elements(
        lambda _: rng.random(), polars_datatypes.Float64()
    )


def randn(seed: int | None = None) -> Expr:
    """Random standard-normal double for each row."""
    import random as _random
    rng = _random.Random(seed)
    return polars_functions.int_range(pl.len(), dtype=polars_datatypes.UInt64()).map_elements(
        lambda _: rng.gauss(0.0, 1.0), polars_datatypes.Float64()
    )


def factorial(col: str | Expr) -> Expr:
    return _str_to_col(col).map_elements(
        lambda x: _math.factorial(int(x)) if x is not None else None,
        polars_datatypes.Int64(),
    )


def hex(col: str | Expr) -> Expr:
    """Convert integer to uppercase hexadecimal string."""
    return _str_to_col(col).map_elements(
        lambda x: format(int(x), "X") if x is not None else None,
        polars_datatypes.String(),
    )


def unhex(col: str | Expr) -> Expr:
    """Convert hexadecimal string to integer."""
    return _str_to_col(col).map_elements(
        lambda x: int(x, 16) if x is not None else None,
        polars_datatypes.Int64(),
    )


# ── misc extras ───────────────────────────────────────────────────────────────

lit = polars_functions.lit
typedLit = polars_functions.lit
typed_lit = polars_functions.lit


# ── string extras (batch 2) ───────────────────────────────────────────────────

def chr(col: str | Expr) -> Expr:
    """Convert Unicode code point to character."""
    return _str_to_col(col).chr()


def find_in_set(str_col: str | Expr, str_array_col: str | Expr) -> Expr:
    """Return 1-based position of str_col in comma-delimited str_array_col; 0 if not found."""
    sv = _str_to_col(str_col) if isinstance(str_col, Expr) else str_col
    return _str_to_col(str_array_col).find_in_set(sv)


def regexp_like(col: str | Expr, pattern: str) -> Expr:
    return _str_to_col(col).regexp_like(pattern)


printf = format_string  # alias


def overlay(col: str | Expr, replace: Any, pos: int, length: int | None = None) -> Expr:
    return _str_to_col(col).overlay(replace, pos, length)


# ── math extras (batch 2) ─────────────────────────────────────────────────────

def bround(col: str | Expr, d: int = 0) -> Expr:
    """Banker's rounding (half-to-even)."""
    return _str_to_col(col).bround(d)


def hypot(x: str | Expr, y: str | Expr) -> Expr:
    """Euclidean distance: sqrt(x² + y²)."""
    x_e = _str_to_col(x).cast(polars_datatypes.Float64(), strict=False)
    y_e = _str_to_col(y).cast(polars_datatypes.Float64(), strict=False)
    return (x_e ** 2 + y_e ** 2).sqrt()


def pmod(col: str | Expr, divisor: Any) -> Expr:
    """Positive modulo — result has the same sign as divisor."""
    return _str_to_col(col).pmod(divisor)


def shiftrightunsigned(col: str | Expr, n: int) -> Expr:
    """Unsigned (logical) right shift — fills with zeros on the left."""
    return _str_to_col(col).map_elements(
        lambda x: (x & 0xFFFF_FFFF_FFFF_FFFF) >> n if x is not None else None,
        polars_datatypes.Int64(),
    )


# ── date extras (batch 2) ─────────────────────────────────────────────────────

def trunc(col: str | Expr, fmt: str) -> Expr:
    """Truncate date/timestamp to fmt unit (col first — mirrors PySpark trunc())."""
    return _str_to_col(col).date_trunc(fmt)


def next_day(col: str | Expr, day_of_week: str) -> Expr:
    """Return the first date strictly after col that is the named day of week."""
    return _str_to_col(col).next_day(day_of_week)


def make_timestamp(
    year: Any,
    month: Any,
    day: Any,
    hour: Any,
    minute: Any,
    second: Any,
    timezone: str | None = None,  # noqa: ARG001  (timezone not supported in Polars)
) -> Expr:
    """Construct a Datetime from year/month/day/hour/minute/second components."""
    import datetime as _dt

    def _to_expr(v: Any, alias: str) -> Expr:
        return (_str_to_col(v) if isinstance(v, (str, Expr)) else polars_functions.lit(v)).alias(alias)

    s = polars_functions.struct([
        _to_expr(year, "_y"), _to_expr(month, "_mo"), _to_expr(day, "_d"),
        _to_expr(hour, "_h"), _to_expr(minute, "_min"), _to_expr(second, "_s"),
    ])

    def _make(row: Any) -> Any:
        try:
            return _dt.datetime(
                int(row["_y"]), int(row["_mo"]), int(row["_d"]),
                int(row["_h"]), int(row["_min"]), int(row["_s"]),
            )
        except (ValueError, TypeError):
            return None

    return s.map_elements(_make, polars_datatypes.Datetime())


def unix_date(col: str | Expr) -> Expr:
    """Number of days since Unix epoch (1970-01-01)."""
    return _str_to_col(col).unix_date()


def from_unixtime(col: str | Expr, fmt: str | None = None) -> Expr:
    """Convert integer epoch-seconds to Datetime; optionally format as string."""
    dt_expr = polars_functions.from_epoch(_str_to_col(col), time_unit="s")
    if fmt is not None:
        return dt_expr.dt.strftime(fmt)
    return dt_expr


# ── array extras (batch 2) ────────────────────────────────────────────────────

def arrays_zip(*cols: str | Expr) -> Expr:
    """Zip arrays into an array of structs (one struct per position across all arrays)."""
    exprs = [_str_to_col(c) for c in cols]
    names: list[str] = []
    for i, e in enumerate(exprs):
        try:
            names.append(e.meta.output_name())
        except Exception:
            names.append(f"_{i}")

    def _batch(series_list: list) -> pl.Series:
        rows: list = []
        for row_idx in range(len(series_list[0])):
            row_lists = [series_list[j][row_idx].to_list() for j in range(len(series_list))]
            min_len = row_lists[0].__len__()
            for rl in row_lists[1:]:
                if len(rl) < min_len:
                    min_len = len(rl)
            rows.append([
                {names[j]: row_lists[j][k] for j in range(len(names))}
                for k in range(min_len)
            ])
        return pl.Series(rows)

    return polars_functions.map_batches(exprs, _batch, return_dtype=None)


def shuffle(col: str | Expr, seed: int | None = None) -> Expr:
    """Return the array with elements in random order."""
    return _str_to_col(col).shuffle(seed)


def exists(col: str | Expr, f: Any) -> Expr:
    """Return true if any element satisfies predicate f."""
    return _str_to_col(col).exists(f)


def aggregate(col: str | Expr, zero: Any, merge: Any, finish: Any = None) -> Expr:
    """Fold over list: reduce(merge, elements, zero), then optionally apply finish."""
    return _str_to_col(col).aggregate(zero, merge, finish)


# ── aggregate extras (batch 2) ────────────────────────────────────────────────

def corr(col1: str | Expr, col2: str | Expr, method: str = "pearson") -> Expr:
    """Correlation between two columns (pearson or spearman)."""
    return pl.corr(_str_to_col(col1), _str_to_col(col2), method=method)


def covar_samp(col1: str | Expr, col2: str | Expr) -> Expr:
    """Sample covariance between two columns."""
    return pl.cov(_str_to_col(col1), _str_to_col(col2))


def covar_pop(col1: str | Expr, col2: str | Expr) -> Expr:
    """Population covariance between two columns (ddof=0)."""
    c1, c2 = _str_to_col(col1), _str_to_col(col2)

    def _pop_cov(series_list: list) -> pl.Series:
        s1 = series_list[0].cast(pl.Float64)
        s2 = series_list[1].cast(pl.Float64)
        if len(s1) == 0:
            return pl.Series([None], dtype=pl.Float64)
        cov = ((s1 - s1.mean()) * (s2 - s2.mean())).mean()
        return pl.Series([cov], dtype=pl.Float64)

    return polars_functions.map_batches([c1, c2], _pop_cov, return_dtype=polars_datatypes.Float64())


def kurtosis(col: str | Expr) -> Expr:
    return _str_to_col(col).kurtosis(fisher=True, bias=True).alias("kurtosis")


def skewness(col: str | Expr) -> Expr:
    return _str_to_col(col).skew(bias=True).alias("skewness")


def mode(col: str | Expr) -> Expr:
    """Most frequent value; smallest if there is a tie."""
    c = _str_to_col(col)
    return polars_functions.map_batches(
        [c],
        lambda s: pl.Series([s[0].mode().sort()[0]]),
        return_dtype=None,
    )


def percentile(col: str | Expr, pct: float, accuracy: int = 10_000) -> Expr:  # noqa: ARG001
    """Exact percentile (pct on 0.0–1.0 scale)."""
    return _str_to_col(col).quantile(pct, interpolation="nearest").alias("percentile")


def sum_distinct(col: str | Expr) -> Expr:
    """Sum of distinct values."""
    return _str_to_col(col).sum_distinct().alias("sum_distinct")


any_value = first  # arbitrary non-null value


def approx_count_distinct(col: str | Expr, rsd: float = 0.05) -> Expr:  # noqa: ARG001
    """Approximate distinct count (exact in Polars)."""
    return _str_to_col(col).n_unique()


# ── null / type-check standalones ─────────────────────────────────────────────

def isnull(col: str | Expr) -> Expr:
    """Return boolean expression: true where col is null."""
    return _str_to_col(col).is_null()


def isnan(col: str | Expr) -> Expr:
    """Return boolean expression: true where col is NaN."""
    return _str_to_col(col).is_nan()


def isnotnull(col: str | Expr) -> Expr:
    """Return boolean expression: true where col is not null."""
    return _str_to_col(col).is_not_null()


# ── hash extras ───────────────────────────────────────────────────────────────

def sha2(col: str | Expr, num_bits: int = 256) -> Expr:
    """SHA-2 family hash (num_bits: 224, 256, 384, 512; 0 → 256)."""
    import hashlib as _hashlib

    _SHA2_ALGOS = {0: "sha256", 224: "sha224", 256: "sha256", 384: "sha384", 512: "sha512"}
    algo_name = _SHA2_ALGOS.get(num_bits, "sha256")
    h = getattr(_hashlib, algo_name)
    return _str_to_col(col).map_elements(
        lambda x: h(str(x).encode()).hexdigest() if x is not None else None,
        polars_datatypes.String(),
    )


def crc32(col: str | Expr) -> Expr:
    """CRC32 checksum as unsigned 32-bit integer."""
    import binascii as _binascii

    return _str_to_col(col).map_elements(
        lambda x: _binascii.crc32(str(x).encode()) & 0xFFFF_FFFF if x is not None else None,
        polars_datatypes.UInt32(),
    )


def hash(*cols: str | Expr) -> Expr:
    """MurmurHash3-style hash of one or more columns (uses Polars native hash)."""
    if len(cols) == 1:
        return _str_to_col(cols[0]).hash(seed=0)
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.struct(exprs).hash(seed=0)


def xxhash64(*cols: str | Expr) -> Expr:
    """xxHash64-style hash of one or more columns (uses Polars native hash with seed 42)."""
    if len(cols) == 1:
        return _str_to_col(cols[0]).hash(seed=42)
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.struct(exprs).hash(seed=42)


# ── URL helpers ───────────────────────────────────────────────────────────────

def url_encode(col: str | Expr) -> Expr:
    """Percent-encode a string for use in a URL."""
    from urllib.parse import quote as _quote

    return _str_to_col(col).map_elements(
        lambda x: _quote(str(x), safe="") if x is not None else None,
        polars_datatypes.String(),
    )


def url_decode(col: str | Expr) -> Expr:
    """Decode a percent-encoded URL string."""
    from urllib.parse import unquote as _unquote

    return _str_to_col(col).map_elements(
        lambda x: _unquote(str(x)) if x is not None else None,
        polars_datatypes.String(),
    )


# ── JSON helper ───────────────────────────────────────────────────────────────

def get_json_object(col: str | Expr, path: str) -> Expr:
    """Extract a value from a JSON string using a JSONPath expression."""
    return _str_to_col(col).str.json_path_match(path)


# ── string extras (batch 3) ───────────────────────────────────────────────────

def unbase64(col: str | Expr) -> Expr:
    """Decode a Base64-encoded string column to UTF-8."""
    return _str_to_col(col).unbase64()


def levenshtein(col1: str | Expr, col2: str | Expr) -> Expr:
    """Compute Levenshtein edit distance between two string columns."""
    return _str_to_col(col1).levenshtein(_str_to_col(col2))


def regexp_substr(col: str | Expr, pattern: str) -> Expr:
    """Return first substring matching the regexp."""
    return _str_to_col(col).regexp_substr(pattern)


def elt(*cols: str | Expr) -> Expr:
    """Return the n-th input (1-based index): elt(index, str1, str2, ...)."""
    if len(cols) < 2:
        raise ValueError("elt() requires at least 2 arguments: index + strings")
    index_expr = _str_to_col(cols[0])
    str_exprs = [_str_to_col(c) for c in cols[1:]]

    def _batch(series_list: list) -> pl.Series:
        idx_series = series_list[0]
        str_series = series_list[1:]
        results = []
        for row_i in range(len(idx_series)):
            idx = idx_series[row_i]
            if idx is None or idx < 1 or idx > len(str_series):
                results.append(None)
            else:
                results.append(str_series[idx - 1][row_i])
        return pl.Series(results, dtype=polars_datatypes.String())

    return polars_functions.map_batches([index_expr] + str_exprs, _batch, return_dtype=polars_datatypes.String())


# ── math extras (batch 3) ─────────────────────────────────────────────────────

import math as _math  # noqa: E402


def log1p(col: str | Expr) -> Expr:
    """Natural log of (1 + x)."""
    return _str_to_col(col).log1p()


def expm1(col: str | Expr) -> Expr:
    """e^x - 1."""
    return _str_to_col(col).expm1()


def rint(col: str | Expr) -> Expr:
    """Round to nearest integer (returns float)."""
    return _str_to_col(col).rint()


def remainder(col1: str | Expr, col2: str | Expr) -> Expr:
    """IEEE 754 remainder: x - round(x/y)*y."""
    c1 = _str_to_col(col1)
    c2 = _str_to_col(col2)
    s = polars_functions.struct([c1.alias("_x"), c2.alias("_y")])
    return s.map_elements(
        lambda r: _math.remainder(r["_x"], r["_y"]) if r["_x"] is not None and r["_y"] is not None else None,
        polars_datatypes.Float64(),
    )


def gcd(col1: str | Expr, col2: str | Expr) -> Expr:
    """Greatest common divisor of two integer columns."""
    import math as _m

    c1 = _str_to_col(col1)
    c2 = _str_to_col(col2)
    s = polars_functions.struct([c1.alias("_a"), c2.alias("_b")])
    return s.map_elements(
        lambda r: _m.gcd(int(r["_a"]), int(r["_b"])) if r["_a"] is not None and r["_b"] is not None else None,
        polars_datatypes.Int64(),
    )


def lcm(col1: str | Expr, col2: str | Expr) -> Expr:
    """Least common multiple of two integer columns."""
    import math as _m

    c1 = _str_to_col(col1)
    c2 = _str_to_col(col2)
    s = polars_functions.struct([c1.alias("_a"), c2.alias("_b")])
    return s.map_elements(
        lambda r: _m.lcm(int(r["_a"]), int(r["_b"])) if r["_a"] is not None and r["_b"] is not None else None,
        polars_datatypes.Int64(),
    )


def bitcount(col: str | Expr) -> Expr:
    """Count the number of set bits in the integer value."""
    return _str_to_col(col).bitcount()


def toDegrees(col: str | Expr) -> Expr:
    """Convert radians to degrees."""
    return _str_to_col(col) * (180.0 / _math.pi)


def toRadians(col: str | Expr) -> Expr:
    """Convert degrees to radians."""
    return _str_to_col(col) * (_math.pi / 180.0)


# ── date extras (batch 3) ─────────────────────────────────────────────────────

def months_between(end: str | Expr, start: str | Expr, roundOff: bool = True) -> Expr:  # noqa: N803
    """Number of months between two date/timestamp columns."""
    e = _str_to_col(end)
    s = _str_to_col(start)
    struct_expr = polars_functions.struct([e.alias("_e"), s.alias("_s")])

    def _mb(row: Any) -> Any:
        import datetime as _dt

        ev, sv = row["_e"], row["_s"]
        if ev is None or sv is None:
            return None
        # Convert to date-only for month arithmetic
        if hasattr(ev, "date"):
            ed, sd = ev.date(), sv.date()
        else:
            ed, sd = ev, sv
        months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
        day_diff = (ed.day - sd.day) / 31.0
        result = months + day_diff
        if roundOff:
            import builtins as _builtins
            return _builtins.round(result, 8)
        return result

    return struct_expr.map_elements(_mb, polars_datatypes.Float64())


def to_utc_timestamp(col: str | Expr, tz: str) -> Expr:
    """Interpret timestamp as being in *tz* and convert to UTC."""
    return _str_to_col(col).to_utc_timestamp(tz)


def from_utc_timestamp(col: str | Expr, tz: str) -> Expr:
    """Interpret UTC timestamp and convert to *tz*."""
    return _str_to_col(col).from_utc_timestamp(tz)


# ── array extras (batch 3) ────────────────────────────────────────────────────

def array_reverse(col: str | Expr) -> Expr:
    """Reverse the elements in an array column."""
    return _str_to_col(col).array_reverse()


def array_insert(col: str | Expr, pos: int, value: Any) -> Expr:
    """Insert *value* at position *pos* (1-based, supports negative) in each array.

    For negative *pos*, insertion is measured from the end (PySpark semantics).
    Positive *pos* is 1-based; implemented with pure Polars expressions.
    Negative *pos* falls back to a UDF (eager DataFrames only).
    """
    c = _str_to_col(col)
    if pos > 0:
        idx = pos - 1  # 0-based
        val_lit = polars_functions.lit(pl.Series([value])).implode()
        part1 = c.list.slice(0, idx)
        part2 = c.list.slice(idx)
        return polars_functions.concat_list([part1, val_lit, part2])
    # negative pos: need runtime list length — wrap with .list.len()
    # insert at position (list.len + pos + 1) from start (0-based: len + pos)
    len_expr = c.list.len()
    idx_expr = len_expr + pos  # 0-based insert index (pos is negative)
    # Clamp to [0, len]
    idx_expr = pl.when(idx_expr < 0).then(0).otherwise(idx_expr)
    val_lit = polars_functions.lit(pl.Series([value])).implode()
    # Can't do variable-length slice with expression idx in Polars, fall back to UDF
    s_expr = polars_functions.struct([c.alias("_lst"), idx_expr.cast(polars_datatypes.Int64()).alias("_idx")])

    def _insert_neg(row: Any) -> Any:
        lst = row["_lst"]
        if lst is None:
            return None
        items = lst.to_list() if hasattr(lst, "to_list") else list(lst)
        items.insert(int(row["_idx"]), value)
        return items

    return s_expr.map_elements(_insert_neg, return_dtype=None)


# ── aggregate/window extras (batch 3) ─────────────────────────────────────────

def mean(col: str | Expr) -> Expr:
    """Alias for avg()."""
    return _str_to_col(col).mean().alias(f"avg({_col_name(col)})")


def ntile(n: int) -> _WindowFuncExpr:
    """Divide the ordered partition into *n* buckets (1-based)."""
    return _WindowFuncExpr("ntile", n)


def cume_dist() -> _WindowFuncExpr:
    """Cumulative distribution: position / total rows within the partition."""
    return _WindowFuncExpr("cume_dist")


def percent_rank() -> _WindowFuncExpr:
    """Percent rank: (rank - 1) / (total - 1), 0.0 when only one row."""
    return _WindowFuncExpr("percent_rank")


# ── struct / map extras (batch 3) ─────────────────────────────────────────────

def to_json(col: str | Expr) -> Expr:
    """Serialize a struct column to a JSON string."""
    return _str_to_col(col).to_json()


def map_keys(col: str | Expr) -> Expr:
    """Return keys of a map column (list-of-structs with 'key' field)."""
    return _str_to_col(col).map_keys()


def map_values(col: str | Expr) -> Expr:
    """Return values of a map column (list-of-structs with 'value' field)."""
    return _str_to_col(col).map_values()


# ── fix unix_timestamp to support format string ───────────────────────────────

def unix_timestamp(col: str | Expr | None = None, fmt: str | None = None) -> Expr:
    """Seconds since epoch.  With no args returns current time."""
    if col is None:
        import time as _time

        return pl.lit(int(_time.time())).cast(polars_datatypes.Int64())
    e = _str_to_col(col)
    if fmt is not None:
        return e.str.strptime(polars_datatypes.Datetime("us"), fmt).dt.epoch(time_unit="s")
    return e.dt.epoch(time_unit="s")


# ── monotonic_id alias ────────────────────────────────────────────────────────

def monotonic_id() -> Expr:
    """Monotonically increasing 64-bit integer (row index)."""
    return polars_functions.int_range(
        start=0,
        end=polars_functions.len(),
        dtype=polars_datatypes.UInt64(),
    )


# ── NotImplementedError stubs ─────────────────────────────────────────────────

def soundex(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("soundex is not supported in Polars")


def metaphone(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("metaphone is not supported in Polars")


def json_tuple(col: str | Expr, *fields: str) -> Expr:
    """Extract multiple JSON fields into a struct column."""
    c = _str_to_col(col)
    field_exprs = [c.str.json_path_match(f"$.{f}").alias(f) for f in fields]
    return polars_functions.struct(field_exprs)


def from_json(col: str | Expr, schema: Any) -> Expr:  # noqa: ARG001
    raise NotImplementedError("from_json is not supported in Polars")


def map_from_entries(col: str | Expr) -> Expr:
    """Identity — Polars already stores maps as list(struct({key, value}))."""
    return _str_to_col(col)


def map_concat(*cols: str | Expr) -> Expr:
    """Concatenate map columns (list-of-structs) into one."""
    exprs = [_str_to_col(c) for c in cols]
    return polars_functions.concat_list(*exprs)


def map_filter(col: str | Expr, f: Callable) -> Expr:
    """Keep only entries where f(key, value) is True."""
    c = _str_to_col(col)
    return c.list.eval(
        pl.element().filter(f(pl.element().struct.field("key"), pl.element().struct.field("value")))
    )


def transform_keys(col: str | Expr, f: Callable) -> Expr:
    """Apply f(key, value) to every key in a map column."""
    c = _str_to_col(col)
    return c.list.eval(
        polars_functions.struct(
            f(pl.element().struct.field("key"), pl.element().struct.field("value")).alias("key"),
            pl.element().struct.field("value").alias("value"),
        )
    )


def transform_values(col: str | Expr, f: Callable) -> Expr:
    """Apply f(key, value) to every value in a map column."""
    c = _str_to_col(col)
    return c.list.eval(
        polars_functions.struct(
            pl.element().struct.field("key").alias("key"),
            f(pl.element().struct.field("key"), pl.element().struct.field("value")).alias("value"),
        )
    )


def grouping(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("grouping is not supported outside GROUPING SETS context")


def grouping_id(*cols: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("grouping_id is not supported outside GROUPING SETS context")


def input_file_name() -> Expr:
    raise NotImplementedError("input_file_name is not applicable to Polars DataFrames")


def assert_true(col: str | Expr, error_msg: str = "") -> Expr:
    """Raise a RuntimeError if any value in *col* is False/null; else return null column."""
    c = _str_to_col(col)

    def _check(s: pl.Series) -> pl.Series:
        if not s.cast(pl.Boolean).fill_null(False).all():
            msg = error_msg or f"assert_true failed: {s.name}"
            raise RuntimeError(msg)
        return pl.Series([None] * len(s), dtype=pl.Null)

    return c.map_batches(_check, return_dtype=pl.Null)


def schema_of_csv(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("schema_of_csv is not supported in Polars")


def schema_of_json(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("schema_of_json is not supported in Polars")


def window(time_col: str | Expr, window_duration: str, slide_duration: str | None = None, start_time: str | None = None) -> Expr:  # noqa: ARG001
    raise NotImplementedError("window (time-based tumbling/sliding) is not supported in Polars")


def posexplode(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("posexplode is not supported; use explode() with a row index instead")


def posexplode_outer(col: str | Expr) -> Expr:  # noqa: ARG001
    raise NotImplementedError("posexplode_outer is not supported; use explode() with a row index instead")


# ── batch-4: aliases & new functions ─────────────────────────────────────────

def ln(col: str | Expr) -> Expr:
    """Natural logarithm (alias for log with base e)."""
    return _str_to_col(col).log(2.718281828459045)


sha = sha1  # SHA-1 alias


def cardinality(col: str | Expr) -> Expr:
    """Number of elements in an array/map column (alias for size)."""
    return _str_to_col(col).list.len()


def approx_count_distinct(col: str | Expr) -> Expr:
    """Approximate distinct count (HyperLogLog approximation via n_unique)."""
    c = _str_to_col(col)
    name = _col_name(col)
    return c.approx_n_unique().alias(f"approx_count_distinct({name})")


approxCountDistinct = approx_count_distinct


def sum_distinct(col: str | Expr) -> Expr:
    """Sum of distinct values."""
    c = _str_to_col(col)
    name = _col_name(col)
    return c.drop_nulls().unique().sum().alias(f"sum_distinct({name})")


sumDistinct = sum_distinct


def regexp(col: str | Expr, pattern: str) -> Expr:
    """Alias for regexp_like."""
    return regexp_like(col, pattern)


def format_number(col: str | Expr, d: int) -> Expr:
    """Format a number to *d* decimal places as a string."""
    import builtins as _b
    fmt = f"{{:.{_b.max(0, d)}f}}"
    return _str_to_col(col).map_elements(lambda v: fmt.format(v), return_dtype=pl.String)


def regexp_instr(col: str | Expr, pattern: str) -> Expr:
    """Return 1-based start position of first regex match, 0 if no match."""
    import re as _re
    def _find(s: str) -> int:
        if s is None:
            return None
        m = _re.search(pattern, s)
        return (m.start() + 1) if m else 0
    return _str_to_col(col).map_elements(_find, return_dtype=pl.Int64)


def typeof(col: str | Expr) -> Expr:
    """Return the Polars dtype of the column as a string literal."""
    c = _str_to_col(col)

    def _dtype_name(s: pl.Series) -> pl.Series:
        return pl.Series([str(s.dtype)] * len(s))

    return c.map_batches(_dtype_name, return_dtype=pl.String)


def uniform(min_val: float, max_val: float, seed: int | None = None) -> Expr:
    """Uniformly distributed random float in [min_val, max_val] per row."""
    import random as _random
    rng = _random.Random(seed)

    def _uniform(s: pl.Series) -> pl.Series:
        return pl.Series([rng.uniform(min_val, max_val) for _ in range(len(s))], dtype=pl.Float64)

    return polars_functions.int_range(start=0, end=polars_functions.len(), dtype=pl.Int64).map_batches(
        _uniform, return_dtype=pl.Float64
    )


def percentile_approx(col: str | Expr, percentage: float, accuracy: int = 10000) -> Expr:  # noqa: ARG002
    """Approximate percentile (exact via Polars quantile)."""
    c = _str_to_col(col)
    name = _col_name(col)
    return c.quantile(percentage, interpolation="nearest").alias(f"percentile_approx({name}, {percentage})")


def zip_with(col1: str | Expr, col2: str | Expr, f: Callable) -> Expr:
    """Element-wise merge of two list columns using function f(x, y)."""
    c1 = _str_to_col(col1)
    c2 = _str_to_col(col2)
    return polars_functions.concat_list(c1, c2).list.eval(
        f(pl.element().list.first(), pl.element().list.last())
    )


# ── map/struct extras (batch-4) ───────────────────────────────────────────────

def map_contains_key(col: str | Expr, key: Any) -> Expr:
    """Return True if the map column contains *key*."""
    return _str_to_col(col).list.eval(
        pl.element().struct.field("key").eq(pl.lit(key))
    ).list.any()


def map_entries(col: str | Expr) -> Expr:
    """Return list of structs {key, value} — identity for Polars map columns."""
    return _str_to_col(col)


def map_from_arrays(keys: str | Expr, values: str | Expr) -> Expr:
    """Build a map column (list-of-structs) from parallel key and value list columns."""
    k = _str_to_col(keys)
    v = _str_to_col(values)

    def _zip_row(row: dict) -> list:
        return [{"key": ki, "value": vi} for ki, vi in zip(row["keys"] or [], row["values"] or [])]

    return polars_functions.struct(k.alias("keys"), v.alias("values")).map_elements(
        _zip_row, return_dtype=pl.List(pl.Struct({"key": pl.String, "value": pl.Int64}))
    )


def named_struct(*args: Any) -> Expr:
    """Build a struct from alternating name/value pairs: named_struct("a", col1, "b", col2)."""
    if len(args) % 2 != 0:
        raise ValueError("named_struct requires an even number of arguments (name, value pairs)")
    fields = []
    for i in range(0, len(args), 2):
        name = args[i]
        value = args[i + 1]
        if not isinstance(name, str):
            raise TypeError(f"named_struct field names must be strings, got {type(name)}")
        e = _str_to_col(value) if not isinstance(value, Expr) else value
        fields.append(e.alias(name))
    return polars_functions.struct(fields)
