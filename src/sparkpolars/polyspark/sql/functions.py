"""Polars SQL functions for Polyspark."""

import datetime
from functools import reduce
from typing import Any

import polars as pl
import polars.datatypes as plf_types
import polars.functions as plf
from polars import lit
from polars._utils.parse import parse_into_list_of_expressions
from polars.datatypes import DataType
from polars.expr import Expr

Expr._original_cast = Expr.cast


def cast_strict(self: Expr, dtype: DataType, strict: bool = False) -> Expr:
    # Filter out columns that don't exist
    return self._original_cast(dtype, strict=strict)


Expr.cast = cast_strict


def broadcast(df: Any) -> Any:
    return df


# Add Spark-compatible methods to Polars Expr
def isNull(self: Expr) -> Expr:
    return self.is_null()


def isNotNull(self: Expr) -> Expr:
    return self.is_not_null()


def isin(self: Expr, *cols: Any) -> Expr:
    if len(cols) == 1:
        return self.is_in(cols[0])
    return self.is_in(cols)


def over(
    self: Expr,
    partition_by: Any = None,
    *more_exprs: Any,
    order_by: Any = None,
    sort_by: Any = None,
) -> Expr:
    # Handle Window object
    if hasattr(partition_by, "_partition_by"):
        # Extract values from Window object
        window_partition_by = partition_by._partition_by
        window_order_by = partition_by._order_by
        window_sort_by = partition_by._sort_by

        # Use Window object values, overriding with any explicit parameters
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
            order_by_nulls_last=False,  # does not work yet
            mapping_strategy="group_to_rows",
        ),
    )

    # Only apply sort_by if both order_by and sort_by are not None
    if order_by is not None and sort_by is not None:
        result = result.sort_by(order_by, descending=sort_by)

    return result


# Monkey-patch the Expr class to add Spark methods
Expr.isNull = isNull
Expr.isNotNull = isNotNull
Expr.isin = isin
Expr.over = over

col = plf.col
column = col
Column = col


def _str_to_col(name: str | Expr) -> Expr:
    if isinstance(name, str):
        return col(name)
    return name


def eqNullSafe(self: Expr, other: str | Expr) -> Expr:
    other = _str_to_col(other)
    return self.eq_missing(other)


Expr.eqNullSafe = eqNullSafe


def rlike(str: str | Expr, regexp: str) -> Expr:
    return _str_to_col(str).str.contains(regexp, literal=False)


Expr.rlike = rlike
regexp_like = rlike
regexp = rlike


def between(self: Expr, lowerBound: Any, upperBound: Any) -> Expr:
    return self.ge(lowerBound) & self.le(upperBound)


Expr.between = between


def startswith(str: str | Expr, prefix: str) -> Expr:
    return _str_to_col(str).str.starts_with(prefix)


Expr.startswith = startswith


def endswith(str: str | Expr, suffix: str) -> Expr:
    return _str_to_col(str).str.ends_with(suffix)


Expr.endswith = endswith


def substring(str: str | Expr, pos: int, len: int | None = None) -> Expr:
    if len is not None:
        return _str_to_col(str).str.slice(pos - 1, len)
    return _str_to_col(str).str.slice(pos - 1)


substr = substring
Expr.substr = substr


def trim(col: str | Expr, trim: str) -> Expr:
    return _str_to_col(col).str.strip_chars(trim)


def when(condition: Expr, value: Any) -> Expr:
    return plf.when(condition).then(value)


def concat_ws(separator: str, *cols: Any) -> Expr:
    exprs = []
    for c in cols:
        if hasattr(c, "_expr"):
            exprs.append(c)
        else:
            exprs.append(c)
    if len(exprs) == 1:
        return exprs[0].list.join(separator)
    return plf.concat_str(*exprs, separator=separator)


def expr(str: str) -> Expr:
    return plf.sql_expr(str)


def upper(str: str | Expr) -> Expr:
    return _str_to_col(str).str.to_uppercase()


ucase = upper


def lower(str: str | Expr) -> Expr:
    return _str_to_col(str).str.to_lowercase()


lcase = lower


def regexp_count(str: str | Expr, regexp: str) -> Expr:
    return _str_to_col(str).str.count_matches(regexp)


def regexp_extract(str: str | Expr, pattern: str, idx: int) -> Expr:
    return _str_to_col(str).str.extract(pattern, idx)


def regexp_extract_all(str: str | Expr, regexp: str, idx: int | None = None) -> Expr:
    if idx is None:
        return _str_to_col(str).str.extract_all(regexp)
    msg = "idx parameter is not supported in Polars for regexp_extract_all"
    raise NotImplementedError(msg)


def regexp_replace(str: str | Expr, pattern: str, replacement: str) -> Expr:
    return _str_to_col(str).str.replace(pattern, replacement)


def replace(src: str | Expr, search: str, replace: str | None = None) -> Expr:
    if replace is None:
        return _str_to_col(src).str.replace_all(search, "")
    return _str_to_col(src).str.replace(search, replace)


def rtrim(src: str | Expr, trim: str | None = None) -> Expr:
    if trim is None:
        return _str_to_col(src).str.strip_chars_end()
    return _str_to_col(src).str.strip_chars_end(trim)


def ltrim(src: str | Expr, trim: str | None = None) -> Expr:
    if trim is None:
        return _str_to_col(src).str.strip_chars_start()
    return _str_to_col(src).str.strip_chars_start(trim)


def rpad(src: str | Expr, length: int, pad: str | None = None) -> Expr:
    if pad is None:
        return _str_to_col(src).str.pad_end(length)
    return _str_to_col(src).str.pad(length, pad)


def lpad(src: str | Expr, length: int, pad: str | None = None) -> Expr:
    if pad is None:
        return _str_to_col(src).str.pad_start(length)
    return _str_to_col(src).str.pad(length, pad)


def base64(col: str | Expr) -> Expr:
    return _str_to_col(col).str.encode("base64")


def btrim(str: str | Expr, trim: str | None = None) -> Expr:
    if trim is None:
        return _str_to_col(str).str.strip_chars()
    return _str_to_col(str).str.strip_chars(trim)


def contains(left: str | Expr, right: str | Expr) -> Expr:
    return _str_to_col(left).str.contains(right)


Expr.contains = contains


def encode(col: str | Expr, charset: str) -> Expr:
    return _str_to_col(col).str.encode(charset)


def decode(col: str | Expr, charset: str) -> Expr:
    return _str_to_col(col).str.decode(charset)


def left(str: str | Expr, len: int) -> Expr:
    return _str_to_col(str).str.slice(0, len)


def right(str: str | Expr, len: int) -> Expr:
    return _str_to_col(str).str.slice(-len, len) if len > 0 else _str_to_col(str).str.slice(-len)


def length(col: str | Expr) -> Expr:
    return _str_to_col(col).str.len_chars()


def locate(substr: str, str: str | Expr, _pos: int | None = None) -> Expr:
    return _str_to_col(str).str.find(substr)


position = locate


def repeat(col: str | Expr, n: int) -> Expr:
    return reduce(
        lambda acc, _: acc + col.cast(plf_types.String(), strict=False),
        range(n - 1),
        col.cast(plf_types.String(), strict=False),
    )


def concat(*cols: str | Expr) -> Expr:
    return reduce(lambda acc, col: acc + col.cast(plf_types.String(), strict=False), cols)


def round(col: str | Expr, scale: int = 0) -> Expr:
    return col.round(scale)


def sqrt(col: str | Expr) -> Expr:
    return col.sqrt()


def pow(col: str | Expr, exponent: int | float) -> Expr:
    return col.pow(exponent)


power = pow


def negate(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return lit(-1) * col


negative = negate


def translate(srcCol: str | Expr, matching: str, replace: str) -> Expr:
    srcCol = _str_to_col(srcCol)
    if len(matching) != len(replace):
        msg = "Matching and replace strings must have the same length."
        raise ValueError(msg)
    trans_table = str.maketrans(matching, replace)
    return srcCol.map_elements(lambda x: x.translate(trans_table), plf_types.String())


def collect_list(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.implode()


array_agg = collect_list
listagg = collect_list


def collect_set(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.unique().implode()


array_agg_distinct = collect_set
listagg_distinct = collect_set


def sum(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.sum().alias("sum")


def min(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.min().alias("min")


def max(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.max().alias("max")


def abs(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.abs().alias("abs")


def avg(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.mean().alias("avg")


def first(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.first().alias("first")


first_value = first


def last(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.last().alias("last")


last_value = last


def array_distinct(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.list.unique()


def greatest(*cols: str | Expr) -> Expr:
    cols = [_str_to_col(col) for col in cols]
    return plf.max_horizontal(*cols)


def least(*cols: str | Expr) -> Expr:
    cols = [_str_to_col(col) for col in cols]
    return plf.min_horizontal(*cols)


def _get_desc_status(self: Expr) -> bool:
    return getattr(self, "_desc_status_value", False)


def _set_desc_status(self: Expr, status: bool) -> None:
    self._desc_status_value = status


Expr._desc_status = property(_get_desc_status, _set_desc_status)


def asc(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    col._desc_status = False
    return col.sort(descending=False, nulls_last=False)


def asc_nulls_first(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    col._desc_status = False
    return col.sort(descending=False, nulls_last=False)


def asc_nulls_last(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    col._desc_status = False
    return col.sort(descending=False, nulls_last=True)


def desc(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    col._desc_status = True
    return col.sort(descending=True, nulls_last=False)


def desc_nulls_first(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    col._desc_status = True
    return col.sort(descending=True, nulls_last=False)


def desc_nulls_last(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    col._desc_status = True
    return col.sort(descending=True, nulls_last=True)


def array(*cols: str | Expr) -> Expr:
    if not cols:
        return plf.concat_list(plf.lit([]))
    cols = [_str_to_col(col) for col in cols]
    return plf.concat_list(*cols)


def array_append(col: str | Expr, value: Any) -> Expr:
    col = _str_to_col(col)
    if isinstance(value, str):
        value = lit(value)
    return col.list.concat(value)


def array_compact(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.list.drop_nulls()


def array_contains(col: str | Expr, value: Any) -> Expr:
    col = _str_to_col(col)
    if isinstance(value, str):
        value = lit(value)
    return col.list.contains(value)


def array_except(col1: str | Expr, col2: str | Expr) -> Expr:
    col1 = _str_to_col(col1)
    col2 = _str_to_col(col2)
    return col1.list.set_difference(col2)


def array_intersect(col1: str | Expr, col2: str | Expr) -> Expr:
    col1 = _str_to_col(col1)
    col2 = _str_to_col(col2)
    return col1.list.set_intersection(col2)


def array_join(col: str | Expr, delimiter: str, null_replacement: str | None = None) -> Expr:
    col = _str_to_col(col)
    if null_replacement is not None:
        col = col.list.eval(
            pl.coalesce(
                pl.element().cast(plf_types.String(), strict=False),
                lit(null_replacement).cast(plf_types.String(), strict=False),
            ),
        )
    else:
        col = col.list.eval(pl.element().cast(plf_types.String(), strict=False))
    return col.list.join(delimiter, ignore_nulls=True)


def array_max(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.list.max()


def array_min(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.list.min()


def array_size(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.list.len()


size = array_size


def array_union(col1: str | Expr, col2: str | Expr) -> Expr:
    col1 = _str_to_col(col1)
    col2 = _str_to_col(col2)
    return col1.list.set_union(col2)


def array_sort(col: str | Expr, comparator: Any = None) -> Expr:
    if comparator is not None:
        msg = "Custom comparator for array_sort is not supported in Polars."
        raise NotImplementedError(msg)
    col = _str_to_col(col)
    return col.list.sort(descending=False, nulls_last=False)


def sort_array(col: str | Expr, asc: bool = True) -> Expr:
    col = _str_to_col(col)
    if asc:
        return col.list.sort(descending=False, nulls_last=False)
    return col.list.sort(descending=True, nulls_last=True)


def slice(x: str | Expr, start: int, length: int | None = None) -> Expr:
    x = _str_to_col(x)
    # Adjust for 1-based indexing (Spark) to 0-based (Polars)
    start_idx = start - 1 if start > 0 else start
    if length is not None:
        return x.list.slice(start_idx, length)
    return x.list.slice(start_idx)


def array_remove(col: str | Expr, element: Any) -> Expr:
    col = _str_to_col(col)
    if isinstance(element, str):
        element = lit(element)
    return col.list.eval(pl.element().filter(pl.element() != element))


def flatten(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.flatten()


def explode(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.list.drop_nulls().explode()


def count(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.count()


coalesce = plf.coalesce


def monotonically_increasing_id() -> Expr:
    return plf.int_range(pl.len(), dtype=pl.UInt32)


def product(self: str | Expr) -> Expr:
    self = _str_to_col(self)
    return self.product()


def year(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.dt.year()


def month(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.dt.month()


def hour(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.dt.hour()


def last_day(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.dt.month_end()


def dayofmonth(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.dt.day()


def dayofweek(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.dt.weekday() + 1  # Spark's dayofweek starts from 1 (Sunday)


def dayofyear(col: str | Expr) -> Expr:
    col = _str_to_col(col)
    return col.dt.ordinal_day()


def current_date() -> Expr:
    return plf.lit(datetime.datetime.now().date(), dtype=plf_types.Date)  # noqa: DTZ005


now = current_date
curdate = current_date


def current_timestamp() -> Expr:
    return plf.lit(datetime.datetime.now(), dtype=plf_types.Datetime)  # noqa: DTZ005


localtimestamp = current_timestamp


def date_sub(col: str | Expr, days: int) -> Expr:
    col = _str_to_col(col)
    return col.dt.offset_by(f"{days}d")


def date_add(col: str | Expr, days: int) -> Expr:
    col = _str_to_col(col)
    return col.dt.offset_by(f"{days}d")


def datediff(end: str | Expr, start: str | Expr) -> Expr:
    end = _str_to_col(end)
    start = _str_to_col(start)
    return end - start


def add_months(col: str | Expr, months: int) -> Expr:
    col = _str_to_col(col)
    return col.dt.offset_by(f"{months}M")


def sequence(start: int | Expr, stop: int | Expr, step: int | None = None) -> Expr:
    if step is None:
        return plf.int_range(start, stop + 1, eager=True)
    return plf.int_range(start, stop + 1, step=step, eager=True)
