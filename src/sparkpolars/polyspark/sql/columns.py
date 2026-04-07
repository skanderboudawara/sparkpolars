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


# ── string methods ────────────────────────────────────────────────────────────

def upper(self: Expr) -> Expr:
    return self.str.to_uppercase()


def lower(self: Expr) -> Expr:
    return self.str.to_lowercase()


def trim(self: Expr, trim_chars: str | None = None) -> Expr:
    return self.str.strip_chars(trim_chars)


def btrim(self: Expr, trim_chars: str | None = None) -> Expr:
    return self.str.strip_chars(trim_chars)


def ltrim(self: Expr, trim_chars: str | None = None) -> Expr:
    return self.str.strip_chars_start(trim_chars)


def rtrim(self: Expr, trim_chars: str | None = None) -> Expr:
    return self.str.strip_chars_end(trim_chars)


def lpad(self: Expr, length: int, pad: str = " ") -> Expr:
    return self.str.pad_start(length, fill_char=pad)


def rpad(self: Expr, length: int, pad: str = " ") -> Expr:
    return self.str.pad_end(length, fill_char=pad)


def left(self: Expr, n: int) -> Expr:
    return self.str.slice(0, n)


def right(self: Expr, n: int) -> Expr:
    return self.str.slice(-n, n) if n > 0 else self.str.slice(-n)


def length(self: Expr) -> Expr:
    return self.str.len_chars()


def locate(self: Expr, substr: str) -> Expr:
    return self.str.find(substr)


def repeat(self: Expr, n: int) -> Expr:
    from functools import reduce
    s = self.cast(pl.String, strict=False)
    return reduce(lambda acc, _: acc + s, range(n - 1), s)


def reverse_str(self: Expr) -> Expr:
    return self.str.reverse()


def split_str(self: Expr, pattern: str, limit: int = -1) -> Expr:
    if limit <= 0:
        return self.str.split(pattern)
    return self.str.split(pattern).list.head(limit)


def base64(self: Expr) -> Expr:
    return self.str.encode("base64")


def encode(self: Expr, charset: str) -> Expr:
    return self.str.encode(charset)


def decode(self: Expr, charset: str) -> Expr:
    return self.str.decode(charset)


def regexp_count(self: Expr, pattern: str) -> Expr:
    return self.str.count_matches(pattern)


def regexp_extract(self: Expr, pattern: str, idx: int) -> Expr:
    return self.str.extract(pattern, idx)


def regexp_extract_all(self: Expr, pattern: str) -> Expr:
    return self.str.extract_all(pattern)


def regexp_replace(self: Expr, pattern: str, replacement: str) -> Expr:
    return self.str.replace(pattern, replacement)


def str_replace(self: Expr, search: str, replacement: str | None = None) -> Expr:
    if replacement is None:
        return self.str.replace_all(search, "")
    return self.str.replace(search, replacement)


def translate(self: Expr, matching: str, replace: str) -> Expr:
    import polars.datatypes as _dt
    if len(matching) != len(replace):
        msg = "matching and replace strings must have the same length"
        raise ValueError(msg)
    trans_table = str.maketrans(matching, replace)
    return self.map_elements(lambda x: x.translate(trans_table), _dt.String())


Expr.upper = upper
Expr.lower = lower
Expr.trim = trim
Expr.btrim = btrim
Expr.ltrim = ltrim
Expr.rtrim = rtrim
Expr.lpad = lpad
Expr.rpad = rpad
Expr.left = left
Expr.right = right
Expr.length = length
Expr.locate = locate
Expr.position = locate
Expr.repeat = repeat
Expr.reverse = reverse_str
Expr.split = split_str
Expr.base64 = base64
Expr.encode = encode
Expr.decode = decode
Expr.regexp_count = regexp_count
Expr.regexp_extract = regexp_extract
Expr.regexp_extract_all = regexp_extract_all
Expr.regexp_replace = regexp_replace
Expr.str_replace = str_replace
Expr.translate = translate


# ── date / time methods ───────────────────────────────────────────────────────

def year(self: Expr) -> Expr:
    return self.dt.year()


def month(self: Expr) -> Expr:
    return self.dt.month()


def hour(self: Expr) -> Expr:
    return self.dt.hour()


def last_day(self: Expr) -> Expr:
    return self.dt.month_end()


def dayofmonth(self: Expr) -> Expr:
    return self.dt.day()


def dayofweek(self: Expr) -> Expr:
    return self.dt.weekday() + 1  # Spark: 1=Sunday


def dayofyear(self: Expr) -> Expr:
    return self.dt.ordinal_day()


def date_add(self: Expr, days: int) -> Expr:
    return self.dt.offset_by(f"{days}d")


def date_sub(self: Expr, days: int) -> Expr:
    return self.dt.offset_by(f"-{days}d")


def add_months(self: Expr, months: int) -> Expr:
    return self.dt.offset_by(f"{months}mo")


Expr.year = year
Expr.month = month
Expr.hour = hour
Expr.last_day = last_day
Expr.dayofmonth = dayofmonth
Expr.dayofweek = dayofweek
Expr.dayofyear = dayofyear
Expr.date_add = date_add
Expr.date_sub = date_sub
Expr.add_months = add_months


# ── hash methods ──────────────────────────────────────────────────────────────

import hashlib as _hashlib  # noqa: E402
import polars.datatypes as _polars_datatypes  # noqa: E402


def md5(self: Expr) -> Expr:
    return self.map_elements(
        lambda x: _hashlib.md5(str(x).encode()).hexdigest() if x is not None else None,  # noqa: S324
        _polars_datatypes.String(),
    )


def sha1(self: Expr) -> Expr:
    return self.map_elements(
        lambda x: _hashlib.sha1(str(x).encode()).hexdigest() if x is not None else None,  # noqa: S324
        _polars_datatypes.String(),
    )


def sha256(self: Expr) -> Expr:
    return self.map_elements(
        lambda x: _hashlib.sha256(str(x).encode()).hexdigest() if x is not None else None,
        _polars_datatypes.String(),
    )


Expr.md5 = md5
Expr.sha1 = sha1
Expr.sha256 = sha256


# ── array / list methods ──────────────────────────────────────────────────────

def array_distinct(self: Expr) -> Expr:
    return self.list.unique()


def array_compact(self: Expr) -> Expr:
    return self.list.drop_nulls()


def array_contains(self: Expr, value: Any) -> Expr:
    v = pl.lit(value) if isinstance(value, str) else value
    return self.list.contains(v)


def array_append(self: Expr, value: Any) -> Expr:
    v = pl.lit(value) if isinstance(value, str) else value
    return self.list.concat(v)


def array_remove(self: Expr, element: Any) -> Expr:
    e = pl.lit(element) if isinstance(element, str) else element
    return self.list.eval(pl.element().filter(pl.element() != e))


def array_union(self: Expr, other: Expr) -> Expr:
    return self.list.set_union(other)


def array_intersect(self: Expr, other: Expr) -> Expr:
    return self.list.set_intersection(other)


def array_except(self: Expr, other: Expr) -> Expr:
    return self.list.set_difference(other)


def array_join(self: Expr, delimiter: str, null_replacement: str | None = None) -> Expr:
    col = self
    if null_replacement is not None:
        col = col.list.eval(
            pl.coalesce(
                pl.element().cast(pl.String, strict=False),
                pl.lit(null_replacement).cast(pl.String, strict=False),
            )
        )
    else:
        col = col.list.eval(pl.element().cast(pl.String, strict=False))
    return col.list.join(delimiter, ignore_nulls=True)


def array_max(self: Expr) -> Expr:
    return self.list.max()


def array_min(self: Expr) -> Expr:
    return self.list.min()


def array_size(self: Expr) -> Expr:
    return self.list.len()


def array_sort(self: Expr, asc: bool = True) -> Expr:
    return self.list.sort(descending=not asc, nulls_last=False)


def array_slice(self: Expr, start: int, length: int | None = None) -> Expr:
    # Spark uses 1-based indexing
    start_idx = start - 1 if start > 0 else start
    if length is not None:
        return self.list.slice(start_idx, length)
    return self.list.slice(start_idx)






def list_filter(self: Expr, f: Callable) -> Expr:
    return self.list.eval(pl.element().filter(f(pl.element())))


def forall(self: Expr, f: Callable) -> Expr:
    return self.list.eval(f(pl.element())).list.all()


def collect_list(self: Expr) -> Expr:
    return self.implode()


def collect_set(self: Expr) -> Expr:
    return self.unique().implode()


Expr.array_distinct = array_distinct
Expr.array_compact = array_compact
Expr.array_contains = array_contains
Expr.array_append = array_append
Expr.array_remove = array_remove
Expr.array_union = array_union
Expr.array_intersect = array_intersect
Expr.array_except = array_except
Expr.array_join = array_join
Expr.array_max = array_max
Expr.array_min = array_min
Expr.array_size = array_size
Expr.size = array_size
Expr.array_sort = array_sort
Expr.array_slice = array_slice
# Expr.flatten — native Polars Expr.flatten() already flattens List[List[T]] correctly
Expr.list_filter = list_filter
Expr.forall = forall
Expr.collect_list = collect_list
Expr.collect_set = collect_set


# ── math / aggregation ────────────────────────────────────────────────────────
# round, sqrt, abs, pow, sum, min, max, mean, count, first, last, product
# are already native on Polars Expr — no patching needed.

def negate(self: Expr) -> Expr:
    return -self


def avg(self: Expr) -> Expr:
    return self.mean()


Expr.negate = negate
Expr.avg = avg


# ── math Expr methods ─────────────────────────────────────────────────────────
# ceil, floor, abs, round, sqrt, pow, product, log, exp, sign — already native.
# We only expose the missing aliases.

def log2(self: Expr) -> Expr:
    return self.log(2.0)


def log10(self: Expr) -> Expr:
    return self.log(10.0)


def cbrt(self: Expr) -> Expr:
    return self ** (1.0 / 3.0)


def signum(self: Expr) -> Expr:
    return self.sign()


Expr.log2 = log2
Expr.log10 = log10
Expr.cbrt = cbrt
Expr.signum = signum


# ── null-handling Expr methods ────────────────────────────────────────────────

def nvl(self: Expr, replacement: Any) -> Expr:
    r = replacement if isinstance(replacement, Expr) else pl.lit(replacement)
    return polars_functions.coalesce(self, r)


def nullif(self: Expr, value: Any) -> Expr:
    v = value if isinstance(value, Expr) else pl.lit(value)
    return polars_functions.when(self != v).then(self)


Expr.nvl = nvl
Expr.ifnull = nvl
Expr.nullif = nullif


# ── date / time Expr methods ──────────────────────────────────────────────────

def to_date(self: Expr, fmt: str = "%Y-%m-%d") -> Expr:
    return self.str.to_date(fmt)


def to_timestamp(self: Expr, fmt: str = "%Y-%m-%d %H:%M:%S") -> Expr:
    return self.str.to_datetime(fmt)


def date_format(self: Expr, fmt: str) -> Expr:
    return self.dt.strftime(fmt)


def unix_timestamp(self: Expr) -> Expr:
    return self.dt.epoch(time_unit="s")


Expr.to_date = to_date
Expr.to_timestamp = to_timestamp
Expr.date_format = date_format
Expr.unix_timestamp = unix_timestamp


# ── array Expr methods ────────────────────────────────────────────────────────

def array_position(self: Expr, value: Any) -> Expr:
    """1-based index of value in the array, 0 if not found."""
    v = value if isinstance(value, Expr) else pl.lit(value)
    idx = self.list.eval(pl.arg_where(pl.element() == v).first()).list.first()
    return polars_functions.when(self.list.contains(v)).then(idx + 1).otherwise(pl.lit(0))


def array_prepend(self: Expr, value: Any) -> Expr:
    v = value if isinstance(value, Expr) else pl.lit(value)
    result = polars_functions.concat_list([v.implode(), self])
    try:
        return result.alias(self.meta.output_name())
    except Exception:
        return result


Expr.array_position = array_position
Expr.array_prepend = array_prepend


# ── trig aliases ──────────────────────────────────────────────────────────────
# All trig functions are native on Polars Expr; add PySpark-style aliases.

Expr.asin = Expr.arcsin
Expr.acos = Expr.arccos
Expr.atan = Expr.arctan
Expr.asinh = Expr.arcsinh
Expr.acosh = Expr.arccosh
Expr.atanh = Expr.arctanh


# ── string extras ──────────────────────────────────────────────────────────────

def initcap(self: Expr) -> Expr:
    return self.str.to_titlecase()


def ascii_code(self: Expr) -> Expr:
    return self.map_elements(
        lambda x: ord(x[0]) if x else None,
        _polars_datatypes.Int32(),
    )


def instr(self: Expr, substr: str) -> Expr:
    """Return 1-based position of substr in string; 0 if not found."""
    idx = self.str.find(substr, literal=True)
    return polars_functions.when(idx.is_not_null()).then(idx + 1).otherwise(pl.lit(0))


def split_part(self: Expr, delimiter: str, part_num: int) -> Expr:
    """Split by delimiter and return part_num-th part (1-based); '' if out of range."""
    return self.str.split(delimiter).list.get(part_num - 1, null_on_oob=True).fill_null("")


def substring_index(self: Expr, delimiter: str, count: int) -> Expr:
    """Return substring before the count-th occurrence of delimiter."""
    if count > 0:
        return self.str.split(delimiter).list.head(count).list.join(delimiter)
    return self.str.split(delimiter).list.tail(-count).list.join(delimiter)


Expr.initcap = initcap
Expr.ascii_code = ascii_code
Expr.instr = instr
Expr.split_part = split_part
Expr.substring_index = substring_index


# ── date extras ────────────────────────────────────────────────────────────────

def quarter(self: Expr) -> Expr:
    return ((self.dt.month() - 1) // 3 + 1).cast(_polars_datatypes.Int32())


def minute(self: Expr) -> Expr:
    return self.dt.minute()


def second(self: Expr) -> Expr:
    return self.dt.second()


def weekofyear(self: Expr) -> Expr:
    return self.dt.week()


def weekday(self: Expr) -> Expr:
    """Day of week: 0=Monday, 6=Sunday (same as PySpark weekday())."""
    return self.dt.weekday() - 1


_DATE_TRUNC_UNIT_MAP = {
    "year": "1y", "yyyy": "1y", "yy": "1y",
    "month": "1mo", "mm": "1mo", "mon": "1mo",
    "week": "1w",
    "day": "1d", "dd": "1d",
    "hour": "1h",
    "minute": "1m",
    "second": "1s",
}


def date_trunc(self: Expr, unit: str) -> Expr:
    """Truncate date/timestamp to the given Spark unit string."""
    unit_lower = unit.lower()
    if unit_lower in ("quarter", "q"):
        import datetime as _dt

        def _to_quarter_start(d: Any) -> Any:
            if d is None:
                return None
            q_month = ((d.month - 1) // 3) * 3 + 1
            return _dt.date(d.year, q_month, 1)

        return self.map_elements(_to_quarter_start, _polars_datatypes.Date())
    polars_unit = _DATE_TRUNC_UNIT_MAP.get(unit_lower, unit_lower)
    return self.dt.truncate(polars_unit)


Expr.quarter = quarter
Expr.minute = minute
Expr.second = second
Expr.weekofyear = weekofyear
Expr.weekday = weekday
Expr.date_trunc = date_trunc


# ── aggregate extras ───────────────────────────────────────────────────────────

def stddev(self: Expr) -> Expr:
    return self.std()


def variance(self: Expr) -> Expr:
    return self.var()


def count_distinct(self: Expr) -> Expr:
    return self.n_unique()


def bool_and(self: Expr) -> Expr:
    return self.all()


def bool_or(self: Expr) -> Expr:
    return self.any()


Expr.stddev = stddev
Expr.std_dev = stddev
Expr.variance = variance
Expr.count_distinct = count_distinct
Expr.bool_and = bool_and
Expr.every = bool_and
Expr.bool_or = bool_or
Expr.some = bool_or


# ── null extras ────────────────────────────────────────────────────────────────

def nanvl(self: Expr, replacement: Any) -> Expr:
    """Return replacement when col is NaN, else col."""
    r = replacement if isinstance(replacement, Expr) else pl.lit(replacement)
    result = polars_functions.when(self.is_nan()).then(r).otherwise(self)
    try:
        return result.alias(self.meta.output_name())
    except Exception:
        return result


def nvl2(self: Expr, not_null_val: Any, null_val: Any) -> Expr:
    """Return not_null_val when col is not null, null_val when col is null."""
    n = not_null_val if isinstance(not_null_val, Expr) else pl.lit(not_null_val)
    nv = null_val if isinstance(null_val, Expr) else pl.lit(null_val)
    result = polars_functions.when(self.is_not_null()).then(n).otherwise(nv)
    try:
        return result.alias(self.meta.output_name())
    except Exception:
        return result


Expr.nanvl = nanvl
Expr.nvl2 = nvl2


# ── bitwise extras ─────────────────────────────────────────────────────────────

def bitwiseNOT(self: Expr) -> Expr:
    return ~self


def shiftLeft(self: Expr, n: int) -> Expr:
    return self.map_elements(
        lambda x: x << n if x is not None else None, _polars_datatypes.Int64()
    )


def shiftRight(self: Expr, n: int) -> Expr:
    return self.map_elements(
        lambda x: x >> n if x is not None else None, _polars_datatypes.Int64()
    )


Expr.bitwiseNOT = bitwiseNOT
Expr.shiftLeft = shiftLeft
Expr.shiftRight = shiftRight
Expr.shiftleft = shiftLeft
Expr.shiftright = shiftRight


# ── array extras ───────────────────────────────────────────────────────────────

def element_at(self: Expr, index: int) -> Expr:
    """Return element at 1-based index (negative counts from end). Returns null if OOB."""
    idx = index - 1 if index > 0 else index
    return self.list.get(idx, null_on_oob=True)


def arrays_overlap(self: Expr, other: Expr) -> Expr:
    """Return true if the arrays share at least one common element."""
    return self.list.set_intersection(other).list.len() > 0


Expr.element_at = element_at
Expr.arrays_overlap = arrays_overlap


# ── window helpers ─────────────────────────────────────────────────────────────

def lag(self: Expr, offset: int = 1, default: Any = None) -> Expr:
    """Shift values forward by offset rows within the current partition."""
    if default is not None:
        return self.shift(offset, fill_value=default)
    return self.shift(offset)


def lead(self: Expr, offset: int = 1, default: Any = None) -> Expr:
    """Shift values backward by offset rows within the current partition."""
    if default is not None:
        return self.shift(-offset, fill_value=default)
    return self.shift(-offset)


Expr.lag = lag
Expr.lead = lead


# ── string extras (batch 2) ───────────────────────────────────────────────────

def chr_char(self: Expr) -> Expr:
    """Convert Unicode code point integer to a single character string."""
    return self.map_elements(
        lambda x: chr(int(x)) if x is not None else None,
        _polars_datatypes.String(),
    )


def find_in_set(self: Expr, str_val: str) -> Expr:
    """Return 1-based position of str_val in comma-delimited self; 0 if not found."""
    parts = self.str.split(",")
    idx = parts.list.eval(pl.arg_where(pl.element() == str_val).first()).list.first()
    return polars_functions.when(parts.list.contains(str_val)).then(idx + 1).otherwise(pl.lit(0))


def regexp_like(self: Expr, pattern: str) -> Expr:
    """Return true if the string matches the regex pattern."""
    return self.str.contains(pattern, literal=False)


def overlay(self: Expr, replace: Any, pos: int, length: int | None = None) -> Expr:
    """Replace portion of self starting at 1-based pos with replace string."""
    r = replace if isinstance(replace, Expr) else pl.lit(replace)
    s = polars_functions.struct([self.alias("_s"), r.alias("_r")])

    def _overlay(row: Any) -> Any:
        sv, rv = row["_s"], row["_r"]
        if sv is None or rv is None:
            return None
        start = pos - 1
        end = start + (length if length is not None else len(rv))
        return sv[:start] + rv + sv[end:]

    result = s.map_elements(_overlay, _polars_datatypes.String())
    try:
        return result.alias(self.meta.output_name())
    except Exception:
        return result


Expr.chr = chr_char
Expr.find_in_set = find_in_set
Expr.regexp_like = regexp_like
Expr.overlay = overlay


# ── math extras (batch 2) ─────────────────────────────────────────────────────

def bround(self: Expr, d: int = 0) -> Expr:
    """Banker's rounding (half-to-even). Polars round() already uses HALF_TO_EVEN."""
    return self.round(d)


def pmod(self: Expr, divisor: Any) -> Expr:
    """Positive modulo — result always has the same sign as the divisor."""
    dv = divisor if isinstance(divisor, Expr) else pl.lit(divisor)
    return ((self % dv) + dv) % dv


Expr.bround = bround
Expr.pmod = pmod


# ── date extras (batch 2) ─────────────────────────────────────────────────────

_WEEKDAY_NAME_MAP = {
    "mon": 0, "monday": 0,
    "tue": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


def next_day(self: Expr, day_of_week: str) -> Expr:
    """Return the first date strictly after self that is the named day of week."""
    import datetime as _dt

    target_wd = _WEEKDAY_NAME_MAP[day_of_week.lower()[:3]]

    def _next(d: Any) -> Any:
        if d is None:
            return None
        days_ahead = (target_wd - d.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return d + _dt.timedelta(days=days_ahead)

    return self.map_elements(_next, _polars_datatypes.Date())


def unix_date(self: Expr) -> Expr:
    """Number of days since Unix epoch (1970-01-01)."""
    return self.dt.epoch(time_unit="d")


Expr.next_day = next_day
Expr.unix_date = unix_date


# ── array extras (batch 2) ────────────────────────────────────────────────────

def exists(self: Expr, f: Callable) -> Expr:
    """Return true if any element in the list satisfies predicate f."""
    return self.list.eval(f(pl.element())).list.any()


def shuffle(self: Expr, seed: int | None = None) -> Expr:
    """Return the list with elements in a random order."""
    return self.list.sample(n=self.list.len(), with_replacement=False, shuffle=True, seed=seed)


def aggregate_list(self: Expr, zero: Any, merge: Callable, finish: Callable | None = None) -> Expr:
    """Fold over list: reduce(merge, elements, zero), then optionally apply finish."""
    from functools import reduce as _reduce

    def _fold(lst: Any) -> Any:
        if lst is None:
            return None
        result = _reduce(merge, lst.to_list(), zero)
        return finish(result) if finish is not None else result

    return self.map_elements(_fold, return_dtype=None)


Expr.exists = exists
Expr.shuffle = shuffle
Expr.aggregate = aggregate_list


# ── aggregate extras (batch 2) ────────────────────────────────────────────────
# kurtosis() is already native on Polars Expr; add skewness alias and helpers.

Expr.skewness = Expr.skew


def percentile(self: Expr, pct: float) -> Expr:
    """Return the pct-th percentile (0.0–1.0 scale)."""
    return self.quantile(pct, interpolation="nearest")


def sum_distinct(self: Expr) -> Expr:
    """Sum of distinct (unique) values."""
    return self.unique().sum()


def approx_count_distinct(self: Expr, rsd: float = 0.05) -> Expr:  # noqa: ARG001
    """Approximate distinct count (exact in Polars)."""
    return self.n_unique()


Expr.percentile = percentile
Expr.sum_distinct = sum_distinct
Expr.approx_count_distinct = approx_count_distinct


# ── string extras (batch 3) ───────────────────────────────────────────────────

def unbase64(self: Expr) -> Expr:
    """Decode a Base64-encoded string column to a UTF-8 string."""
    import base64 as _b64

    return self.map_elements(
        lambda x: _b64.b64decode(x).decode("utf-8") if x is not None else None,
        _polars_datatypes.String(),
    )


def regexp_substr(self: Expr, pattern: str) -> Expr:
    """Return the first substring matching the regexp, or null if no match."""
    return self.str.extract(pattern, 0)


def levenshtein(self: Expr, other: Any) -> Expr:
    """Compute the Levenshtein edit distance between two string columns."""
    other_expr = other if isinstance(other, Expr) else pl.lit(other)
    s = polars_functions.struct([self.alias("_l"), other_expr.alias("_r")])

    def _lev(row: Any) -> Any:
        s1, s2 = row["_l"], row["_r"]
        if s1 is None or s2 is None:
            return None
        m, n = len(s1), len(s2)
        row_arr = list(range(n + 1))
        for i in range(1, m + 1):
            prev = i
            for j in range(1, n + 1):
                curr = row_arr[j - 1] if s1[i - 1] == s2[j - 1] else 1 + min(row_arr[j - 1], row_arr[j], prev)
                row_arr[j - 1] = prev
                prev = curr
            row_arr[n] = prev
        return row_arr[n]

    return s.map_elements(_lev, _polars_datatypes.Int32())


Expr.unbase64 = unbase64
Expr.regexp_substr = regexp_substr
Expr.levenshtein = levenshtein


# ── math extras (batch 3) ─────────────────────────────────────────────────────

import math as _math  # noqa: E402


def log1p(self: Expr) -> Expr:
    """Natural log of (1 + x)."""
    return (self + 1.0).log(_math.e)


def expm1(self: Expr) -> Expr:
    """e^x - 1."""
    return self.exp() - 1.0


def rint(self: Expr) -> Expr:
    """Round to nearest integer (returns float)."""
    return self.round(0)


def bitcount(self: Expr) -> Expr:
    """Count the number of set bits (1s) in the binary representation."""
    return self.map_elements(
        lambda x: bin(int(x)).count("1") if x is not None else None,
        _polars_datatypes.Int32(),
    )


Expr.log1p = log1p
Expr.expm1 = expm1
Expr.rint = rint
Expr.bitcount = bitcount


# ── timestamp extras (batch 3) ────────────────────────────────────────────────

def to_utc_timestamp(self: Expr, tz: str) -> Expr:
    """Interpret timestamp as being in *tz* and convert to UTC."""
    return self.dt.replace_time_zone(tz).dt.convert_time_zone("UTC")


def from_utc_timestamp(self: Expr, tz: str) -> Expr:
    """Interpret timestamp as UTC and convert to *tz*."""
    return self.dt.replace_time_zone("UTC").dt.convert_time_zone(tz)


Expr.to_utc_timestamp = to_utc_timestamp
Expr.from_utc_timestamp = from_utc_timestamp


# ── array extras (batch 3) ────────────────────────────────────────────────────

def array_reverse(self: Expr) -> Expr:
    """Reverse the elements of an array column."""
    return self.list.reverse()


Expr.array_reverse = array_reverse


# ── struct / map extras (batch 3) ─────────────────────────────────────────────

def to_json(self: Expr) -> Expr:
    """Serialize a struct column to a JSON string."""
    return self.struct.json_encode()


def map_keys(self: Expr) -> Expr:
    """Return the keys of a map column (stored as list-of-structs with 'key' field)."""
    return self.list.eval(pl.element().struct.field("key"))


def map_values(self: Expr) -> Expr:
    """Return the values of a map column (stored as list-of-structs with 'value' field)."""
    return self.list.eval(pl.element().struct.field("value"))


Expr.to_json = to_json
Expr.map_keys = map_keys
Expr.map_values = map_values
