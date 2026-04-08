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
    """Cast column to the given dtype (non-strict by default for PySpark compatibility).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> df.select(pl.col("a").cast(pl.Float64))["a"].dtype == pl.Float64
        True
        >>> df.select(pl.col("a").astype(pl.Float64))["a"].dtype == pl.Float64
        True
        >>> df.select(pl.col("a").astype(pl.String))["a"][0]
        '1'
    """
    return self._original_cast(dtype, strict=strict)


Expr.cast = _cast_strict


# ── helpers ───────────────────────────────────────────────────────────────────

def _str_to_col(name: str | Expr) -> Expr:
    if isinstance(name, str):
        return pl.col(name)
    return name


# ── null checks ───────────────────────────────────────────────────────────────

def isNull(self: Expr) -> Expr:
    """Return a boolean expression that is True when the value is null.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, None, 3]})
        >>> df.select(pl.col("a").isNull())["a"].to_list()
        [False, True, False]
    """
    return self.is_null()


def isNotNull(self: Expr) -> Expr:
    """Return a boolean expression that is True when the value is not null.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, None, 3]})
        >>> df.select(pl.col("a").isNotNull())["a"].to_list()
        [True, False, True]
    """
    return self.is_not_null()


Expr.isNull = isNull
Expr.isNotNull = isNotNull


# ── isin ──────────────────────────────────────────────────────────────────────

def isin(self: Expr, *cols: Any) -> Expr:
    """Return a boolean expression that is True when the value is in the given list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> df.filter(pl.col("a").isin(1, 3, 5))["a"].to_list()
        [1, 3, 5]
    """
    if len(cols) == 1:
        return self.is_in(cols[0])
    return self.is_in(cols)


Expr.isin = isin


# ── between ───────────────────────────────────────────────────────────────────

def between(self: Expr, lowerBound: Any, upperBound: Any) -> Expr:
    """Return a boolean expression that is True when the value is between bounds (inclusive).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> df.filter(pl.col("a").between(2, 4))["a"].to_list()
        [2, 3, 4]
    """
    return self.ge(lowerBound) & self.le(upperBound)


Expr.between = between


# ── eqNullSafe ────────────────────────────────────────────────────────────────

def eqNullSafe(self: Expr, other: str | Expr) -> Expr:
    """Null-safe equality comparison.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> df.select(pl.col("a").eqNullSafe(pl.col("b")))["a"].to_list()
        [False, False, True, False, False]
    """
    return self.eq_missing(_str_to_col(other))


Expr.eqNullSafe = eqNullSafe


# ── string methods ────────────────────────────────────────────────────────────

def rlike(self: Expr, regexp: str) -> Expr:
    """Filter rows where the string matches the given regex pattern.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo", "bar", "Hello World"]})
        >>> "hello" in df.filter(pl.col("s").rlike("^h.*"))["s"].to_list()
        True
    """
    return self.str.contains(regexp, literal=False)


Expr.rlike = rlike


def startswith(self: Expr, prefix: str) -> Expr:
    """Filter rows where the string starts with the given prefix.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo", "bar", "Hello World"]})
        >>> df.filter(pl.col("s").startswith("he"))["s"].to_list()
        ['hello']
    """
    return self.str.starts_with(prefix)


Expr.startswith = startswith


def endswith(self: Expr, suffix: str) -> Expr:
    """Filter rows where the string ends with the given suffix.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo", "bar", "Hello World"]})
        >>> sorted(df.filter(pl.col("s").endswith("ld"))["s"].to_list())
        ['Hello World', 'world']
    """
    return self.str.ends_with(suffix)


Expr.endswith = endswith


def substr(self: Expr, pos: int, length: int | None = None) -> Expr:
    """Extract a substring starting at 1-based position with optional length.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo", "bar", "Hello World"]})
        >>> df.select(pl.col("s").substr(1, 3))["s"][0]
        'hel'
    """
    if length is not None:
        return self.str.slice(pos - 1, length)
    return self.str.slice(pos - 1)


Expr.substr = substr


def contains(self: Expr, right: str) -> Expr:
    """Filter rows where the string contains the given substring.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo", "bar", "Hello World"]})
        >>> df.filter(pl.col("s").contains("oo"))["s"].to_list()
        ['foo']
    """
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
    """SQL LIKE pattern matching (case-sensitive). ``%`` matches any chars, ``_`` matches one char.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo", "bar", "Hello World"]})
        >>> df.filter(pl.col("s").like("hel%"))["s"].to_list()
        ['hello']
        >>> df.filter(pl.col("s").like("_oo"))["s"].to_list()
        ['foo']
        >>> df.filter(pl.col("s").like("bar"))["s"].to_list()
        ['bar']
        >>> "Hello World" not in df.filter(pl.col("s").like("hel%"))["s"].to_list()
        True
    """
    return self.str.contains(_sql_like_to_regex(pattern), literal=False)


def ilike(self: Expr, pattern: str) -> Expr:
    """SQL ILIKE pattern matching (case-insensitive). ``%`` matches any chars, ``_`` matches one char.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo", "bar", "Hello World"]})
        >>> result = df.filter(pl.col("s").ilike("hel%"))["s"].to_list()
        >>> "hello" in result and "Hello World" in result
        True
        >>> df.filter(pl.col("s").ilike("_oo"))["s"].to_list()
        ['foo']
    """
    return self.str.contains("(?i)" + _sql_like_to_regex(pattern), literal=False)


Expr.like = like
Expr.ilike = ilike


# ── isNaN ─────────────────────────────────────────────────────────────────────

def isNaN(self: Expr) -> Expr:
    """Return a boolean expression that is True when the value is NaN.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"v": [1.0, float("nan"), 3.0, float("nan"), 5.0]})
        >>> df.select(pl.col("v").isNaN())["v"].to_list()
        [False, True, False, True, False]
    """
    return self.is_nan()


Expr.isNaN = isNaN


# ── astype ────────────────────────────────────────────────────────────────────

Expr.astype = Expr.cast


# ── bitwise ───────────────────────────────────────────────────────────────────

def bitwiseAND(self: Expr, other: Any) -> Expr:
    """Compute bitwise AND with another expression.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [0b1010, 0b1100], "b": [0b1100, 0b1010]})
        >>> df.select(pl.col("a").bitwiseAND(pl.col("b")).alias("r"))["r"].to_list()
        [8, 8]
        >>> df2 = pl.DataFrame({"a": [0b1111, 0b1010]})
        >>> df2.select(pl.col("a").bitwiseAND(pl.lit(0b1010)).alias("r"))["r"].to_list()
        [10, 10]
    """
    return self & other


def bitwiseOR(self: Expr, other: Any) -> Expr:
    """Compute bitwise OR with another expression.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [0b1010, 0b0000], "b": [0b0101, 0b1111]})
        >>> df.select(pl.col("a").bitwiseOR(pl.col("b")).alias("r"))["r"].to_list()
        [15, 15]
    """
    return self | other


def bitwiseXOR(self: Expr, other: Any) -> Expr:
    """Compute bitwise XOR with another expression.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [0b1010, 0b1100], "b": [0b1100, 0b1010]})
        >>> df.select(pl.col("a").bitwiseXOR(pl.col("b")).alias("r"))["r"].to_list()
        [6, 6]
    """
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
    """Sort expression in ascending order.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> df.select(pl.col("a").asc())["a"].to_list()
        [1, 2, 3, 4, 5]
    """
    self._desc_status = False
    return self.sort(descending=False, nulls_last=False)


def asc_nulls_first(self: Expr) -> Expr:
    """Sort ascending with nulls first.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [None, 3, 1]})
        >>> df.select(pl.col("a").asc_nulls_first())["a"].to_list()
        [None, 1, 3]
    """
    self._desc_status = False
    return self.sort(descending=False, nulls_last=False)


def asc_nulls_last(self: Expr) -> Expr:
    """Sort ascending with nulls last.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [None, 3, 1]})
        >>> df.select(pl.col("a").asc_nulls_last())["a"].to_list()
        [1, 3, None]
    """
    self._desc_status = False
    return self.sort(descending=False, nulls_last=True)


def desc(self: Expr) -> Expr:
    """Sort expression in descending order.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> df.select(pl.col("a").desc())["a"].to_list()
        [5, 4, 3, 2, 1]
    """
    self._desc_status = True
    return self.sort(descending=True, nulls_last=False)


def desc_nulls_first(self: Expr) -> Expr:
    """Sort descending with nulls first.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [None, 1, 3]})
        >>> df.select(pl.col("a").desc_nulls_first())["a"].to_list()
        [None, 3, 1]
    """
    self._desc_status = True
    return self.sort(descending=True, nulls_last=False)


def desc_nulls_last(self: Expr) -> Expr:
    """Sort descending with nulls last.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [None, 1, 3]})
        >>> df.select(pl.col("a").desc_nulls_last())["a"].to_list()
        [3, 1, None]
    """
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
    """Extract a field from a struct column by name.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]})
        >>> df.select(pl.col("s").getField("x"))["x"].to_list()
        [1, 3]
        >>> df.select(pl.col("s").getField("y"))["y"].to_list()
        [2, 4]
    """
    return self.struct.field(name)


Expr.getField = getField


def withField(self: Expr, fieldName: str, col: Expr | str) -> Expr:
    """Add or replace a field in a struct column.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]})
        >>> df.select(pl.col("s").withField("z", pl.lit(99)))["s"][0]["z"]
        99
        >>> result = df.select(pl.col("s").withField("x", pl.lit(0)))["s"][0]
        >>> result["x"]
        0
        >>> result["y"]
        2
    """
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
    """Cast to dtype, returning null on failure instead of raising.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["1", "2", "3"]})
        >>> df.select(pl.col("s").try_cast(pl.Int64))["s"].to_list()
        [1, 2, 3]
        >>> df2 = pl.DataFrame({"s": ["1", "abc", "3"]})
        >>> df2.select(pl.col("s").try_cast(pl.Int64))["s"][1] is None
        True
    """
    return self._original_cast(dtype, strict=False)


Expr.try_cast = try_cast


# ── transform (list columns) ──────────────────────────────────────────────────

def transform_list(self: Expr, f: Callable) -> Expr:
    """Apply a function to each element in a list column.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"arr": [[1, 2, 3], [4, 5, 6]]})
        >>> df.select(pl.col("arr").transform(lambda e: e * 2))["arr"].to_list()[0]
        [2, 4, 6]
    """
    return self.list.eval(f(pl.element()))


Expr.transform = transform_list


# ── getItem (JSON map) ────────────────────────────────────────────────────────

def getItem(self: Expr, key: str | Expr) -> Expr:
    """Extract a value from a JSON-encoded map column using a key.

    Examples:
        >>> import polars as pl
        >>> import json
        >>> df = pl.DataFrame({"m": [json.dumps({"a": "v1"}), json.dumps({"a": "v2"})]})
        >>> df.select(pl.col("m").getItem(pl.lit("a")).alias("val"))["val"].to_list()
        ['v1', 'v2']
    """
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
    """Apply an expression over a window partition.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> from src.sparkpolars.polyspark.sql.window import Window, WindowSpec
        >>> df = pl.DataFrame({"dept": ["A", "A", "B", "B"], "sal": [100, 200, 300, 400]})
        >>> df.select(pl.col("sal").sum().over(Window.partitionBy("dept")).alias("total"))["total"].to_list()
        [300, 300, 700, 700]
    """
    from .window import WindowSpec, _to_pyexprs  # noqa: PLC0415

    # ── WindowSpec (new API) ──────────────────────────────────────────────────
    if isinstance(partition_by, WindowSpec):
        ws: WindowSpec = partition_by
        pb = ws._partition_by
        ob = ws._order_by
        desc = ws._order_by_desc

        if not pb and not ob:
            return self

        if pb:
            # Use the saved original Polars .over() — it properly handles Sequence[bool]
            if ob:
                return self._polars_over(
                    pb,
                    order_by=ob,
                    descending=desc[0] if desc else False,
                    mapping_strategy="group_to_rows",
                )
            return self._polars_over(pb, mapping_strategy="group_to_rows")

        # only order_by, no partition
        return self.sort_by(ob, descending=desc)

    # ── legacy WindowClass / raw columns ─────────────────────────────────────
    if hasattr(partition_by, "_partition_by"):
        window_partition_by = partition_by._partition_by
        window_order_by = partition_by._order_by
        window_sort_by = partition_by._sort_by
        if window_partition_by is not None:
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


Expr._polars_over = Expr.over  # keep a reference to the real Polars .over() for internal use
Expr.over = over


# ── outer (correlated subquery no-op) ─────────────────────────────────────────

def outer(self: Expr) -> Expr:
    """No-op marker for PySpark correlated-subquery syntax.

    PySpark uses ``col("x").outer()`` to reference an outer-query column inside
    a correlated subquery.  Polars has no correlated-subquery concept at the
    DataFrame API level — rewrite as a semi/anti join instead.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> expr = pl.col("x")
        >>> expr.outer() is expr
        True
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
    """Start a CASE WHEN chain.  Equivalent to sf.when(condition, value).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        >>> df.select(
        ...     pl.col("a").when(pl.col("a") > 3, pl.lit(99)).otherwise(pl.lit(0)).alias("r")
        ... )["r"].to_list()
        [0, 0, 0, 99, 99]
    """
    return SparkWhen(condition, value)


def otherwise_on_expr(self: Expr, value: Any) -> Expr:
    """Return lit(value) for null rows, self for non-null rows.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, None, 3]})
        >>> df.select(pl.col("a").otherwise(-1).alias("r"))["r"].to_list()
        [1, -1, 3]
    """
    return pl.when(self.is_not_null()).then(self).otherwise(pl.lit(value))


Expr.when = when_on_expr
Expr.otherwise = otherwise_on_expr


# ── alias / name ──────────────────────────────────────────────────────────────
# `alias` and `name` already exist natively on Expr; no patching needed.


# ── string methods ────────────────────────────────────────────────────────────

def upper(self: Expr) -> Expr:
    """Convert string column to uppercase.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo"]})
        >>> df.select(pl.col("s").upper())["s"].to_list()
        ['HELLO', 'WORLD', 'FOO']
    """
    return self.str.to_uppercase()


def lower(self: Expr) -> Expr:
    """Convert string column to lowercase.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["HELLO", "World"]})
        >>> df.select(pl.col("s").lower())["s"].to_list()
        ['hello', 'world']
    """
    return self.str.to_lowercase()


def trim(self: Expr, trim_chars: str | None = None) -> Expr:
    """Strip leading and trailing whitespace (or specified characters).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["  hello  ", "  world"]})
        >>> df.select(pl.col("s").trim())["s"].to_list()
        ['hello', 'world']
    """
    return self.str.strip_chars(trim_chars)


def btrim(self: Expr, trim_chars: str | None = None) -> Expr:
    """Strip leading and trailing whitespace (or specified characters). Alias for trim.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["  hello  ", "  world"]})
        >>> df.select(pl.col("s").btrim())["s"].to_list()
        ['hello', 'world']
    """
    return self.str.strip_chars(trim_chars)


def ltrim(self: Expr, trim_chars: str | None = None) -> Expr:
    """Strip leading whitespace (or specified characters).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["  hello  ", "  world"]})
        >>> df.select(pl.col("s").ltrim())["s"].to_list()
        ['hello  ', 'world']
    """
    return self.str.strip_chars_start(trim_chars)


def rtrim(self: Expr, trim_chars: str | None = None) -> Expr:
    """Strip trailing whitespace (or specified characters).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["  hello  ", "  world"]})
        >>> df.select(pl.col("s").rtrim())["s"].to_list()
        ['  hello', '  world']
    """
    return self.str.strip_chars_end(trim_chars)


def lpad(self: Expr, length: int, pad: str = " ") -> Expr:
    """Left-pad the string to the given length with the specified character.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hi"]})
        >>> df.select(pl.col("s").lpad(5, "0"))["s"][0]
        '000hi'
    """
    return self.str.pad_start(length, fill_char=pad)


def rpad(self: Expr, length: int, pad: str = " ") -> Expr:
    """Right-pad the string to the given length with the specified character.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hi"]})
        >>> df.select(pl.col("s").rpad(5, "0"))["s"][0]
        'hi000'
    """
    return self.str.pad_end(length, fill_char=pad)


def left(self: Expr, n: int) -> Expr:
    """Return the leftmost n characters of the string.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo"]})
        >>> df.select(pl.col("s").left(3))["s"].to_list()
        ['hel', 'wor', 'foo']
    """
    return self.str.slice(0, n)


def right(self: Expr, n: int) -> Expr:
    """Return the rightmost n characters of the string.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo"]})
        >>> df.select(pl.col("s").right(3))["s"].to_list()
        ['llo', 'rld', 'foo']
    """
    return self.str.slice(-n, n) if n > 0 else self.str.slice(-n)


def length(self: Expr) -> Expr:
    """Return the number of characters in the string.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo"]})
        >>> df.select(pl.col("s").length())["s"].to_list()
        [5, 5, 3]
    """
    return self.str.len_chars()


def locate(self: Expr, substr: str) -> Expr:
    """Return the 0-based index of the first occurrence of substr.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(pl.col("s").locate("ll"))["s"][0]
        2
    """
    return self.str.find(substr)


def repeat(self: Expr, n: int) -> Expr:
    """Repeat the string n times.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["ab"]})
        >>> df.select(pl.col("s").repeat(3))["s"][0]
        'ababab'
    """
    from functools import reduce
    s = self.cast(pl.String, strict=False)
    return reduce(lambda acc, _: acc + s, range(n - 1), s)


def reverse_str(self: Expr) -> Expr:
    """Reverse the characters in each string.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello", "world", "foo"]})
        >>> df.select(pl.col("s").reverse())["s"].to_list()
        ['olleh', 'dlrow', 'oof']
    """
    return self.str.reverse()


def split_str(self: Expr, pattern: str, limit: int = -1) -> Expr:
    """Split string by pattern into a list. Optionally limit number of splits.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["a,b,c"]})
        >>> df.select(pl.col("s").split(","))["s"][0].to_list()
        ['a', 'b', 'c']
        >>> df.select(pl.col("s").split(",", limit=2))["s"][0].to_list()
        ['a', 'b']
    """
    if limit <= 0:
        return self.str.split(pattern)
    return self.str.split(pattern).list.head(limit)


def base64(self: Expr) -> Expr:
    """Encode a string column to Base64.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(pl.col("s").base64())["s"][0] is not None
        True
    """
    return self.str.encode("base64")


def encode(self: Expr, charset: str) -> Expr:
    return self.str.encode(charset)


def decode(self: Expr, charset: str) -> Expr:
    return self.str.decode(charset)


def regexp_count(self: Expr, pattern: str) -> Expr:
    """Count the number of non-overlapping matches of a regex pattern.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(pl.col("s").regexp_count("l"))["s"][0]
        3
    """
    return self.str.count_matches(pattern)


def regexp_extract(self: Expr, pattern: str, idx: int) -> Expr:
    """Extract the idx-th capture group from the first regex match.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["2023-01-15"]})
        >>> df.select(pl.col("s").regexp_extract(r"(\\d{4})", 1))["s"][0]
        '2023'
    """
    return self.str.extract(pattern, idx)


def regexp_extract_all(self: Expr, pattern: str) -> Expr:
    """Extract all non-overlapping matches of a regex pattern.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(pl.col("s").regexp_extract_all(r"\\w+"))["s"][0].to_list()
        ['hello', 'world']
    """
    return self.str.extract_all(pattern)


def regexp_replace(self: Expr, pattern: str, replacement: str) -> Expr:
    """Replace the first occurrence of a regex pattern.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(pl.col("s").regexp_replace("l", "r"))["s"][0]
        'herlo'
    """
    return self.str.replace(pattern, replacement)


def str_replace(self: Expr, search: str, replacement: str | None = None) -> Expr:
    """Replace the first occurrence of search, or remove all occurrences if no replacement given.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["hello"]})
        >>> df.select(pl.col("s").str_replace("l", "r"))["s"][0]
        'herlo'
        >>> df.select(pl.col("s").str_replace("l"))["s"][0]
        'heo'
    """
    if replacement is None:
        return self.str.replace_all(search, "")
    return self.str.replace(search, replacement)


def translate(self: Expr, matching: str, replace: str) -> Expr:
    """Translate characters using a mapping (like Python str.translate).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": ["abc", "xyz"]})
        >>> df.select(pl.col("s").translate("abc", "xyz"))["s"].to_list()
        ['xyz', 'xyz']
    """
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
    """Extract the year from a date/datetime column.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(pl.col("d").year())["d"][0]
        2023
    """
    return self.dt.year()


def month(self: Expr) -> Expr:
    """Extract the month from a date/datetime column.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(pl.col("d").month())["d"][0]
        1
    """
    return self.dt.month()


def hour(self: Expr) -> Expr:
    """Extract the hour from a datetime column.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 1, 16, 14, 30, 0)]})
        >>> df.select(pl.col("t").hour())["t"][0]
        14
    """
    return self.dt.hour()


def last_day(self: Expr) -> Expr:
    """Return the last day of the month for a date/datetime column.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(pl.col("d").last_day())["d"][0]
        datetime.date(2023, 1, 31)
    """
    return self.dt.month_end()


def dayofmonth(self: Expr) -> Expr:
    """Extract the day of the month from a date/datetime column.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(pl.col("d").dayofmonth())["d"][0]
        16
    """
    return self.dt.day()


def dayofweek(self: Expr) -> Expr:
    """Extract the day of the week (Spark convention: 1=Sunday, 2=Monday, ..., 7=Saturday).

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(pl.col("d").dayofweek())["d"][0]
        2
    """
    return self.dt.weekday() + 1  # Spark: 1=Sunday


def dayofyear(self: Expr) -> Expr:
    """Extract the day of the year from a date/datetime column.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(pl.col("d").dayofyear())["d"][0]
        16
    """
    return self.dt.ordinal_day()


def date_add(self: Expr, days: int) -> Expr:
    """Add a number of days to a date/datetime column.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(pl.col("d").date_add(5))["d"][0]
        datetime.date(2023, 1, 21)
    """
    return self.dt.offset_by(f"{days}d")


def date_sub(self: Expr, days: int) -> Expr:
    """Subtract a number of days from a date/datetime column.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(pl.col("d").date_sub(5))["d"][0]
        datetime.date(2023, 1, 11)
    """
    return self.dt.offset_by(f"-{days}d")


def add_months(self: Expr, months: int) -> Expr:
    """Add a number of months to a date/datetime column.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})
        >>> df.select(pl.col("d").add_months(2))["d"][0]
        datetime.date(2023, 3, 16)
    """
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
    """Compute the MD5 hash of each value as a hex string.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": [""]})
        >>> df.select(pl.col("s").md5())["s"][0]
        'd41d8cd98f00b204e9800998ecf8427e'
    """
    return self.map_elements(
        lambda x: _hashlib.md5(str(x).encode()).hexdigest() if x is not None else None,  # noqa: S324
        _polars_datatypes.String(),
    )


def sha1(self: Expr) -> Expr:
    """Compute the SHA-1 hash of each value as a hex string.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": [""]})
        >>> df.select(pl.col("s").sha1())["s"][0]
        'da39a3ee5e6b4b0d3255bfef95601890afd80709'
    """
    return self.map_elements(
        lambda x: _hashlib.sha1(str(x).encode()).hexdigest() if x is not None else None,  # noqa: S324
        _polars_datatypes.String(),
    )


def sha256(self: Expr) -> Expr:
    """Compute the SHA-256 hash of each value as a hex string.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"s": [""]})
        >>> df.select(pl.col("s").sha256())["s"][0]
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    return self.map_elements(
        lambda x: _hashlib.sha256(str(x).encode()).hexdigest() if x is not None else None,
        _polars_datatypes.String(),
    )


Expr.md5 = md5
Expr.sha1 = sha1
Expr.sha256 = sha256


# ── array / list methods ──────────────────────────────────────────────────────

def array_distinct(self: Expr) -> Expr:
    """Return unique elements in the list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 2, 2, 3]]})
        >>> sorted(df.select(pl.col("a").array_distinct())["a"][0].to_list())
        [1, 2, 3]
    """
    return self.list.unique()


def array_compact(self: Expr) -> Expr:
    """Remove null values from the list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, None, 2]]})
        >>> df.select(pl.col("a").array_compact())["a"][0].to_list()
        [1, 2]
    """
    return self.list.drop_nulls()


def array_contains(self: Expr, value: Any) -> Expr:
    """Check if the list contains the given value.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(pl.col("a").array_contains(2))["a"][0]
        True
        >>> df.select(pl.col("a").array_contains(99))["a"][0]
        False
    """
    v = pl.lit(value) if isinstance(value, str) else value
    return self.list.contains(v)


def array_append(self: Expr, value: Any) -> Expr:
    """Append a value to the end of the list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 2]]})
        >>> df.select(pl.col("a").array_append(3))["a"][0].to_list()
        [1, 2, 3]
    """
    v = pl.lit(value) if isinstance(value, str) else value
    return self.list.concat(v)


def array_remove(self: Expr, element: Any) -> Expr:
    """Remove all occurrences of element from the list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 2, 1, 3]]})
        >>> df.select(pl.col("a").array_remove(1))["a"][0].to_list()
        [2, 3]
    """
    e = pl.lit(element) if isinstance(element, str) else element
    return self.list.eval(pl.element().filter(pl.element() != e))


def array_union(self: Expr, other: Expr) -> Expr:
    """Return the union of two list columns.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
        >>> sorted(df.select(pl.col("a").array_union(pl.col("b")))["a"][0].to_list())
        [1, 2, 3, 4]
    """
    return self.list.set_union(other)


def array_intersect(self: Expr, other: Expr) -> Expr:
    """Return the intersection of two list columns.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
        >>> sorted(df.select(pl.col("a").array_intersect(pl.col("b")))["a"][0].to_list())
        [2, 3]
    """
    return self.list.set_intersection(other)


def array_except(self: Expr, other: Expr) -> Expr:
    """Return elements in self that are not in other.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3]]})
        >>> df.select(pl.col("a").array_except(pl.col("b")))["a"][0].to_list()
        [1]
    """
    return self.list.set_difference(other)


def array_join(self: Expr, delimiter: str, null_replacement: str | None = None) -> Expr:
    """Join list elements into a string with a delimiter.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [["a", "b", "c"]]})
        >>> df.select(pl.col("a").array_join(","))["a"][0]
        'a,b,c'
        >>> df2 = pl.DataFrame({"a": [["a", None, "c"]]})
        >>> df2.select(pl.col("a").array_join(",", "X"))["a"][0]
        'a,X,c'
    """
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
    """Return the maximum value in the list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 3, 2]]})
        >>> df.select(pl.col("a").array_max())["a"][0]
        3
    """
    return self.list.max()


def array_min(self: Expr) -> Expr:
    """Return the minimum value in the list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 3, 2]]})
        >>> df.select(pl.col("a").array_min())["a"][0]
        1
    """
    return self.list.min()


def array_size(self: Expr) -> Expr:
    """Return the number of elements in the list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[3, 1, 2, 1], [4, 5, 6]]})
        >>> df.select(pl.col("a").array_size())["a"].to_list()
        [4, 3]
    """
    return self.list.len()


def array_sort(self: Expr, asc: bool = True) -> Expr:
    """Sort the elements in the list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[3, 1, 2]]})
        >>> df.select(pl.col("a").array_sort())["a"][0].to_list()
        [1, 2, 3]
        >>> df.select(pl.col("a").array_sort(asc=False))["a"][0].to_list()
        [3, 2, 1]
    """
    return self.list.sort(descending=not asc, nulls_last=False)


def array_slice(self: Expr, start: int, length: int | None = None) -> Expr:
    """Slice the list starting at 1-based position with optional length.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
        >>> df.select(pl.col("a").array_slice(2, 2))["a"][0].to_list()
        [2, 3]
    """
    # Spark uses 1-based indexing
    start_idx = start - 1 if start > 0 else start
    if length is not None:
        return self.list.slice(start_idx, length)
    return self.list.slice(start_idx)






def list_filter(self: Expr, f: Callable) -> Expr:
    """Filter list elements using a predicate function.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
        >>> df.select(pl.col("a").list_filter(lambda e: e > 2))["a"][0].to_list()
        [3, 4]
    """
    return self.list.eval(pl.element().filter(f(pl.element())))


def forall(self: Expr, f: Callable) -> Expr:
    """Return True if all elements in the list satisfy the predicate.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[2, 4, 6]]})
        >>> df.select(pl.col("a").forall(lambda e: e > 0))["a"][0]
        True
        >>> df2 = pl.DataFrame({"a": [[2, -1, 6]]})
        >>> df2.select(pl.col("a").forall(lambda e: e > 0))["a"][0]
        False
    """
    return self.list.eval(f(pl.element())).list.all()


def collect_list(self: Expr) -> Expr:
    """Aggregate all values into a single list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> df.select(pl.col("x").collect_list())["x"][0].to_list()
        [1, 2, 3]
    """
    return self.implode()


def collect_set(self: Expr) -> Expr:
    """Aggregate unique values into a single list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3]})
        >>> sorted(df.select(pl.col("x").collect_set())["x"][0].to_list())
        [1, 2, 3]
    """
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
    """Negate the values in the column.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, -2, 3]})
        >>> df.select(pl.col("x").negate())["x"].to_list()
        [-1, 2, -3]
    """
    return -self


def avg(self: Expr) -> Expr:
    """Compute the mean (average) of the column.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.select(pl.col("x").avg())["x"][0]
        3.0
    """
    return self.mean()


Expr.negate = negate
Expr.avg = avg


# ── math Expr methods ─────────────────────────────────────────────────────────
# ceil, floor, abs, round, sqrt, pow, product, log, exp, sign — already native.
# We only expose the missing aliases.

def log2(self: Expr) -> Expr:
    """Compute the base-2 logarithm.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [8.0]})
        >>> df.select(pl.col("x").log2())["x"][0]
        3.0
    """
    return self.log(2.0)


def log10(self: Expr) -> Expr:
    """Compute the base-10 logarithm.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [100.0]})
        >>> df.select(pl.col("x").log10())["x"][0]
        2.0
    """
    return self.log(10.0)


def cbrt(self: Expr) -> Expr:
    """Compute the cube root.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [8.0, 27.0]})
        >>> df.select(pl.col("x").cbrt())["x"].to_list()
        [2.0, 3.0]
    """
    return self ** (1.0 / 3.0)


def signum(self: Expr) -> Expr:
    """Return the sign of each value (-1, 0, or 1).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [3.0, -2.0, 0.0]})
        >>> df.select(pl.col("x").signum())["x"].to_list()
        [1.0, -1.0, 0.0]
    """
    return self.sign()


Expr.log2 = log2
Expr.log10 = log10
Expr.cbrt = cbrt
Expr.signum = signum


# ── null-handling Expr methods ────────────────────────────────────────────────

def nvl(self: Expr, replacement: Any) -> Expr:
    """Replace null values with a replacement value (coalesce).

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [None, 2, None]})
        >>> df.select(pl.col("a").nvl(-1))["a"].to_list()
        [-1, 2, -1]
    """
    r = replacement if isinstance(replacement, Expr) else pl.lit(replacement)
    return polars_functions.coalesce(self, r)


def nullif(self: Expr, value: Any) -> Expr:
    """Return null if the value equals the given value, otherwise return the value.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").nullif(2))["a"].to_list()
        [1, None, 3]
    """
    v = value if isinstance(value, Expr) else pl.lit(value)
    return polars_functions.when(self != v).then(self)


Expr.nvl = nvl
Expr.ifnull = nvl
Expr.nullif = nullif


# ── date / time Expr methods ──────────────────────────────────────────────────

def to_date(self: Expr, fmt: str = "%Y-%m-%d") -> Expr:
    """Parse a string column to a date using the given format.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"s": ["2023-06-15"]})
        >>> df.select(pl.col("s").to_date())["s"][0]
        datetime.date(2023, 6, 15)
    """
    return self.str.to_date(fmt)


def to_timestamp(self: Expr, fmt: str = "%Y-%m-%d %H:%M:%S") -> Expr:
    """Parse a string column to a datetime using the given format.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"s": ["2023-06-15 10:30:00"]})
        >>> df.select(pl.col("s").to_timestamp())["s"][0]
        datetime.datetime(2023, 6, 15, 10, 30)
    """
    return self.str.to_datetime(fmt)


def date_format(self: Expr, fmt: str) -> Expr:
    """Format a date/datetime column as a string.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
        >>> df.select(pl.col("d").date_format("%Y/%m/%d"))["d"][0]
        '2023/06/15'
    """
    return self.dt.strftime(fmt)


def unix_timestamp(self: Expr) -> Expr:
    """Convert a datetime column to Unix epoch seconds.

    Examples:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 6, 15, 0, 0, 0)]})
        >>> df.select(pl.col("t").unix_timestamp())["t"][0]
        1686787200
    """
    return self.dt.epoch(time_unit="s")


Expr.to_date = to_date
Expr.to_timestamp = to_timestamp
Expr.date_format = date_format
Expr.unix_timestamp = unix_timestamp


# ── array Expr methods ────────────────────────────────────────────────────────

def array_position(self: Expr, value: Any) -> Expr:
    """1-based index of value in the array, 0 if not found.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[10, 20, 30]]})
        >>> df.select(pl.col("a").array_position(20))["a"][0]
        2
        >>> df.select(pl.col("a").array_position(99))["a"][0]
        0
    """
    v = value if isinstance(value, Expr) else pl.lit(value)
    idx = self.list.eval(pl.arg_where(pl.element() == v).first()).list.first()
    return polars_functions.when(self.list.contains(v)).then(idx + 1).otherwise(pl.lit(0))


def array_prepend(self: Expr, value: Any) -> Expr:
    """Prepend a value to the beginning of the list.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [[10, 20, 30]]})
        >>> df.select(pl.col("a").array_prepend(5))["a"][0].to_list()
        [5, 10, 20, 30]
    """
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
    """Convert string to title case (capitalize first letter of each word).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(pl.col("s").initcap())["s"][0]
        'Hello World'
    """
    return self.str.to_titlecase()


def ascii_code(self: Expr) -> Expr:
    """Return the ASCII code of the first character of each string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["A"]})
        >>> df.select(pl.col("s").ascii_code())["s"][0]
        65
    """
    return self.map_elements(
        lambda x: ord(x[0]) if x else None,
        _polars_datatypes.Int32(),
    )


def instr(self: Expr, substr: str) -> Expr:
    """Return 1-based position of substr in string; 0 if not found.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(pl.col("s").instr("world"))["s"][0]
        7
    """
    idx = self.str.find(substr, literal=True)
    return polars_functions.when(idx.is_not_null()).then(idx + 1).otherwise(pl.lit(0))


def split_part(self: Expr, delimiter: str, part_num: int) -> Expr:
    """Split by delimiter and return part_num-th part (1-based); '' if out of range.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["a,b,c"]})
        >>> df.select(pl.col("s").split_part(",", 2))["s"][0]
        'b'
    """
    return self.str.split(delimiter).list.get(part_num - 1, null_on_oob=True).fill_null("")


def substring_index(self: Expr, delimiter: str, count: int) -> Expr:
    """Return substring before the count-th occurrence of delimiter.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["a.b.c.d"]})
        >>> df.select(pl.col("s").substring_index(".", 2))["s"][0]
        'a.b'
    """
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
    """Extract the quarter (1-4) from a date/datetime column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 4, 15)]})
        >>> df.select(pl.col("d").quarter())["d"][0]
        2
    """
    return ((self.dt.month() - 1) // 3 + 1).cast(_polars_datatypes.Int32())


def minute(self: Expr) -> Expr:
    """Extract the minute from a datetime column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 1, 1, 14, 35, 0)]})
        >>> df.select(pl.col("t").minute())["t"][0]
        35
    """
    return self.dt.minute()


def second(self: Expr) -> Expr:
    """Extract the second from a datetime column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import datetime
        >>> df = pl.DataFrame({"t": [datetime.datetime(2023, 1, 1, 14, 35, 45)]})
        >>> df.select(pl.col("t").second())["t"][0]
        45
    """
    return self.dt.second()


def weekofyear(self: Expr) -> Expr:
    """Extract the ISO week number of the year.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 9)]})
        >>> df.select(pl.col("d").weekofyear())["d"][0]
        2
    """
    return self.dt.week()


def weekday(self: Expr) -> Expr:
    """Day of week: 0=Monday, 6=Sunday (same as PySpark weekday()).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 9)]})
        >>> df.select(pl.col("d").weekday())["d"][0]
        0
    """
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
    """Truncate date/timestamp to the given Spark unit string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
        >>> df.select(pl.col("d").date_trunc("month"))["d"][0]
        datetime.date(2023, 6, 1)
    """
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
    """Compute the sample standard deviation.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> round(df.select(pl.col("x").stddev())["x"][0], 6)
        1.0
    """
    return self.std()


def variance(self: Expr) -> Expr:
    """Compute the sample variance.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> round(df.select(pl.col("x").variance())["x"][0], 6)
        1.0
    """
    return self.var()


def count_distinct(self: Expr) -> Expr:
    """Count the number of distinct values.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3, 3]})
        >>> df.select(pl.col("x").count_distinct())["x"][0]
        3
    """
    return self.n_unique()


def bool_and(self: Expr) -> Expr:
    """Return True if all values are True (boolean AND aggregate).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [True, True, True]})
        >>> df.select(pl.col("x").bool_and())["x"][0]
        True
    """
    return self.all()


def bool_or(self: Expr) -> Expr:
    """Return True if any value is True (boolean OR aggregate).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [False, False, True]})
        >>> df.select(pl.col("x").bool_or())["x"][0]
        True
    """
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
    """Return replacement when col is NaN, else col.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [float("nan"), 2.0, float("nan")]})
        >>> df.select(pl.col("x").nanvl(-1.0))["x"].to_list()
        [-1.0, 2.0, -1.0]
    """
    r = replacement if isinstance(replacement, Expr) else pl.lit(replacement)
    result = polars_functions.when(self.is_nan()).then(r).otherwise(self)
    try:
        return result.alias(self.meta.output_name())
    except Exception:
        return result


def nvl2(self: Expr, not_null_val: Any, null_val: Any) -> Expr:
    """Return not_null_val when col is not null, null_val when col is null.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [1, None, 3]})
        >>> df.select(pl.col("a").nvl2(100, 0))["a"].to_list()
        [100, 0, 100]
    """
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
    """Compute bitwise NOT (inversion) of integer values.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [5]})
        >>> df.select(pl.col("x").bitwiseNOT())["x"][0]
        -6
    """
    return ~self


def shiftLeft(self: Expr, n: int) -> Expr:
    """Shift integer values left by n bits.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [1]})
        >>> df.select(pl.col("x").shiftLeft(3))["x"][0]
        8
    """
    return self.map_elements(
        lambda x: x << n if x is not None else None, _polars_datatypes.Int64()
    )


def shiftRight(self: Expr, n: int) -> Expr:
    """Shift integer values right by n bits.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [8]})
        >>> df.select(pl.col("x").shiftRight(2))["x"][0]
        2
    """
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
    """Return element at 1-based index (negative counts from end). Returns null if OOB.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [[10, 20, 30]]})
        >>> df.select(pl.col("a").element_at(2))["a"][0]
        20
    """
    idx = index - 1 if index > 0 else index
    return self.list.get(idx, null_on_oob=True)


def arrays_overlap(self: Expr, other: Expr) -> Expr:
    """Return true if the arrays share at least one common element.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[3, 4, 5]]})
        >>> df.select(pl.col("a").arrays_overlap(pl.col("b")))["a"][0]
        True
    """
    return self.list.set_intersection(other).list.len() > 0


Expr.element_at = element_at
Expr.arrays_overlap = arrays_overlap


# ── window helpers ─────────────────────────────────────────────────────────────

def lag(self: Expr, offset: int = 1, default: Any = None) -> Expr:
    """Shift values forward by offset rows within the current partition.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> from src.sparkpolars.polyspark.sql.window import WindowSpec
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4]})
        >>> df.select(pl.col("x").lag(1).over(WindowSpec()).alias("l"))["l"].to_list()
        [None, 1, 2, 3]
    """
    if default is not None:
        return self.shift(offset, fill_value=default)
    return self.shift(offset)


def lead(self: Expr, offset: int = 1, default: Any = None) -> Expr:
    """Shift values backward by offset rows within the current partition.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> from src.sparkpolars.polyspark.sql.window import WindowSpec
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4]})
        >>> df.select(pl.col("x").lead(1).over(WindowSpec()).alias("l"))["l"].to_list()
        [2, 3, 4, None]
    """
    if default is not None:
        return self.shift(-offset, fill_value=default)
    return self.shift(-offset)


Expr.lag = lag
Expr.lead = lead


# ── string extras (batch 2) ───────────────────────────────────────────────────

def chr_char(self: Expr) -> Expr:
    """Convert Unicode code point integer to a single character string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [65, 66, 67]})
        >>> df.select(pl.col("x").chr())["x"].to_list()
        ['A', 'B', 'C']
    """
    return self.map_elements(
        lambda x: chr(int(x)) if x is not None else None,
        _polars_datatypes.String(),
    )


def find_in_set(self: Expr, str_val: str) -> Expr:
    """Return 1-based position of str_val in comma-delimited self; 0 if not found.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["a,b,c,d"]})
        >>> df.select(pl.col("s").find_in_set("c"))["s"][0]
        3
    """
    parts = self.str.split(",")
    idx = parts.list.eval(pl.arg_where(pl.element() == str_val).first()).list.first()
    return polars_functions.when(parts.list.contains(str_val)).then(idx + 1).otherwise(pl.lit(0))


def regexp_like(self: Expr, pattern: str) -> Expr:
    """Return true if the string matches the regex pattern.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["hello123", "world"]})
        >>> df.select(pl.col("s").regexp_like(r"\\d+"))["s"].to_list()
        [True, False]
    """
    return self.str.contains(pattern, literal=False)


def overlay(self: Expr, replace: Any, pos: int, length: int | None = None) -> Expr:
    """Replace portion of self starting at 1-based pos with replace string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["hello world"]})
        >>> df.select(pl.col("s").overlay("there", 7))["s"][0]
        'hello there'
    """
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
    """Banker's rounding (half-to-even). Polars round() already uses HALF_TO_EVEN.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [2.5, 3.5]})
        >>> df.select(pl.col("x").bround())["x"].to_list()
        [2.0, 4.0]
    """
    return self.round(d)


def pmod(self: Expr, divisor: Any) -> Expr:
    """Positive modulo -- result always has the same sign as the divisor.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [-7]})
        >>> df.select(pl.col("x").pmod(3))["x"][0]
        2
    """
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
    """Return the first date strictly after self that is the named day of week.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(2023, 1, 11)]})
        >>> df.select(pl.col("d").next_day("Mon"))["d"][0]
        datetime.date(2023, 1, 16)
    """
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
    """Number of days since Unix epoch (1970-01-01).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import datetime
        >>> df = pl.DataFrame({"d": [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2)]})
        >>> df.select(pl.col("d").unix_date())["d"].to_list()
        [0, 1]
    """
    return self.dt.epoch(time_unit="d")


Expr.next_day = next_day
Expr.unix_date = unix_date


# ── array extras (batch 2) ────────────────────────────────────────────────────

def exists(self: Expr, f: Callable) -> Expr:
    """Return true if any element in the list satisfies predicate f.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(pl.col("a").exists(lambda e: e > 2))["a"][0]
        True
    """
    return self.list.eval(f(pl.element())).list.any()


def shuffle(self: Expr, seed: int | None = None) -> Expr:
    """Return the list with elements in a random order.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4, 5]]})
        >>> sorted(df.select(pl.col("a").shuffle(seed=0))["a"][0].to_list())
        [1, 2, 3, 4, 5]
    """
    return self.list.sample(n=self.list.len(), with_replacement=False, shuffle=True, seed=seed)


def aggregate_list(self: Expr, zero: Any, merge: Callable, finish: Callable | None = None) -> Expr:
    """Fold over list: reduce(merge, elements, zero), then optionally apply finish.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
        >>> df.select(pl.col("a").aggregate(0, lambda acc, e: acc + e))["a"][0]
        10
    """
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
    """Return the pct-th percentile (0.0-1.0 scale).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        >>> df.select(pl.col("x").percentile(0.5))["x"][0]
        3.0
    """
    return self.quantile(pct, interpolation="nearest")


def sum_distinct(self: Expr) -> Expr:
    """Sum of distinct (unique) values.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3]})
        >>> df.select(pl.col("x").sum_distinct())["x"][0]
        6
    """
    return self.unique().sum()


def approx_count_distinct(self: Expr, rsd: float = 0.05) -> Expr:  # noqa: ARG001
    """Approximate distinct count (exact in Polars).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [1, 2, 2, 3, 3, 3]})
        >>> df.select(pl.col("x").approx_count_distinct())["x"][0]
        3
    """
    return self.n_unique()


Expr.percentile = percentile
Expr.sum_distinct = sum_distinct
Expr.approx_count_distinct = approx_count_distinct


# ── string extras (batch 3) ───────────────────────────────────────────────────

def unbase64(self: Expr) -> Expr:
    """Decode a Base64-encoded string column to a UTF-8 string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import base64
        >>> encoded = base64.b64encode(b"hello").decode()
        >>> df = pl.DataFrame({"s": [encoded]})
        >>> df.select(pl.col("s").unbase64())["s"][0]
        'hello'
    """
    import base64 as _b64

    return self.map_elements(
        lambda x: _b64.b64decode(x).decode("utf-8") if x is not None else None,
        _polars_datatypes.String(),
    )


def regexp_substr(self: Expr, pattern: str) -> Expr:
    """Return the first substring matching the regexp, or null if no match.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"s": ["hello world 123"]})
        >>> df.select(pl.col("s").regexp_substr(r"\\d+"))["s"][0]
        '123'
    """
    return self.str.extract(pattern, 0)


def levenshtein(self: Expr, other: Any) -> Expr:
    """Compute the Levenshtein edit distance between two string columns.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": ["kitten"], "b": ["sitting"]})
        >>> df.select(pl.col("a").levenshtein(pl.col("b")).alias("d"))["d"][0]
        3
    """
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
    """Natural log of (1 + x).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(pl.col("x").log1p())["x"][0]
        0.0
    """
    return (self + 1.0).log(_math.e)


def expm1(self: Expr) -> Expr:
    """e^x - 1.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [0.0]})
        >>> df.select(pl.col("x").expm1())["x"][0]
        0.0
    """
    return self.exp() - 1.0


def rint(self: Expr) -> Expr:
    """Round to nearest integer (returns float).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [1.4, 1.6, -1.5]})
        >>> df.select(pl.col("x").rint())["x"].to_list()
        [1.0, 2.0, -2.0]
    """
    return self.round(0)


def bitcount(self: Expr) -> Expr:
    """Count the number of set bits (1s) in the binary representation.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"x": [7]})
        >>> df.select(pl.col("x").bitcount())["x"][0]
        3
    """
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
    """Reverse the elements of an array column.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"a": [[1, 2, 3]]})
        >>> df.select(pl.col("a").array_reverse())["a"][0].to_list()
        [3, 2, 1]
    """
    return self.list.reverse()


Expr.array_reverse = array_reverse


# ── struct / map extras (batch 3) ─────────────────────────────────────────────

def to_json(self: Expr) -> Expr:
    """Serialize a struct column to a JSON string.

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> import json
        >>> df = pl.DataFrame({"s": [{"a": 1, "b": 2}]})
        >>> json.loads(df.select(pl.col("s").to_json())["s"][0]) == {"a": 1, "b": 2}
        True
    """
    return self.struct.json_encode()


def map_keys(self: Expr) -> Expr:
    """Return the keys of a map column (stored as list-of-structs with 'key' field).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]})
        >>> sorted(df.select(pl.col("m").map_keys())["m"][0].to_list())
        ['a', 'b']
    """
    return self.list.eval(pl.element().struct.field("key"))


def map_values(self: Expr) -> Expr:
    """Return the values of a map column (stored as list-of-structs with 'value' field).

    Examples:
        >>> import polars as pl
        >>> import src.sparkpolars.polyspark.sql.columns  # noqa: F401
        >>> df = pl.DataFrame({"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]})
        >>> sorted(df.select(pl.col("m").map_values())["m"][0].to_list())
        ['1', '2']
    """
    return self.list.eval(pl.element().struct.field("value"))


Expr.to_json = to_json
Expr.map_keys = map_keys
Expr.map_values = map_values
