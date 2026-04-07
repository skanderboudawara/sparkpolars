"""Tests for polyspark standalone functions (each delegates to an Expr method)."""

import datetime
import hashlib

import polars as pl
import pytest

import src.sparkpolars.polyspark.sql.functions as sf  # noqa: F401


# ── string functions ──────────────────────────────────────────────────────────

def test_sf_upper():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.upper(pl.col("s")))["s"][0] == "HELLO"


def test_sf_lower():
    df = pl.DataFrame({"s": ["HELLO"]})
    assert df.select(sf.lower(pl.col("s")))["s"][0] == "hello"


def test_sf_trim():
    df = pl.DataFrame({"s": ["  hi  "]})
    assert df.select(sf.trim(pl.col("s")))["s"][0] == "hi"


def test_sf_ltrim():
    df = pl.DataFrame({"s": ["  hi  "]})
    assert df.select(sf.ltrim(pl.col("s")))["s"][0] == "hi  "


def test_sf_rtrim():
    df = pl.DataFrame({"s": ["  hi  "]})
    assert df.select(sf.rtrim(pl.col("s")))["s"][0] == "  hi"


def test_sf_btrim():
    df = pl.DataFrame({"s": ["  hi  "]})
    assert df.select(sf.btrim(pl.col("s")))["s"][0] == "hi"


def test_sf_lpad():
    df = pl.DataFrame({"s": ["hi"]})
    assert df.select(sf.lpad(pl.col("s"), 5, "0"))["s"][0] == "000hi"


def test_sf_rpad():
    df = pl.DataFrame({"s": ["hi"]})
    assert df.select(sf.rpad(pl.col("s"), 5, "0"))["s"][0] == "hi000"


def test_sf_left():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.left(pl.col("s"), 3))["s"][0] == "hel"


def test_sf_right():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.right(pl.col("s"), 3))["s"][0] == "llo"


def test_sf_length():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.length(pl.col("s")))["s"][0] == 5


def test_sf_locate():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.locate("ll", pl.col("s")))["s"][0] == 2


def test_sf_repeat():
    df = pl.DataFrame({"s": ["ab"]})
    assert df.select(sf.repeat(pl.col("s"), 3))["s"][0] == "ababab"


def test_sf_reverse():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.reverse(pl.col("s")))["s"][0] == "olleh"


def test_sf_split():
    df = pl.DataFrame({"s": ["a,b,c"]})
    assert df.select(sf.split(pl.col("s"), ","))["s"][0].to_list() == ["a", "b", "c"]


def test_sf_regexp_count():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.regexp_count(pl.col("s"), "l"))["s"][0] == 2


def test_sf_regexp_extract():
    df = pl.DataFrame({"s": ["2023-01-15"]})
    assert df.select(sf.regexp_extract(pl.col("s"), r"(\d{4})", 1))["s"][0] == "2023"


def test_sf_regexp_extract_all():
    df = pl.DataFrame({"s": ["hello world"]})
    assert df.select(sf.regexp_extract_all(pl.col("s"), r"\w+"))["s"][0].to_list() == ["hello", "world"]


def test_sf_regexp_replace():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.regexp_replace(pl.col("s"), "l", "r"))["s"][0] == "herlo"


def test_sf_replace():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.replace(pl.col("s"), "l", "r"))["s"][0] == "herlo"


def test_sf_replace_delete():
    df = pl.DataFrame({"s": ["hello"]})
    # no replacement → delete all occurrences of search
    assert df.select(sf.replace(pl.col("s"), "l"))["s"][0] == "heo"


def test_sf_translate():
    df = pl.DataFrame({"s": ["abc"]})
    assert df.select(sf.translate(pl.col("s"), "abc", "xyz"))["s"][0] == "xyz"


def test_sf_ucase():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.ucase(pl.col("s")))["s"][0] == "HELLO"


def test_sf_lcase():
    df = pl.DataFrame({"s": ["HELLO"]})
    assert df.select(sf.lcase(pl.col("s")))["s"][0] == "hello"


def test_sf_contains():
    df = pl.DataFrame({"s": ["hello world"]})
    assert df.filter(sf.contains(pl.col("s"), "world")).height == 1


def test_sf_concat_ws():
    df = pl.DataFrame({"a": ["x"], "b": ["y"]})
    result = df.select(sf.concat_ws("-", pl.col("a"), pl.col("b")))["a"][0]
    assert result == "x-y"


def test_sf_base64():
    df = pl.DataFrame({"s": ["hello"]})
    result = df.select(sf.base64(pl.col("s")))["s"][0]
    assert result is not None  # returns binary


# ── date / time functions ─────────────────────────────────────────────────────

@pytest.fixture()
def date_df():
    return pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})


@pytest.fixture()
def datetime_df():
    return pl.DataFrame({"t": [datetime.datetime(2023, 1, 16, 14, 30, 0)]})


def test_sf_year(date_df):
    assert date_df.select(sf.year(pl.col("d")))["d"][0] == 2023


def test_sf_month(date_df):
    assert date_df.select(sf.month(pl.col("d")))["d"][0] == 1


def test_sf_dayofmonth(date_df):
    assert date_df.select(sf.dayofmonth(pl.col("d")))["d"][0] == 16


def test_sf_dayofweek(date_df):
    assert date_df.select(sf.dayofweek(pl.col("d")))["d"][0] == 2  # Monday


def test_sf_dayofyear(date_df):
    assert date_df.select(sf.dayofyear(pl.col("d")))["d"][0] == 16


def test_sf_hour(datetime_df):
    assert datetime_df.select(sf.hour(pl.col("t")))["t"][0] == 14


def test_sf_last_day(date_df):
    assert date_df.select(sf.last_day(pl.col("d")))["d"][0] == datetime.date(2023, 1, 31)


def test_sf_date_add(date_df):
    assert date_df.select(sf.date_add(pl.col("d"), 5))["d"][0] == datetime.date(2023, 1, 21)


def test_sf_date_sub(date_df):
    assert date_df.select(sf.date_sub(pl.col("d"), 5))["d"][0] == datetime.date(2023, 1, 11)


def test_sf_add_months(date_df):
    assert date_df.select(sf.add_months(pl.col("d"), 2))["d"][0] == datetime.date(2023, 3, 16)


# ── hash functions ────────────────────────────────────────────────────────────

def test_sf_md5():
    df = pl.DataFrame({"s": ["hello"]})
    expected = hashlib.md5(b"hello").hexdigest()
    assert df.select(sf.md5(pl.col("s")))["s"][0] == expected


def test_sf_sha1():
    df = pl.DataFrame({"s": ["hello"]})
    expected = hashlib.sha1(b"hello").hexdigest()
    assert df.select(sf.sha1(pl.col("s")))["s"][0] == expected


def test_sf_sha256():
    df = pl.DataFrame({"s": ["hello"]})
    expected = hashlib.sha256(b"hello").hexdigest()
    assert df.select(sf.sha256(pl.col("s")))["s"][0] == expected


# ── array functions ───────────────────────────────────────────────────────────

def test_sf_array_distinct():
    df = pl.DataFrame({"a": [[1, 2, 2, 3]]})
    assert sorted(df.select(sf.array_distinct(pl.col("a")))["a"][0].to_list()) == [1, 2, 3]


def test_sf_array_compact():
    df = pl.DataFrame({"a": [[1, None, 2]]})
    assert df.select(sf.array_compact(pl.col("a")))["a"][0].to_list() == [1, 2]


def test_sf_array_contains():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    assert df.select(sf.array_contains(pl.col("a"), 2))["a"][0] is True


def test_sf_array_append():
    df = pl.DataFrame({"a": [[1, 2]]})
    assert df.select(sf.array_append(pl.col("a"), 3))["a"][0].to_list() == [1, 2, 3]


def test_sf_array_remove():
    df = pl.DataFrame({"a": [[1, 2, 1, 3]]})
    assert df.select(sf.array_remove(pl.col("a"), 1))["a"][0].to_list() == [2, 3]


def test_sf_array_max():
    df = pl.DataFrame({"a": [[1, 3, 2]]})
    assert df.select(sf.array_max(pl.col("a")))["a"][0] == 3


def test_sf_array_min():
    df = pl.DataFrame({"a": [[1, 3, 2]]})
    assert df.select(sf.array_min(pl.col("a")))["a"][0] == 1


def test_sf_array_size():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    assert df.select(sf.array_size(pl.col("a")))["a"][0] == 3


def test_sf_size():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    assert df.select(sf.size(pl.col("a")))["a"][0] == 3


def test_sf_array_sort():
    df = pl.DataFrame({"a": [[3, 1, 2]]})
    assert df.select(sf.array_sort(pl.col("a")))["a"][0].to_list() == [1, 2, 3]


def test_sf_sort_array_desc():
    df = pl.DataFrame({"a": [[3, 1, 2]]})
    assert df.select(sf.sort_array(pl.col("a"), asc=False))["a"][0].to_list() == [3, 2, 1]


def test_sf_slice():
    df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
    assert df.select(sf.slice(pl.col("a"), 2, 2))["a"][0].to_list() == [2, 3]


def test_sf_array_join():
    df = pl.DataFrame({"a": [["a", "b", "c"]]})
    assert df.select(sf.array_join(pl.col("a"), ","))["a"][0] == "a,b,c"


def test_sf_array_union():
    df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
    result = sorted(df.select(sf.array_union(pl.col("a"), pl.col("b")))["a"][0].to_list())
    assert result == [1, 2, 3, 4]


def test_sf_array_intersect():
    df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
    result = sorted(df.select(sf.array_intersect(pl.col("a"), pl.col("b")))["a"][0].to_list())
    assert result == [2, 3]


def test_sf_array_except():
    df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3]]})
    result = df.select(sf.array_except(pl.col("a"), pl.col("b")))["a"][0].to_list()
    assert result == [1]


def test_sf_transform():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    assert df.select(sf.transform(pl.col("a"), lambda e: e * 2))["a"][0].to_list() == [2, 4, 6]


def test_sf_filter():
    df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
    assert df.select(sf.filter(pl.col("a"), lambda e: e > 2))["a"][0].to_list() == [3, 4]


def test_sf_forall():
    df = pl.DataFrame({"a": [[2, 4, 6]]})
    assert df.select(sf.forall(pl.col("a"), lambda e: e > 0))["a"][0] is True


# ── math / aggregation functions ──────────────────────────────────────────────

def test_sf_negate():
    df = pl.DataFrame({"x": [1, -2, 3]})
    assert df.select(sf.negate(pl.col("x")))["x"].to_list() == [-1, 2, -3]


def test_sf_negative():
    df = pl.DataFrame({"x": [1, -2, 3]})
    assert df.select(sf.negative(pl.col("x")))["x"].to_list() == [-1, 2, -3]


def test_sf_avg():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    # standalone avg aliases the result column to "avg"
    assert df.select(sf.avg(pl.col("x")))["avg"][0] == 3.0


def test_sf_collect_list():
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert df.select(sf.collect_list(pl.col("x")))["x"][0].to_list() == [1, 2, 3]


def test_sf_collect_set():
    df = pl.DataFrame({"x": [1, 2, 2, 3]})
    result = sorted(df.select(sf.collect_set(pl.col("x")))["x"][0].to_list())
    assert result == [1, 2, 3]


def test_sf_greatest():
    df = pl.DataFrame({"a": [1, 5], "b": [3, 2]})
    assert df.select(sf.greatest(pl.col("a"), pl.col("b")))["a"].to_list() == [3, 5]


def test_sf_least():
    df = pl.DataFrame({"a": [1, 5], "b": [3, 2]})
    assert df.select(sf.least(pl.col("a"), pl.col("b")))["a"].to_list() == [1, 2]


def test_sf_monotonically_increasing_id():
    df = pl.DataFrame({"x": [10, 20, 30]})
    result = df.select(sf.monotonically_increasing_id().alias("id"))["id"].to_list()
    assert result == [0, 1, 2]


def test_sf_coalesce():
    df = pl.DataFrame({"a": [None, 2, None], "b": [1, None, 3]})
    result = df.select(sf.coalesce(pl.col("a"), pl.col("b")))["a"].to_list()
    assert result == [1, 2, 3]


# ── misc functions ────────────────────────────────────────────────────────────

def test_sf_when():
    df = pl.DataFrame({"x": [1, 2, 3]})
    result = df.select(sf.when(pl.col("x") > 1, pl.lit(99)).otherwise(pl.lit(0)).alias("r"))
    assert result["r"].to_list() == [0, 99, 99]


def test_sf_expr():
    df = pl.DataFrame({"x": [1, 2, 3]})
    result = df.select(sf.expr("x * 2").alias("r"))
    assert result["r"].to_list() == [2, 4, 6]


def test_sf_broadcast():
    df = pl.DataFrame({"x": [1]})
    assert sf.broadcast(df) is df


def test_sf_array():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.select(sf.array(pl.col("a"), pl.col("b")).alias("arr"))
    assert result["arr"][0].to_list() == [1, 3]


def test_sf_sequence():
    df = pl.DataFrame({"start": [1], "stop": [5]})
    result = df.select(sf.sequence(pl.col("start"), pl.col("stop")).alias("s"))
    assert result["s"][0].to_list() == [1, 2, 3, 4, 5]


# ── new standalone functions ──────────────────────────────────────────────────

def test_sf_substr():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.substr(pl.col("s"), 2, 3))["s"][0] == "ell"


def test_sf_substring():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.substring(pl.col("s"), 1, 3))["s"][0] == "hel"


def test_sf_struct():
    df = pl.DataFrame({"a": [1], "b": [2]})
    result = df.select(sf.struct(pl.col("a"), pl.col("b")).alias("s"))["s"][0]
    assert result == {"a": 1, "b": 2}


# ── math functions ────────────────────────────────────────────────────────────

def test_sf_ceil():
    df = pl.DataFrame({"x": [1.2, -1.9]})
    assert df.select(sf.ceil(pl.col("x")))["x"].to_list() == [2.0, -1.0]


def test_sf_floor():
    df = pl.DataFrame({"x": [1.9, -1.2]})
    assert df.select(sf.floor(pl.col("x")))["x"].to_list() == [1.0, -2.0]


def test_sf_log2():
    df = pl.DataFrame({"x": [8.0]})
    assert df.select(sf.log2(pl.col("x")))["x"][0] == 3.0


def test_sf_log10():
    df = pl.DataFrame({"x": [100.0]})
    assert df.select(sf.log10(pl.col("x")))["x"][0] == 2.0


def test_sf_log_natural():
    import math
    df = pl.DataFrame({"x": [1.0]})
    assert round(df.select(sf.log(pl.col("x")))["x"][0], 6) == 0.0


def test_sf_exp():
    df = pl.DataFrame({"x": [0.0]})
    assert df.select(sf.exp(pl.col("x")))["x"][0] == 1.0


def test_sf_cbrt():
    df = pl.DataFrame({"x": [27.0]})
    assert df.select(sf.cbrt(pl.col("x")))["x"][0] == 3.0


def test_sf_signum():
    df = pl.DataFrame({"x": [3.0, -2.0, 0.0]})
    assert df.select(sf.signum(pl.col("x")))["x"].to_list() == [1.0, -1.0, 0.0]


def test_sf_sign_alias():
    df = pl.DataFrame({"x": [3.0, -2.0]})
    assert df.select(sf.sign(pl.col("x")))["x"].to_list() == [1.0, -1.0]


# ── null-handling functions ───────────────────────────────────────────────────

def test_sf_nvl():
    df = pl.DataFrame({"a": [None, 2, None]})
    assert df.select(sf.nvl(pl.col("a"), -1))["a"].to_list() == [-1, 2, -1]


def test_sf_ifnull():
    df = pl.DataFrame({"a": [None, 2, None]})
    assert df.select(sf.ifnull(pl.col("a"), -1))["a"].to_list() == [-1, 2, -1]


def test_sf_nullif():
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert df.select(sf.nullif(pl.col("a"), 2))["a"].to_list() == [1, None, 3]


# ── date / time functions ─────────────────────────────────────────────────────

def test_sf_to_date():
    df = pl.DataFrame({"s": ["2023-06-15"]})
    assert df.select(sf.to_date(pl.col("s")))["s"][0] == datetime.date(2023, 6, 15)


def test_sf_to_timestamp():
    df = pl.DataFrame({"s": ["2023-06-15 10:30:00"]})
    result = df.select(sf.to_timestamp(pl.col("s")))["s"][0]
    assert result == datetime.datetime(2023, 6, 15, 10, 30, 0)


def test_sf_date_format():
    df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    assert df.select(sf.date_format(pl.col("d"), "%Y/%m/%d"))["d"][0] == "2023/06/15"


def test_sf_unix_timestamp():
    df = pl.DataFrame({"t": [datetime.datetime(2023, 6, 15, 0, 0, 0)]})
    assert df.select(sf.unix_timestamp(pl.col("t")))["t"][0] == 1686787200


def test_sf_from_unixtime():
    df = pl.DataFrame({"ts": [1686787200]})
    result = df.select(sf.from_unixtime(pl.col("ts")))["ts"][0]
    assert result.date() == datetime.date(2023, 6, 15)


# ── array functions ───────────────────────────────────────────────────────────

def test_sf_array_position_found():
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    assert df.select(sf.array_position(pl.col("a"), 20))["a"][0] == 2


def test_sf_array_position_missing():
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    assert df.select(sf.array_position(pl.col("a"), 99))["a"][0] == 0


def test_sf_array_prepend():
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    assert df.select(sf.array_prepend(pl.col("a"), 5))["a"][0].to_list() == [5, 10, 20, 30]


# ── string helper ─────────────────────────────────────────────────────────────

def test_sf_format_string():
    df = pl.DataFrame({"a": ["hello"], "b": [42]})
    result = df.select(sf.format_string("{} {}", pl.col("a"), pl.col("b")))["a"][0]
    assert result == "hello 42"


def test_sf_concat_string():
    df = pl.DataFrame({"a": [1, 2, 3]})
    # concat() in functions.py does string concat via reduce
    result = df.select(sf.concat(pl.col("a").cast(pl.String)))["a"].to_list()
    assert result == ["1", "2", "3"]


# ── trig functions ────────────────────────────────────────────────────────────

def test_sf_sin():
    import math
    df = pl.DataFrame({"x": [0.0]})
    assert df.select(sf.sin(pl.col("x")))["x"][0] == 0.0


def test_sf_cos():
    df = pl.DataFrame({"x": [0.0]})
    assert df.select(sf.cos(pl.col("x")))["x"][0] == 1.0


def test_sf_tan():
    df = pl.DataFrame({"x": [0.0]})
    assert df.select(sf.tan(pl.col("x")))["x"][0] == 0.0


def test_sf_asin():
    df = pl.DataFrame({"x": [0.0]})
    assert df.select(sf.asin(pl.col("x")))["x"][0] == 0.0


def test_sf_acos():
    df = pl.DataFrame({"x": [1.0]})
    assert df.select(sf.acos(pl.col("x")))["x"][0] == 0.0


def test_sf_atan():
    df = pl.DataFrame({"x": [0.0]})
    assert df.select(sf.atan(pl.col("x")))["x"][0] == 0.0


def test_sf_atan2():
    import math
    df = pl.DataFrame({"y": [1.0], "x": [1.0]})
    result = df.select(sf.atan2(pl.col("y"), pl.col("x")).alias("r"))["r"][0]
    assert abs(result - math.pi / 4) < 1e-9


def test_sf_sinh():
    df = pl.DataFrame({"x": [0.0]})
    assert df.select(sf.sinh(pl.col("x")))["x"][0] == 0.0


def test_sf_cosh():
    df = pl.DataFrame({"x": [0.0]})
    assert df.select(sf.cosh(pl.col("x")))["x"][0] == 1.0


def test_sf_tanh():
    df = pl.DataFrame({"x": [0.0]})
    assert df.select(sf.tanh(pl.col("x")))["x"][0] == 0.0


def test_sf_degrees():
    import math
    df = pl.DataFrame({"x": [math.pi]})
    assert abs(df.select(sf.degrees(pl.col("x")))["x"][0] - 180.0) < 1e-9


def test_sf_radians():
    import math
    df = pl.DataFrame({"x": [180.0]})
    assert abs(df.select(sf.radians(pl.col("x")))["x"][0] - math.pi) < 1e-9


# ── string extras ─────────────────────────────────────────────────────────────

def test_sf_initcap():
    df = pl.DataFrame({"s": ["hello world"]})
    assert df.select(sf.initcap(pl.col("s")))["s"][0] == "Hello World"


def test_sf_ascii():
    df = pl.DataFrame({"s": ["A"]})
    assert df.select(sf.ascii(pl.col("s")))["s"][0] == 65


def test_sf_instr():
    df = pl.DataFrame({"s": ["hello world"]})
    assert df.select(sf.instr(pl.col("s"), "world"))["s"][0] == 7


def test_sf_instr_missing():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(sf.instr(pl.col("s"), "xyz"))["s"][0] == 0


def test_sf_split_part():
    df = pl.DataFrame({"s": ["a,b,c"]})
    assert df.select(sf.split_part(pl.col("s"), ",", 2))["s"][0] == "b"


def test_sf_substring_index_positive():
    df = pl.DataFrame({"s": ["a.b.c.d"]})
    assert df.select(sf.substring_index(pl.col("s"), ".", 2))["s"][0] == "a.b"


def test_sf_substring_index_negative():
    df = pl.DataFrame({"s": ["a.b.c.d"]})
    assert df.select(sf.substring_index(pl.col("s"), ".", -2))["s"][0] == "c.d"


# ── date extras ───────────────────────────────────────────────────────────────

def test_sf_quarter():
    df = pl.DataFrame({"d": [datetime.date(2023, 4, 15)]})
    assert df.select(sf.quarter(pl.col("d")))["d"][0] == 2


def test_sf_minute():
    df = pl.DataFrame({"t": [datetime.datetime(2023, 1, 1, 14, 35, 0)]})
    assert df.select(sf.minute(pl.col("t")))["t"][0] == 35


def test_sf_second():
    df = pl.DataFrame({"t": [datetime.datetime(2023, 1, 1, 14, 35, 45)]})
    assert df.select(sf.second(pl.col("t")))["t"][0] == 45


def test_sf_weekofyear():
    df = pl.DataFrame({"d": [datetime.date(2023, 1, 9)]})  # week 2 of 2023
    assert df.select(sf.weekofyear(pl.col("d")))["d"][0] == 2


def test_sf_weekday():
    # 2023-01-09 is a Monday → weekday=0
    df = pl.DataFrame({"d": [datetime.date(2023, 1, 9)]})
    assert df.select(sf.weekday(pl.col("d")))["d"][0] == 0


def test_sf_date_trunc_month():
    df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    result = df.select(sf.date_trunc("month", pl.col("d")))["d"][0]
    assert result == datetime.date(2023, 6, 1)


def test_sf_date_trunc_year():
    df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    result = df.select(sf.date_trunc("year", pl.col("d")))["d"][0]
    assert result == datetime.date(2023, 1, 1)


def test_sf_date_trunc_quarter():
    df = pl.DataFrame({"d": [datetime.date(2023, 5, 15)]})  # Q2 → starts April 1
    result = df.select(sf.date_trunc("quarter", pl.col("d")))["d"][0]
    assert result == datetime.date(2023, 4, 1)


def test_sf_make_date():
    df = pl.DataFrame({"y": [2023], "m": [6], "d": [15]})
    result = df.select(sf.make_date(pl.col("y"), pl.col("m"), pl.col("d")).alias("dt"))["dt"][0]
    assert result == datetime.date(2023, 6, 15)


# ── aggregate extras ──────────────────────────────────────────────────────────

def test_sf_stddev():
    # [1, 2, 3]: sample stddev = 1.0
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = df.select(sf.stddev(pl.col("x")))["stddev"][0]
    assert abs(result - 1.0) < 1e-6


def test_sf_variance():
    # [1, 2, 3]: sample variance = 1.0
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = df.select(sf.variance(pl.col("x")))["variance"][0]
    assert abs(result - 1.0) < 1e-6


def test_sf_median():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert df.select(sf.median(pl.col("x")))["median"][0] == 3.0


def test_sf_count_distinct():
    df = pl.DataFrame({"x": [1, 2, 2, 3, 3]})
    assert df.select(sf.count_distinct(pl.col("x")).alias("n"))["n"][0] == 3


def test_sf_count_if():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = df.select(sf.count_if(pl.col("x") > 2).alias("n"))["n"][0]
    assert result == 3


def test_sf_bool_and_true():
    df = pl.DataFrame({"x": [True, True, True]})
    assert df.select(sf.bool_and(pl.col("x")).alias("r"))["r"][0] is True


def test_sf_bool_and_false():
    df = pl.DataFrame({"x": [True, False, True]})
    assert df.select(sf.bool_and(pl.col("x")).alias("r"))["r"][0] is False


def test_sf_bool_or():
    df = pl.DataFrame({"x": [False, False, True]})
    assert df.select(sf.bool_or(pl.col("x")).alias("r"))["r"][0] is True


# ── null extras ───────────────────────────────────────────────────────────────

def test_sf_nanvl():
    df = pl.DataFrame({"x": [float("nan"), 2.0, float("nan")]})
    result = df.select(sf.nanvl(pl.col("x"), -1.0))["x"].to_list()
    assert result[0] == -1.0
    assert result[1] == 2.0
    assert result[2] == -1.0


def test_sf_nvl2_not_null():
    df = pl.DataFrame({"a": [1, None, 3]})
    result = df.select(sf.nvl2(pl.col("a"), 100, 0))["a"].to_list()
    assert result == [100, 0, 100]


# ── bitwise extras ────────────────────────────────────────────────────────────

def test_sf_bitwise_not():
    df = pl.DataFrame({"x": [5]})
    # ~5 = -6 in two's complement
    assert df.select(sf.bitwise_not(pl.col("x")))["x"][0] == -6


def test_sf_shiftleft():
    df = pl.DataFrame({"x": [1]})
    assert df.select(sf.shiftleft(pl.col("x"), 3))["x"][0] == 8


def test_sf_shiftright():
    df = pl.DataFrame({"x": [8]})
    assert df.select(sf.shiftright(pl.col("x"), 2))["x"][0] == 2


# ── array extras ─────────────────────────────────────────────────────────────

def test_sf_element_at():
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    assert df.select(sf.element_at(pl.col("a"), 2))["a"][0] == 20


def test_sf_element_at_negative():
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    assert df.select(sf.element_at(pl.col("a"), -1))["a"][0] == 30


def test_sf_arrays_overlap_true():
    df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[3, 4, 5]]})
    assert df.select(sf.arrays_overlap(pl.col("a"), pl.col("b")))["a"][0] is True


def test_sf_arrays_overlap_false():
    df = pl.DataFrame({"a": [[1, 2]], "b": [[3, 4]]})
    assert df.select(sf.arrays_overlap(pl.col("a"), pl.col("b")))["a"][0] is False


def test_sf_array_repeat():
    df = pl.DataFrame({"x": [1]})
    result = df.select(sf.array_repeat(pl.lit(7), 3).alias("a"))["a"][0].to_list()
    assert result == [7, 7, 7]


# ── window / ranking functions ────────────────────────────────────────────────

def test_sf_row_number():
    df = pl.DataFrame({"x": [10, 20, 30]})
    result = df.select(sf.row_number().alias("rn"))["rn"].to_list()
    assert result == [1, 2, 3]


def test_sf_rank():
    df = pl.DataFrame({"x": [3, 1, 2, 1]})
    result = df.select(sf.rank(pl.col("x")).alias("r"))["r"].to_list()
    assert sorted(result) == [1, 1, 3, 4]


def test_sf_dense_rank():
    df = pl.DataFrame({"x": [3, 1, 2, 1]})
    result = df.select(sf.dense_rank(pl.col("x")).alias("r"))["r"].to_list()
    assert sorted(result) == [1, 1, 2, 3]


def test_sf_lag():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    result = df.select(sf.lag(pl.col("x"), 1).alias("l"))["l"].to_list()
    assert result == [None, 1, 2, 3]


def test_sf_lead():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    result = df.select(sf.lead(pl.col("x"), 1).alias("l"))["l"].to_list()
    assert result == [2, 3, 4, None]


def test_sf_lag_default():
    df = pl.DataFrame({"x": [1, 2, 3]})
    result = df.select(sf.lag(pl.col("x"), 1, 0).alias("l"))["l"].to_list()
    assert result == [0, 1, 2]


# ── math extras ───────────────────────────────────────────────────────────────

def test_sf_pi():
    import math
    df = pl.DataFrame({"x": [1]})
    result = df.select(sf.pi().alias("p"))["p"][0]
    assert abs(result - math.pi) < 1e-12


def test_sf_e():
    import math
    df = pl.DataFrame({"x": [1]})
    result = df.select(sf.e().alias("v"))["v"][0]
    assert abs(result - math.e) < 1e-12


def test_sf_factorial():
    df = pl.DataFrame({"x": [5]})
    assert df.select(sf.factorial(pl.col("x")))["x"][0] == 120


def test_sf_hex():
    df = pl.DataFrame({"x": [255]})
    assert df.select(sf.hex(pl.col("x")))["x"][0] == "FF"


def test_sf_unhex():
    df = pl.DataFrame({"x": ["FF"]})
    assert df.select(sf.unhex(pl.col("x")))["x"][0] == 255


def test_sf_rand():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = df.select(sf.rand(seed=42).alias("r"))["r"].to_list()
    assert len(result) == 5
    assert all(0.0 <= v < 1.0 for v in result)


def test_sf_randn():
    df = pl.DataFrame({"x": [1, 2, 3]})
    result = df.select(sf.randn(seed=0).alias("r"))["r"].to_list()
    assert len(result) == 3


# ── misc extras ───────────────────────────────────────────────────────────────

def test_sf_lit():
    df = pl.DataFrame({"x": [1, 2, 3]})
    result = df.with_columns(sf.lit(42).alias("v"))["v"].to_list()
    assert result == [42, 42, 42]


def test_sf_typed_lit():
    df = pl.DataFrame({"x": [1]})
    assert df.with_columns(sf.typedLit(99).alias("v"))["v"][0] == 99


# ── string extras (batch 2) ───────────────────────────────────────────────────

def test_sf_chr():
    df = pl.DataFrame({"x": [65, 66, 67]})
    assert df.select(sf.chr(pl.col("x")))["x"].to_list() == ["A", "B", "C"]


def test_sf_find_in_set_found():
    df = pl.DataFrame({"s": ["a,b,c,d"]})
    assert df.select(sf.find_in_set("c", pl.col("s")))["s"][0] == 3


def test_sf_find_in_set_missing():
    df = pl.DataFrame({"s": ["a,b,c"]})
    assert df.select(sf.find_in_set("z", pl.col("s")))["s"][0] == 0


def test_sf_regexp_like_match():
    df = pl.DataFrame({"s": ["hello123", "world"]})
    result = df.select(sf.regexp_like(pl.col("s"), r"\d+"))["s"].to_list()
    assert result == [True, False]


def test_sf_printf():
    df = pl.DataFrame({"a": ["hello"], "b": [42]})
    result = df.select(sf.printf("{} {}", pl.col("a"), pl.col("b")))["a"][0]
    assert result == "hello 42"


def test_sf_overlay_default_len():
    df = pl.DataFrame({"s": ["hello world"]})
    result = df.select(sf.overlay(pl.col("s"), "there", 7))["s"][0]
    assert result == "hello there"


def test_sf_overlay_with_len():
    df = pl.DataFrame({"s": ["hello world"]})
    result = df.select(sf.overlay(pl.col("s"), "X", 1, 5))["s"][0]
    assert result == "X world"


# ── math extras (batch 2) ─────────────────────────────────────────────────────

def test_sf_bround():
    # 2.5 rounds to 2.0 (half-to-even), 3.5 rounds to 4.0
    df = pl.DataFrame({"x": [2.5, 3.5]})
    result = df.select(sf.bround(pl.col("x")))["x"].to_list()
    assert result == [2.0, 4.0]


def test_sf_hypot():
    import math
    df = pl.DataFrame({"x": [3.0], "y": [4.0]})
    result = df.select(sf.hypot(pl.col("x"), pl.col("y")).alias("h"))["h"][0]
    assert result == 5.0


def test_sf_pmod_positive():
    df = pl.DataFrame({"x": [7]})
    assert df.select(sf.pmod(pl.col("x"), 3).alias("r"))["r"][0] == 1


def test_sf_pmod_negative_dividend():
    # pmod(-7, 3) should return 2, not -1
    df = pl.DataFrame({"x": [-7]})
    assert df.select(sf.pmod(pl.col("x"), 3).alias("r"))["r"][0] == 2


def test_sf_shiftrightunsigned():
    # -1 as int64 is all 1-bits; unsigned right shift by 1 should give a large positive
    df = pl.DataFrame({"x": [-1]})
    result = df.select(sf.shiftrightunsigned(pl.col("x"), 1).alias("r"))["r"][0]
    assert result > 0


# ── date extras (batch 2) ─────────────────────────────────────────────────────

def test_sf_trunc_month():
    df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    assert df.select(sf.trunc(pl.col("d"), "month"))["d"][0] == datetime.date(2023, 6, 1)


def test_sf_next_day_monday():
    # 2023-01-11 is Wednesday; next Monday is 2023-01-16
    df = pl.DataFrame({"d": [datetime.date(2023, 1, 11)]})
    result = df.select(sf.next_day(pl.col("d"), "Mon"))["d"][0]
    assert result == datetime.date(2023, 1, 16)


def test_sf_next_day_same_weekday():
    # 2023-01-09 is Monday; next Monday should be 2023-01-16 (not same day)
    df = pl.DataFrame({"d": [datetime.date(2023, 1, 9)]})
    result = df.select(sf.next_day(pl.col("d"), "Monday"))["d"][0]
    assert result == datetime.date(2023, 1, 16)


def test_sf_make_timestamp():
    df = pl.DataFrame({"x": [1]})
    result = df.select(
        sf.make_timestamp(2023, 6, 15, 10, 30, 0).alias("ts")
    )["ts"][0]
    assert result == datetime.datetime(2023, 6, 15, 10, 30, 0)


def test_sf_unix_date():
    # 1970-01-01 → 0; 1970-01-02 → 1
    df = pl.DataFrame({"d": [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2)]})
    result = df.select(sf.unix_date(pl.col("d")))["d"].to_list()
    assert result == [0, 1]


def test_sf_from_unixtime_no_fmt():
    df = pl.DataFrame({"ts": [0]})
    result = df.select(sf.from_unixtime(pl.col("ts")))["ts"][0]
    assert result.date() == datetime.date(1970, 1, 1)


def test_sf_from_unixtime_with_fmt():
    df = pl.DataFrame({"ts": [0]})
    result = df.select(sf.from_unixtime(pl.col("ts"), "%Y-%m-%d"))["ts"][0]
    assert result == "1970-01-01"


# ── array extras (batch 2) ────────────────────────────────────────────────────

def test_sf_arrays_zip():
    df = pl.DataFrame({"a": [[1, 2]], "b": [["x", "y"]]})
    result = df.select(sf.arrays_zip(pl.col("a"), pl.col("b")).alias("z"))["z"][0]
    result_list = result.to_list()
    assert result_list[0]["a"] == 1
    assert result_list[0]["b"] == "x"
    assert result_list[1]["a"] == 2
    assert result_list[1]["b"] == "y"


def test_sf_shuffle():
    df = pl.DataFrame({"a": [[1, 2, 3, 4, 5]]})
    result = df.select(sf.shuffle(pl.col("a"), seed=0))["a"][0].to_list()
    assert sorted(result) == [1, 2, 3, 4, 5]
    assert len(result) == 5


def test_sf_exists_true():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    assert df.select(sf.exists(pl.col("a"), lambda e: e > 2))["a"][0] is True


def test_sf_exists_false():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    assert df.select(sf.exists(pl.col("a"), lambda e: e > 10))["a"][0] is False


def test_sf_aggregate():
    df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
    result = df.select(
        sf.aggregate(pl.col("a"), 0, lambda acc, e: acc + e).alias("s")
    )["s"][0]
    assert result == 10


def test_sf_aggregate_with_finish():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    result = df.select(
        sf.aggregate(pl.col("a"), 0, lambda acc, e: acc + e, lambda r: r * 2).alias("s")
    )["s"][0]
    assert result == 12


# ── aggregate extras (batch 2) ────────────────────────────────────────────────

def test_sf_corr():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    result = df.select(sf.corr(pl.col("x"), pl.col("y")).alias("r"))["r"][0]
    assert abs(result - 1.0) < 1e-9


def test_sf_covar_samp():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    result = df.select(sf.covar_samp(pl.col("x"), pl.col("y")).alias("c"))["c"][0]
    assert abs(result - 1.0) < 1e-9


def test_sf_covar_pop():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    result = df.select(sf.covar_pop(pl.col("x"), pl.col("y")).alias("c"))["c"][0]
    # pop covariance of perfectly correlated [1,2,3] vs itself
    assert abs(result - 2.0 / 3.0) < 1e-9


def test_sf_kurtosis():
    df = pl.DataFrame({"x": [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0]})
    result = df.select(sf.kurtosis(pl.col("x")))["kurtosis"][0]
    assert result is not None


def test_sf_skewness():
    # Right-skewed data
    df = pl.DataFrame({"x": [1.0, 1.0, 1.0, 2.0, 5.0]})
    result = df.select(sf.skewness(pl.col("x")))["skewness"][0]
    assert result > 0  # positive skew


def test_sf_mode():
    df = pl.DataFrame({"x": [1, 2, 2, 3, 3, 3]})
    result = df.select(sf.mode(pl.col("x")).alias("m"))["m"][0]
    assert result == 3


def test_sf_percentile():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = df.select(sf.percentile(pl.col("x"), 0.5))["percentile"][0]
    assert result == 3.0


def test_sf_sum_distinct():
    df = pl.DataFrame({"x": [1, 2, 2, 3]})
    result = df.select(sf.sum_distinct(pl.col("x")))["sum_distinct"][0]
    assert result == 6


def test_sf_any_value():
    df = pl.DataFrame({"x": [None, 2, 3]})
    result = df.select(sf.any_value(pl.col("x")).alias("v"))["v"][0]
    assert result is None  # first() returns first row value


def test_sf_approx_count_distinct():
    df = pl.DataFrame({"x": [1, 2, 2, 3, 3, 3]})
    result = df.select(sf.approx_count_distinct(pl.col("x")).alias("n"))["n"][0]
    assert result == 3


# ── null / type-check standalones ─────────────────────────────────────────────

def test_sf_isnull_true():
    df = pl.DataFrame({"x": [None, 1, None]})
    result = df.select(sf.isnull(pl.col("x")))["x"].to_list()
    assert result == [True, False, True]


def test_sf_isnan_true():
    df = pl.DataFrame({"x": [float("nan"), 1.0, float("nan")]})
    result = df.select(sf.isnan(pl.col("x")))["x"].to_list()
    assert result == [True, False, True]


def test_sf_isnotnull():
    df = pl.DataFrame({"x": [None, 2, None]})
    result = df.select(sf.isnotnull(pl.col("x")))["x"].to_list()
    assert result == [False, True, False]


# ── hash extras ───────────────────────────────────────────────────────────────

def test_sf_sha2_256():
    import hashlib
    df = pl.DataFrame({"s": ["hello"]})
    expected = hashlib.sha256(b"hello").hexdigest()
    assert df.select(sf.sha2(pl.col("s"), 256))["s"][0] == expected


def test_sf_sha2_512():
    import hashlib
    df = pl.DataFrame({"s": ["hello"]})
    expected = hashlib.sha512(b"hello").hexdigest()
    assert df.select(sf.sha2(pl.col("s"), 512))["s"][0] == expected


def test_sf_crc32():
    import binascii
    df = pl.DataFrame({"s": ["hello"]})
    expected = binascii.crc32(b"hello") & 0xFFFF_FFFF
    assert df.select(sf.crc32(pl.col("s")))["s"][0] == expected


def test_sf_hash_consistent():
    df = pl.DataFrame({"x": [1, 2, 3]})
    r1 = df.select(sf.hash(pl.col("x")).alias("h"))["h"].to_list()
    r2 = df.select(sf.hash(pl.col("x")).alias("h"))["h"].to_list()
    assert r1 == r2


def test_sf_xxhash64_consistent():
    df = pl.DataFrame({"x": [10, 20]})
    r1 = df.select(sf.xxhash64(pl.col("x")).alias("h"))["h"].to_list()
    r2 = df.select(sf.xxhash64(pl.col("x")).alias("h"))["h"].to_list()
    assert r1 == r2


# ── URL helpers ───────────────────────────────────────────────────────────────

def test_sf_url_encode():
    df = pl.DataFrame({"s": ["hello world"]})
    assert df.select(sf.url_encode(pl.col("s")))["s"][0] == "hello%20world"


def test_sf_url_decode():
    df = pl.DataFrame({"s": ["hello%20world"]})
    assert df.select(sf.url_decode(pl.col("s")))["s"][0] == "hello world"


# ── JSON helper ───────────────────────────────────────────────────────────────

def test_sf_get_json_object():
    df = pl.DataFrame({"s": ['{"name": "alice", "age": 30}']})
    result = df.select(sf.get_json_object(pl.col("s"), "$.name"))["s"][0]
    assert result == "alice"


# ── string extras (batch 3) ───────────────────────────────────────────────────

def test_sf_unbase64():
    import base64
    encoded = base64.b64encode(b"hello").decode()
    df = pl.DataFrame({"s": [encoded]})
    assert df.select(sf.unbase64(pl.col("s")))["s"][0] == "hello"


def test_sf_levenshtein():
    df = pl.DataFrame({"a": ["kitten"], "b": ["sitting"]})
    result = df.select(sf.levenshtein(pl.col("a"), pl.col("b")).alias("r"))["r"][0]
    assert result == 3


def test_sf_regexp_substr():
    df = pl.DataFrame({"s": ["hello world 123"]})
    result = df.select(sf.regexp_substr(pl.col("s"), r"\d+"))["s"][0]
    assert result == "123"


def test_sf_elt():
    df = pl.DataFrame({"idx": [1, 2, 3], "a": ["x", "x", "x"], "b": ["y", "y", "y"]})
    result = df.select(sf.elt(pl.col("idx"), pl.col("a"), pl.col("b")).alias("r"))["r"].to_list()
    assert result == ["x", "y", None]


# ── math extras (batch 3) ─────────────────────────────────────────────────────

def test_sf_log1p():
    import math
    df = pl.DataFrame({"x": [0.0, 1.0]})
    result = df.select(sf.log1p(pl.col("x")))["x"].to_list()
    assert abs(result[0] - 0.0) < 1e-9
    assert abs(result[1] - math.log(2)) < 1e-9


def test_sf_expm1():
    import math
    df = pl.DataFrame({"x": [0.0, 1.0]})
    result = df.select(sf.expm1(pl.col("x")))["x"].to_list()
    assert abs(result[0] - 0.0) < 1e-9
    assert abs(result[1] - (math.e - 1)) < 1e-9


def test_sf_rint():
    df = pl.DataFrame({"x": [1.4, 1.6, -1.5]})
    result = df.select(sf.rint(pl.col("x")))["x"].to_list()
    assert result == [1.0, 2.0, -2.0]


def test_sf_remainder():
    import math
    df = pl.DataFrame({"x": [5.0], "y": [3.0]})
    result = df.select(sf.remainder(pl.col("x"), pl.col("y")).alias("r"))["r"][0]
    assert abs(result - math.remainder(5.0, 3.0)) < 1e-9


def test_sf_gcd():
    df = pl.DataFrame({"a": [12], "b": [8]})
    result = df.select(sf.gcd(pl.col("a"), pl.col("b")).alias("r"))["r"][0]
    assert result == 4


def test_sf_lcm():
    df = pl.DataFrame({"a": [4], "b": [6]})
    result = df.select(sf.lcm(pl.col("a"), pl.col("b")).alias("r"))["r"][0]
    assert result == 12


def test_sf_bitcount():
    df = pl.DataFrame({"x": [7]})  # 0b111 → 3 set bits
    result = df.select(sf.bitcount(pl.col("x")))["x"][0]
    assert result == 3


def test_sf_toDegrees():
    import math
    df = pl.DataFrame({"x": [math.pi]})
    result = df.select(sf.toDegrees(pl.col("x")))["x"][0]
    assert abs(result - 180.0) < 1e-9


def test_sf_toRadians():
    import math
    df = pl.DataFrame({"x": [180.0]})
    result = df.select(sf.toRadians(pl.col("x")))["x"][0]
    assert abs(result - math.pi) < 1e-9


# ── date extras (batch 3) ─────────────────────────────────────────────────────

def test_sf_months_between():
    from datetime import date
    df = pl.DataFrame({
        "end": [date(2021, 3, 15)],
        "start": [date(2021, 1, 15)],
    })
    result = df.select(sf.months_between(pl.col("end"), pl.col("start")).alias("r"))["r"][0]
    assert abs(result - 2.0) < 1e-6


# ── array extras (batch 3) ────────────────────────────────────────────────────

def test_sf_array_reverse():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    result = df.select(sf.array_reverse(pl.col("a")))["a"][0].to_list()
    assert result == [3, 2, 1]


def test_sf_array_insert():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    result = df.select(sf.array_insert(pl.col("a"), 2, 99).alias("r"))["r"][0]
    if hasattr(result, "to_list"):
        result = result.to_list()
    assert result == [1, 99, 2, 3]


# ── aggregate/window extras (batch 3) ─────────────────────────────────────────

def test_sf_mean():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = df.select(sf.mean(pl.col("x")))["x"][0]
    assert result == 2.0


def test_sf_ntile():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    result = df.select(sf.ntile(2).alias("t"))["t"].to_list()
    assert result == [1, 1, 2, 2]


def test_sf_cume_dist():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    result = df.select(sf.cume_dist().alias("c"))["c"].to_list()
    assert result == [0.25, 0.5, 0.75, 1.0]


def test_sf_percent_rank():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    result = df.select(sf.percent_rank().alias("p"))["p"].to_list()
    expected = [0.0, 1 / 3, 2 / 3, 1.0]
    for a, b in zip(result, expected):
        assert abs(a - b) < 1e-9


# ── struct / map extras (batch 3) ─────────────────────────────────────────────

def test_sf_to_json():
    df = pl.DataFrame({"s": [{"a": 1, "b": 2}]}, schema={"s": pl.Struct({"a": pl.Int32, "b": pl.Int32})})
    result = df.select(sf.to_json(pl.col("s")))["s"][0]
    import json
    parsed = json.loads(result)
    assert parsed == {"a": 1, "b": 2}


def test_sf_map_keys():
    df = pl.DataFrame({"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]})
    result = df.select(sf.map_keys(pl.col("m")))["m"][0].to_list()
    assert sorted(result) == ["a", "b"]


def test_sf_map_values():
    df = pl.DataFrame({"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]})
    result = df.select(sf.map_values(pl.col("m")))["m"][0].to_list()
    assert sorted(result) == ["1", "2"]


# ── first / last with ignorenulls ─────────────────────────────────────────────

def test_sf_first_ignorenulls():
    df = pl.DataFrame({"x": [None, None, 3, 4]}, schema={"x": pl.Int32})
    result = df.select(sf.first(pl.col("x"), ignorenulls=True))["x"][0]
    assert result == 3


def test_sf_last_ignorenulls():
    df = pl.DataFrame({"x": [1, 2, None, None]}, schema={"x": pl.Int32})
    result = df.select(sf.last(pl.col("x"), ignorenulls=True))["x"][0]
    assert result == 2


# ── unix_timestamp with format ─────────────────────────────────────────────────

def test_sf_unix_timestamp_with_fmt():
    df = pl.DataFrame({"ts": ["2020-01-01 00:00:00"]})
    result = df.select(sf.unix_timestamp(pl.col("ts"), fmt="%Y-%m-%d %H:%M:%S").alias("t"))["t"][0]
    assert result == 1577836800


def test_sf_unix_timestamp_no_args():
    import time
    result = pl.select(sf.unix_timestamp().alias("t"))["t"][0]
    assert abs(result - int(time.time())) <= 2


# ── monotonic_id ──────────────────────────────────────────────────────────────

def test_sf_monotonic_id():
    df = pl.DataFrame({"x": [10, 20, 30]})
    result = df.select(sf.monotonic_id().alias("id"))["id"].to_list()
    assert result == [0, 1, 2]


# ── NotImplementedError stubs ─────────────────────────────────────────────────

import pytest as _pytest


def test_sf_soundex_raises():
    with _pytest.raises(NotImplementedError):
        sf.soundex(pl.col("x"))


def test_sf_from_json_raises():
    with _pytest.raises(NotImplementedError):
        sf.from_json(pl.col("x"), None)


def test_sf_posexplode_raises():
    with _pytest.raises(NotImplementedError):
        sf.posexplode(pl.col("x"))


def test_sf_window_raises():
    with _pytest.raises(NotImplementedError):
        sf.window(pl.col("ts"), "1 day")
