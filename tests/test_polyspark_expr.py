"""Tests for polyspark Expr (Column) monkey-patches."""

import math
import pytest
import polars as pl

import src.sparkpolars.polyspark.sql.functions  # noqa: F401 — installs patches


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def int_df():
    return pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})


@pytest.fixture()
def str_df():
    return pl.DataFrame({"s": ["hello", "world", "foo", "bar", "Hello World"]})


@pytest.fixture()
def float_df():
    return pl.DataFrame({"v": [1.0, float("nan"), 3.0, float("nan"), 5.0]})


@pytest.fixture()
def struct_df():
    return pl.DataFrame({"s": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]})


@pytest.fixture()
def list_df():
    return pl.DataFrame({"arr": [[1, 2, 3], [4, 5, 6]]})


# ── already-patched (smoke tests) ─────────────────────────────────────────────
def test_isNull(int_df):
    df = pl.DataFrame({"a": [1, None, 3]})
    result = df.select(pl.col("a").isNull())
    assert result["a"].to_list() == [False, True, False]


def test_isNotNull(int_df):
    df = pl.DataFrame({"a": [1, None, 3]})
    result = df.select(pl.col("a").isNotNull())
    assert result["a"].to_list() == [True, False, True]


def test_isin(int_df):
    result = int_df.filter(pl.col("a").isin(1, 3, 5))
    assert result["a"].to_list() == [1, 3, 5]


def test_between(int_df):
    result = int_df.filter(pl.col("a").between(2, 4))
    assert result["a"].to_list() == [2, 3, 4]


def test_eqNullSafe(int_df):
    result = int_df.select(pl.col("a").eqNullSafe(pl.col("b")))
    assert result["a"].to_list() == [False, False, True, False, False]


def test_rlike(str_df):
    result = str_df.filter(pl.col("s").rlike("^h.*"))
    assert "hello" in result["s"].to_list()


def test_startswith(str_df):
    result = str_df.filter(pl.col("s").startswith("he"))
    assert result["s"].to_list() == ["hello"]


def test_endswith(str_df):
    result = str_df.filter(pl.col("s").endswith("ld"))
    assert set(result["s"].to_list()) == {"world", "Hello World"}


def test_contains_expr(str_df):
    result = str_df.filter(pl.col("s").contains("oo"))
    assert result["s"].to_list() == ["foo"]


def test_substr(str_df):
    result = str_df.select(pl.col("s").substr(1, 3))
    assert result["s"][0] == "hel"


def test_outer_noop(int_df):
    expr = pl.col("a")
    assert expr.outer() is expr


def test_cast(int_df):
    result = int_df.select(pl.col("a").cast(pl.Float64))
    assert result["a"].dtype == pl.Float64


def test_asc(int_df):
    expr = pl.col("a").asc()
    assert isinstance(expr, pl.Expr)
    result = int_df.select(expr)
    assert result["a"].to_list() == sorted(int_df["a"].to_list())


def test_desc(int_df):
    expr = pl.col("a").desc()
    assert isinstance(expr, pl.Expr)
    result = int_df.select(expr)
    assert result["a"].to_list() == sorted(int_df["a"].to_list(), reverse=True)


def test_asc_nulls_last():
    df = pl.DataFrame({"a": [None, 1, 2]})
    expr = pl.col("a").asc_nulls_last()
    result = df.select(expr)
    assert result["a"][-1] is None


def test_desc_nulls_first():
    df = pl.DataFrame({"a": [None, 1, 2]})
    expr = pl.col("a").desc_nulls_first()
    result = df.select(expr)
    assert result["a"][0] is None


# ── isNaN ─────────────────────────────────────────────────────────────────────
def test_isNaN(float_df):
    result = float_df.select(pl.col("v").isNaN())
    assert result["v"].to_list() == [False, True, False, True, False]


def test_isNaN_no_nan(int_df):
    result = int_df.select(pl.col("a").cast(pl.Float64).isNaN())
    assert all(v is False for v in result["a"].to_list())


# ── astype ────────────────────────────────────────────────────────────────────
def test_astype_to_float(int_df):
    result = int_df.select(pl.col("a").astype(pl.Float64))
    assert result["a"].dtype == pl.Float64


def test_astype_to_string(int_df):
    result = int_df.select(pl.col("a").astype(pl.String))
    assert result["a"][0] == "1"


# ── bitwiseAND / bitwiseOR / bitwiseXOR ───────────────────────────────────────
def test_bitwiseAND(int_df):
    # 0b01 & 0b01 = 1, 0b10 & 0b100 = 0 ...
    df = pl.DataFrame({"a": [0b1010, 0b1100], "b": [0b1100, 0b1010]})
    result = df.select(pl.col("a").bitwiseAND(pl.col("b")).alias("r"))
    assert result["r"].to_list() == [0b1000, 0b1000]


def test_bitwiseOR(int_df):
    df = pl.DataFrame({"a": [0b1010, 0b0000], "b": [0b0101, 0b1111]})
    result = df.select(pl.col("a").bitwiseOR(pl.col("b")).alias("r"))
    assert result["r"].to_list() == [0b1111, 0b1111]


def test_bitwiseXOR():
    df = pl.DataFrame({"a": [0b1010, 0b1100], "b": [0b1100, 0b1010]})
    result = df.select(pl.col("a").bitwiseXOR(pl.col("b")).alias("r"))
    assert result["r"].to_list() == [0b0110, 0b0110]


def test_bitwiseAND_scalar():
    df = pl.DataFrame({"a": [0b1111, 0b1010]})
    result = df.select(pl.col("a").bitwiseAND(pl.lit(0b1010)).alias("r"))
    assert result["r"].to_list() == [0b1010, 0b1010]


# ── like / ilike ──────────────────────────────────────────────────────────────
def test_like_percent_wildcard(str_df):
    result = str_df.filter(pl.col("s").like("hel%"))
    assert result["s"].to_list() == ["hello"]


def test_like_underscore_wildcard(str_df):
    result = str_df.filter(pl.col("s").like("_oo"))
    assert result["s"].to_list() == ["foo"]


def test_like_no_wildcards(str_df):
    result = str_df.filter(pl.col("s").like("bar"))
    assert result["s"].to_list() == ["bar"]


def test_like_case_sensitive(str_df):
    # "hello" matches "hel%" but "Hello World" does not
    result = str_df.filter(pl.col("s").like("hel%"))
    assert "Hello World" not in result["s"].to_list()


def test_ilike_case_insensitive(str_df):
    result = str_df.filter(pl.col("s").ilike("hel%"))
    names = result["s"].to_list()
    assert "hello" in names
    assert "Hello World" in names


def test_ilike_underscore(str_df):
    result = str_df.filter(pl.col("s").ilike("_oo"))
    assert result["s"].to_list() == ["foo"]


# ── getField ──────────────────────────────────────────────────────────────────
def test_getField(struct_df):
    result = struct_df.select(pl.col("s").getField("x"))
    assert result["x"].to_list() == [1, 3]


def test_getField_y(struct_df):
    result = struct_df.select(pl.col("s").getField("y"))
    assert result["y"].to_list() == [2, 4]


# ── try_cast ──────────────────────────────────────────────────────────────────
def test_try_cast_success():
    df = pl.DataFrame({"s": ["1", "2", "3"]})
    result = df.select(pl.col("s").try_cast(pl.Int64))
    assert result["s"].to_list() == [1, 2, 3]


def test_try_cast_failure_returns_null():
    df = pl.DataFrame({"s": ["1", "abc", "3"]})
    result = df.select(pl.col("s").try_cast(pl.Int64))
    assert result["s"][1] is None


# ── withField ─────────────────────────────────────────────────────────────────
def test_withField_add(struct_df):
    result = struct_df.select(pl.col("s").withField("z", pl.lit(99)))
    fields = result["s"][0]
    assert fields["z"] == 99


def test_withField_replace(struct_df):
    result = struct_df.select(pl.col("s").withField("x", pl.lit(0)))
    assert result["s"][0]["x"] == 0
    assert result["s"][0]["y"] == 2  # unchanged


# ── dropFields ────────────────────────────────────────────────────────────────
def test_dropFields_raises(struct_df):
    with pytest.raises(NotImplementedError):
        struct_df.select(pl.col("s").dropFields("x"))


# ── transform (list) ──────────────────────────────────────────────────────────
def test_transform_list(list_df):
    result = list_df.select(pl.col("arr").transform(lambda e: e * 2))
    assert result["arr"].to_list()[0] == [2, 4, 6]


def test_transform_list_filter(list_df):
    result = list_df.select(pl.col("arr").transform(lambda e: e.filter(e > 2)))
    assert result["arr"].to_list()[0] == [3]


# ── when / otherwise on Expr ──────────────────────────────────────────────────
def test_when_on_expr(int_df):
    # col("a").when(condition, value).otherwise(default) — a=[1,2,3,4,5], a>3 → [F,F,F,T,T]
    result = int_df.select(
        pl.col("a").when(pl.col("a") > 3, pl.lit(99)).otherwise(pl.lit(0)).alias("r")
    )
    assert result["r"].to_list() == [0, 0, 0, 99, 99]


def test_otherwise_on_expr():
    df = pl.DataFrame({"a": [1, None, 3]})
    result = df.select(pl.col("a").otherwise(-1).alias("r"))
    assert result["r"].to_list() == [1, -1, 3]


# ── getItem (JSON map) ────────────────────────────────────────────────────────
def test_getItem_json():
    # getItem is designed for JSON-encoded map columns (returns String)
    import json
    df = pl.DataFrame({"m": [json.dumps({"a": "v1"}), json.dumps({"a": "v2"})]})
    result = df.select(pl.col("m").getItem(pl.lit("a")).alias("val"))
    assert result["val"].to_list() == ["v1", "v2"]


# ── string Expr methods ───────────────────────────────────────────────────────

@pytest.fixture()
def words_df():
    return pl.DataFrame({"s": ["hello", "world", "foo"]})


@pytest.fixture()
def padded_df():
    return pl.DataFrame({"s": ["  hello  ", "  world"]})


def test_upper(words_df):
    assert words_df.select(pl.col("s").upper())["s"].to_list() == ["HELLO", "WORLD", "FOO"]


def test_lower():
    df = pl.DataFrame({"s": ["HELLO", "World"]})
    assert df.select(pl.col("s").lower())["s"].to_list() == ["hello", "world"]


def test_trim(padded_df):
    assert padded_df.select(pl.col("s").trim())["s"].to_list() == ["hello", "world"]


def test_ltrim(padded_df):
    result = padded_df.select(pl.col("s").ltrim())["s"].to_list()
    assert result == ["hello  ", "world"]


def test_rtrim(padded_df):
    result = padded_df.select(pl.col("s").rtrim())["s"].to_list()
    assert result == ["  hello", "  world"]


def test_btrim(padded_df):
    assert padded_df.select(pl.col("s").btrim())["s"].to_list() == ["hello", "world"]


def test_lpad():
    df = pl.DataFrame({"s": ["hi"]})
    assert df.select(pl.col("s").lpad(5, "0"))["s"][0] == "000hi"


def test_rpad():
    df = pl.DataFrame({"s": ["hi"]})
    assert df.select(pl.col("s").rpad(5, "0"))["s"][0] == "hi000"


def test_left(words_df):
    assert words_df.select(pl.col("s").left(3))["s"].to_list() == ["hel", "wor", "foo"]


def test_right(words_df):
    assert words_df.select(pl.col("s").right(3))["s"].to_list() == ["llo", "rld", "foo"]


def test_length(words_df):
    assert words_df.select(pl.col("s").length())["s"].to_list() == [5, 5, 3]


def test_locate():
    df = pl.DataFrame({"s": ["hello"]})
    # Polars str.find is 0-based; 'll' starts at index 2
    assert df.select(pl.col("s").locate("ll"))["s"][0] == 2


def test_repeat():
    df = pl.DataFrame({"s": ["ab"]})
    assert df.select(pl.col("s").repeat(3))["s"][0] == "ababab"


def test_reverse(words_df):
    assert words_df.select(pl.col("s").reverse())["s"].to_list() == ["olleh", "dlrow", "oof"]


def test_split():
    df = pl.DataFrame({"s": ["a,b,c"]})
    assert df.select(pl.col("s").split(","))["s"][0].to_list() == ["a", "b", "c"]


def test_split_limit():
    df = pl.DataFrame({"s": ["a,b,c"]})
    assert df.select(pl.col("s").split(",", limit=2))["s"][0].to_list() == ["a", "b"]


def test_regexp_count():
    df = pl.DataFrame({"s": ["hello world"]})
    assert df.select(pl.col("s").regexp_count("l"))["s"][0] == 3


def test_regexp_extract():
    df = pl.DataFrame({"s": ["2023-01-15"]})
    assert df.select(pl.col("s").regexp_extract(r"(\d{4})", 1))["s"][0] == "2023"


def test_regexp_extract_all():
    df = pl.DataFrame({"s": ["hello world"]})
    assert df.select(pl.col("s").regexp_extract_all(r"\w+"))["s"][0].to_list() == ["hello", "world"]


def test_regexp_replace():
    df = pl.DataFrame({"s": ["hello"]})
    # replaces first match only
    assert df.select(pl.col("s").regexp_replace("l", "r"))["s"][0] == "herlo"


def test_str_replace():
    df = pl.DataFrame({"s": ["hello"]})
    assert df.select(pl.col("s").str_replace("l", "r"))["s"][0] == "herlo"


def test_str_replace_no_replacement():
    df = pl.DataFrame({"s": ["hello"]})
    # no replacement → delete all occurrences
    assert df.select(pl.col("s").str_replace("l"))["s"][0] == "heo"


def test_translate():
    df = pl.DataFrame({"s": ["abc", "xyz"]})
    result = df.select(pl.col("s").translate("abc", "xyz"))["s"].to_list()
    assert result == ["xyz", "xyz"]


def test_translate_length_mismatch():
    df = pl.DataFrame({"s": ["abc"]})
    with pytest.raises(ValueError):
        df.select(pl.col("s").translate("ab", "xyz"))


# ── date / time Expr methods ──────────────────────────────────────────────────

import datetime


@pytest.fixture()
def date_df():
    # 2023-01-16 is a Monday
    return pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})


@pytest.fixture()
def datetime_df():
    return pl.DataFrame({"t": [datetime.datetime(2023, 1, 16, 14, 30, 0)]})


def test_year(date_df):
    assert date_df.select(pl.col("d").year())["d"][0] == 2023


def test_month(date_df):
    assert date_df.select(pl.col("d").month())["d"][0] == 1


def test_dayofmonth(date_df):
    assert date_df.select(pl.col("d").dayofmonth())["d"][0] == 16


def test_dayofweek_monday(date_df):
    # Spark: Mon=2, Polars weekday()+1
    assert date_df.select(pl.col("d").dayofweek())["d"][0] == 2


def test_dayofyear(date_df):
    assert date_df.select(pl.col("d").dayofyear())["d"][0] == 16


def test_hour(datetime_df):
    assert datetime_df.select(pl.col("t").hour())["t"][0] == 14


def test_last_day(date_df):
    assert date_df.select(pl.col("d").last_day())["d"][0] == datetime.date(2023, 1, 31)


def test_date_add(date_df):
    assert date_df.select(pl.col("d").date_add(5))["d"][0] == datetime.date(2023, 1, 21)


def test_date_sub(date_df):
    assert date_df.select(pl.col("d").date_sub(5))["d"][0] == datetime.date(2023, 1, 11)


def test_add_months(date_df):
    assert date_df.select(pl.col("d").add_months(2))["d"][0] == datetime.date(2023, 3, 16)


# ── hash Expr methods ─────────────────────────────────────────────────────────

def test_md5_empty_string():
    df = pl.DataFrame({"s": [""]})
    assert df.select(pl.col("s").md5())["s"][0] == "d41d8cd98f00b204e9800998ecf8427e"


def test_sha1_empty_string():
    import hashlib
    expected = hashlib.sha1(b"").hexdigest()
    df = pl.DataFrame({"s": [""]})
    assert df.select(pl.col("s").sha1())["s"][0] == expected


def test_sha256_empty_string():
    df = pl.DataFrame({"s": [""]})
    assert df.select(pl.col("s").sha256())["s"][0] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_md5_known_value():
    import hashlib
    val = "hello"
    expected = hashlib.md5(val.encode()).hexdigest()
    df = pl.DataFrame({"s": [val]})
    assert df.select(pl.col("s").md5())["s"][0] == expected


# ── array / list Expr methods ─────────────────────────────────────────────────

@pytest.fixture()
def ints_list_df():
    return pl.DataFrame({"a": [[3, 1, 2, 1], [4, 5, 6]]})


def test_array_distinct():
    df = pl.DataFrame({"a": [[1, 2, 2, 3]]})
    assert sorted(df.select(pl.col("a").array_distinct())["a"][0].to_list()) == [1, 2, 3]


def test_array_compact():
    df = pl.DataFrame({"a": [[1, None, 2]]})
    assert df.select(pl.col("a").array_compact())["a"][0].to_list() == [1, 2]


def test_array_contains_true():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    assert df.select(pl.col("a").array_contains(2))["a"][0] is True


def test_array_contains_false():
    df = pl.DataFrame({"a": [[1, 2, 3]]})
    assert df.select(pl.col("a").array_contains(99))["a"][0] is False


def test_array_append():
    df = pl.DataFrame({"a": [[1, 2]]})
    assert df.select(pl.col("a").array_append(3))["a"][0].to_list() == [1, 2, 3]


def test_array_remove():
    df = pl.DataFrame({"a": [[1, 2, 1, 3]]})
    assert df.select(pl.col("a").array_remove(1))["a"][0].to_list() == [2, 3]


def test_array_max():
    df = pl.DataFrame({"a": [[1, 3, 2]]})
    assert df.select(pl.col("a").array_max())["a"][0] == 3


def test_array_min():
    df = pl.DataFrame({"a": [[1, 3, 2]]})
    assert df.select(pl.col("a").array_min())["a"][0] == 1


def test_array_size(ints_list_df):
    result = ints_list_df.select(pl.col("a").array_size())["a"].to_list()
    assert result == [4, 3]


def test_size_alias(ints_list_df):
    result = ints_list_df.select(pl.col("a").size())["a"].to_list()
    assert result == [4, 3]


def test_array_sort_asc():
    df = pl.DataFrame({"a": [[3, 1, 2]]})
    assert df.select(pl.col("a").array_sort())["a"][0].to_list() == [1, 2, 3]


def test_array_sort_desc():
    df = pl.DataFrame({"a": [[3, 1, 2]]})
    assert df.select(pl.col("a").array_sort(asc=False))["a"][0].to_list() == [3, 2, 1]


def test_array_slice():
    df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
    # Spark 1-based: slice(2, 2) → elements at positions 2,3 → [2, 3]
    assert df.select(pl.col("a").array_slice(2, 2))["a"][0].to_list() == [2, 3]


def test_array_join():
    df = pl.DataFrame({"a": [["a", "b", "c"]]})
    assert df.select(pl.col("a").array_join(","))["a"][0] == "a,b,c"


def test_array_join_null_replacement():
    df = pl.DataFrame({"a": [["a", None, "c"]]})
    assert df.select(pl.col("a").array_join(",", "X"))["a"][0] == "a,X,c"


def test_array_union():
    df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
    result = sorted(df.select(pl.col("a").array_union(pl.col("b")))["a"][0].to_list())
    assert result == [1, 2, 3, 4]


def test_array_intersect():
    df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
    result = sorted(df.select(pl.col("a").array_intersect(pl.col("b")))["a"][0].to_list())
    assert result == [2, 3]


def test_array_except():
    df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3]]})
    result = df.select(pl.col("a").array_except(pl.col("b")))["a"][0].to_list()
    assert result == [1]


def test_list_filter():
    df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
    assert df.select(pl.col("a").list_filter(lambda e: e > 2))["a"][0].to_list() == [3, 4]


def test_forall_true():
    df = pl.DataFrame({"a": [[2, 4, 6]]})
    assert df.select(pl.col("a").forall(lambda e: e > 0))["a"][0] is True


def test_forall_false():
    df = pl.DataFrame({"a": [[2, -1, 6]]})
    assert df.select(pl.col("a").forall(lambda e: e > 0))["a"][0] is False


def test_collect_list():
    df = pl.DataFrame({"x": [1, 2, 3]})
    result = df.select(pl.col("x").collect_list())["x"][0].to_list()
    assert result == [1, 2, 3]


def test_collect_set():
    df = pl.DataFrame({"x": [1, 2, 2, 3]})
    result = sorted(df.select(pl.col("x").collect_set())["x"][0].to_list())
    assert result == [1, 2, 3]


# ── math / aggregation Expr methods ──────────────────────────────────────────

def test_negate():
    df = pl.DataFrame({"x": [1, -2, 3]})
    assert df.select(pl.col("x").negate())["x"].to_list() == [-1, 2, -3]


def test_avg():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert df.select(pl.col("x").avg())["x"][0] == 3.0


# ── math Expr methods ─────────────────────────────────────────────────────────

def test_log2():
    df = pl.DataFrame({"x": [8.0]})
    assert df.select(pl.col("x").log2())["x"][0] == 3.0


def test_log10():
    df = pl.DataFrame({"x": [100.0]})
    assert df.select(pl.col("x").log10())["x"][0] == 2.0


def test_cbrt():
    df = pl.DataFrame({"x": [8.0, 27.0]})
    assert df.select(pl.col("x").cbrt())["x"].to_list() == [2.0, 3.0]


def test_signum():
    df = pl.DataFrame({"x": [3.0, -2.0, 0.0]})
    assert df.select(pl.col("x").signum())["x"].to_list() == [1.0, -1.0, 0.0]


# ── null-handling Expr methods ────────────────────────────────────────────────

def test_nvl():
    df = pl.DataFrame({"a": [None, 2, None]})
    assert df.select(pl.col("a").nvl(-1))["a"].to_list() == [-1, 2, -1]


def test_ifnull():
    df = pl.DataFrame({"a": [None, 2, None]})
    assert df.select(pl.col("a").ifnull(-1))["a"].to_list() == [-1, 2, -1]


def test_nullif():
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert df.select(pl.col("a").nullif(2))["a"].to_list() == [1, None, 3]


# ── date / time Expr methods ──────────────────────────────────────────────────

def test_to_date():
    df = pl.DataFrame({"s": ["2023-06-15"]})
    assert df.select(pl.col("s").to_date())["s"][0] == datetime.date(2023, 6, 15)


def test_to_timestamp():
    df = pl.DataFrame({"s": ["2023-06-15 10:30:00"]})
    result = df.select(pl.col("s").to_timestamp())["s"][0]
    assert result == datetime.datetime(2023, 6, 15, 10, 30, 0)


def test_date_format():
    df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    assert df.select(pl.col("d").date_format("%Y/%m/%d"))["d"][0] == "2023/06/15"


def test_unix_timestamp():
    df = pl.DataFrame({"t": [datetime.datetime(2023, 6, 15, 0, 0, 0)]})
    assert df.select(pl.col("t").unix_timestamp())["t"][0] == 1686787200


# ── array Expr methods ────────────────────────────────────────────────────────

def test_array_position_found():
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    assert df.select(pl.col("a").array_position(20))["a"][0] == 2  # 1-based


def test_array_position_missing():
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    assert df.select(pl.col("a").array_position(99))["a"][0] == 0


def test_array_prepend():
    df = pl.DataFrame({"a": [[10, 20, 30]]})
    assert df.select(pl.col("a").array_prepend(5))["a"][0].to_list() == [5, 10, 20, 30]
