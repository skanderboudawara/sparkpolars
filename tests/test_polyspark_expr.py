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
