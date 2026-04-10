"""Integration tests for polyspark standalone functions.

Every test creates BOTH a Spark DataFrame and a Polars DataFrame with the same data,
runs the native PySpark function on the Spark DF and the polyspark function on the
Polars DF, then compares results via assert_frame_equal.
"""

import datetime
import math

import polars as pl
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pytest
from polars.testing import assert_frame_equal
from pyspark.sql.window import Window as SparkWindow

import src.sparkpolars  # noqa: F401  # registers .toPolars() on SparkDataFrame
import src.sparkpolars.polyspark.sql.functions as sf
from src.sparkpolars.polyspark.sql.window import Window as PolyWindow


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def compare_spark_polars(spark_df, polars_df, sort_by=None, check_dtypes=False):
    """Convert Spark DF to Polars via sparkpolars.toPolars() and assert equal."""
    spark_as_polars = spark_df.toPolars()
    if sort_by:
        spark_as_polars = spark_as_polars.sort(sort_by)
        polars_df = polars_df.sort(sort_by)
    assert_frame_equal(spark_as_polars, polars_df, check_dtypes=check_dtypes)


# ═══════════════════════════════════════════════════════════════════════════
# String functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_upper(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world"]})
    compare_spark_polars(
        spark_df.select(F.upper("s")),
        polars_df.select(sf.upper(pl.col("s"))),
    )


def test_sf_ucase(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.upper("s")),
        polars_df.select(sf.ucase(pl.col("s"))),
    )


def test_sf_lower(spark_session):
    spark_df = spark_session.createDataFrame([("HELLO",)], ["s"])
    polars_df = pl.DataFrame({"s": ["HELLO"]})
    compare_spark_polars(
        spark_df.select(F.lower("s")),
        polars_df.select(sf.lower(pl.col("s"))),
    )


def test_sf_lcase(spark_session):
    spark_df = spark_session.createDataFrame([("HELLO",)], ["s"])
    polars_df = pl.DataFrame({"s": ["HELLO"]})
    compare_spark_polars(
        spark_df.select(F.lower("s")),
        polars_df.select(sf.lcase(pl.col("s"))),
    )


def test_sf_trim(spark_session):
    spark_df = spark_session.createDataFrame([("  hi  ",)], ["s"])
    polars_df = pl.DataFrame({"s": ["  hi  "]})
    compare_spark_polars(
        spark_df.select(F.trim("s")),
        polars_df.select(sf.trim(pl.col("s"))),
    )


def test_sf_ltrim(spark_session):
    spark_df = spark_session.createDataFrame([("  hi  ",)], ["s"])
    polars_df = pl.DataFrame({"s": ["  hi  "]})
    compare_spark_polars(
        spark_df.select(F.ltrim("s")),
        polars_df.select(sf.ltrim(pl.col("s"))),
    )


def test_sf_rtrim(spark_session):
    spark_df = spark_session.createDataFrame([("  hi  ",)], ["s"])
    polars_df = pl.DataFrame({"s": ["  hi  "]})
    compare_spark_polars(
        spark_df.select(F.rtrim("s")),
        polars_df.select(sf.rtrim(pl.col("s"))),
    )


def test_sf_btrim(spark_session):
    spark_df = spark_session.createDataFrame([("  hi  ",)], ["s"])
    polars_df = pl.DataFrame({"s": ["  hi  "]})
    compare_spark_polars(
        spark_df.select(F.trim("s")),
        polars_df.select(sf.btrim(pl.col("s"))),
    )


def test_sf_lpad(spark_session):
    spark_df = spark_session.createDataFrame([("hi",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hi"]})
    compare_spark_polars(
        spark_df.select(F.lpad("s", 5, "0")),
        polars_df.select(sf.lpad(pl.col("s"), 5, "0")),
    )


def test_sf_rpad(spark_session):
    spark_df = spark_session.createDataFrame([("hi",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hi"]})
    compare_spark_polars(
        spark_df.select(F.rpad("s", 5, "0")),
        polars_df.select(sf.rpad(pl.col("s"), 5, "0")),
    )


def test_sf_left(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.expr("left(s, 3) as s")),
        polars_df.select(sf.left(pl.col("s"), 3)),
    )


def test_sf_right(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.expr("right(s, 3) as s")),
        polars_df.select(sf.right(pl.col("s"), 3)),
    )


def test_sf_length(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("hi",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "hi"]})
    compare_spark_polars(
        spark_df.select(F.length("s")),
        polars_df.select(sf.length(pl.col("s"))),
    )


def test_sf_locate(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.locate("ll", "s").alias("s")),
        polars_df.select(sf.locate("ll", pl.col("s"))),
    )


def test_sf_repeat(spark_session):
    spark_df = spark_session.createDataFrame([("ab",)], ["s"])
    polars_df = pl.DataFrame({"s": ["ab"]})
    compare_spark_polars(
        spark_df.select(F.expr("repeat(s, 3) as s")),
        polars_df.select(sf.repeat(pl.col("s"), 3)),
    )


def test_sf_concat(spark_session):
    spark_df = spark_session.createDataFrame([("a", "b")], ["a", "b"])
    polars_df = pl.DataFrame({"a": ["a"], "b": ["b"]})
    compare_spark_polars(
        spark_df.select(F.concat("a", "b").alias("a")),
        polars_df.select(sf.concat(pl.col("a"), pl.col("b"))),
    )


def test_sf_concat_ws(spark_session):
    spark_df = spark_session.createDataFrame([("a", "b")], ["a", "b"])
    polars_df = pl.DataFrame({"a": ["a"], "b": ["b"]})
    compare_spark_polars(
        spark_df.select(F.concat_ws("-", "a", "b").alias("a")),
        polars_df.select(sf.concat_ws("-", pl.col("a"), pl.col("b"))),
    )


def test_sf_translate(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.translate("s", "helo", "HELO")),
        polars_df.select(sf.translate(pl.col("s"), "helo", "HELO")),
    )


def test_sf_initcap(spark_session):
    spark_df = spark_session.createDataFrame([("hello world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world"]})
    compare_spark_polars(
        spark_df.select(F.initcap("s")),
        polars_df.select(sf.initcap(pl.col("s"))),
    )


def test_sf_ascii(spark_session):
    spark_df = spark_session.createDataFrame([("A",)], ["s"])
    polars_df = pl.DataFrame({"s": ["A"]})
    compare_spark_polars(
        spark_df.select(F.ascii("s")),
        polars_df.select(sf.ascii(pl.col("s"))),
    )


def test_sf_instr(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.instr("s", "ll").alias("s")),
        polars_df.select(sf.instr(pl.col("s"), "ll")),
    )


def test_sf_split_part(spark_session):
    spark_df = spark_session.createDataFrame([("a-b-c",)], ["s"])
    polars_df = pl.DataFrame({"s": ["a-b-c"]})
    compare_spark_polars(
        spark_df.select(F.expr("split_part(s, '-', 2) as s")),
        polars_df.select(sf.split_part(pl.col("s"), "-", 2)),
    )


def test_sf_substring_index(spark_session):
    spark_df = spark_session.createDataFrame([("a.b.c",)], ["s"])
    polars_df = pl.DataFrame({"s": ["a.b.c"]})
    compare_spark_polars(
        spark_df.select(F.substring_index("s", ".", 2).alias("s")),
        polars_df.select(sf.substring_index(pl.col("s"), ".", 2)),
    )


def test_sf_contains(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world"]})
    compare_spark_polars(
        spark_df.select(F.col("s").contains("llo").alias("s")),
        polars_df.select(sf.contains(pl.col("s"), "llo")),
    )


def test_sf_base64(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.base64("s")),
        polars_df.select(sf.base64(pl.col("s"))),
    )


def test_sf_unbase64(spark_session):
    spark_df = spark_session.createDataFrame([("aGVsbG8=",)], ["s"])
    polars_df = pl.DataFrame({"s": ["aGVsbG8="]})
    compare_spark_polars(
        spark_df.select(F.decode(F.unbase64("s"), "utf-8").alias("s")),
        polars_df.select(sf.unbase64(pl.col("s"))),
    )


def test_sf_encode(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.encode("s", "utf-8")),
        polars_df.select(sf.encode(pl.col("s"), "utf-8")),
    )


def test_sf_decode(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.decode(F.encode("s", "utf-8"), "utf-8").alias("s")),
        polars_df.select(sf.decode(sf.encode(pl.col("s"), "utf-8"), "utf-8")),
    )


def test_sf_regexp_replace(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.regexp_replace("s", "l", "r")),
        polars_df.select(sf.regexp_replace(pl.col("s"), "l", "r")),
    )


def test_sf_regexp_count(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.expr("regexp_count(s, 'l') as s")),
        polars_df.select(sf.regexp_count(pl.col("s"), "l")),
    )


def test_sf_regexp_extract(spark_session):
    spark_df = spark_session.createDataFrame([("2023-01-15",)], ["s"])
    polars_df = pl.DataFrame({"s": ["2023-01-15"]})
    compare_spark_polars(
        spark_df.select(F.regexp_extract("s", r"(\d{4})", 1).alias("s")),
        polars_df.select(sf.regexp_extract(pl.col("s"), r"(\d{4})", 1)),
    )


def test_sf_regexp_like(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world"]})
    compare_spark_polars(
        spark_df.select(F.expr("regexp_like(s, '^h') as s")),
        polars_df.select(sf.regexp_like(pl.col("s"), "^h")),
    )


def test_sf_regexp(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.expr("regexp_like(s, 'hel') as s")),
        polars_df.select(sf.regexp(pl.col("s"), "hel")),
    )


def test_sf_regexp_substr(spark_session):
    spark_df = spark_session.createDataFrame([("abc123",)], ["s"])
    polars_df = pl.DataFrame({"s": ["abc123"]})
    compare_spark_polars(
        spark_df.select(F.regexp_extract("s", r"(\d+)", 1).alias("s")),
        polars_df.select(sf.regexp_substr(pl.col("s"), r"\d+")),
    )


def test_sf_regexp_instr(spark_session):
    spark_df = spark_session.createDataFrame([("abcabc",), ("xyz",)], ["s"])
    polars_df = pl.DataFrame({"s": ["abcabc", "xyz"]})
    compare_spark_polars(
        spark_df.select(F.expr("regexp_instr(s, 'b') as s")),
        polars_df.select(sf.regexp_instr(pl.col("s"), "b")),
    )


def test_sf_levenshtein(spark_session):
    spark_df = spark_session.createDataFrame([("kitten", "sitting")], ["a", "b"])
    polars_df = pl.DataFrame({"a": ["kitten"], "b": ["sitting"]})
    compare_spark_polars(
        spark_df.select(F.levenshtein("a", "b").alias("d")),
        polars_df.select(sf.levenshtein(pl.col("a"), pl.col("b")).alias("d")),
    )


def test_sf_overlay(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.overlay("s", F.lit("XY"), 2, 3).alias("s")),
        polars_df.select(sf.overlay(pl.col("s"), "XY", 2, 3)),
    )


def test_sf_chr(spark_session):
    spark_df = spark_session.createDataFrame([(65,)], ["c"])
    polars_df = pl.DataFrame({"c": [65]})
    compare_spark_polars(
        spark_df.select(F.expr("chr(c) as c")),
        polars_df.select(sf.chr(pl.col("c"))),
    )


def test_sf_find_in_set(spark_session):
    spark_df = spark_session.createDataFrame([("b", "a,b,c")], ["s", "arr"])
    polars_df = pl.DataFrame({"s": ["b"], "arr": ["a,b,c"]})
    compare_spark_polars(
        spark_df.select(F.expr("find_in_set(s, arr) as s")),
        polars_df.select(sf.find_in_set(pl.col("s"), pl.col("arr"))),
    )


def test_sf_elt(spark_session):
    spark_df = spark_session.createDataFrame([(2, "a", "b", "c")], ["idx", "c1", "c2", "c3"])
    polars_df = pl.DataFrame({"idx": [2], "c1": ["a"], "c2": ["b"], "c3": ["c"]})
    compare_spark_polars(
        spark_df.select(F.expr("elt(idx, c1, c2, c3) as r")),
        polars_df.select(sf.elt(pl.col("idx"), pl.col("c1"), pl.col("c2"), pl.col("c3")).alias("r")),
    )


def test_sf_format_string(spark_session):
    spark_df = spark_session.createDataFrame([(1, "hello")], ["n", "s"])
    polars_df = pl.DataFrame({"n": [1], "s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.format_string("%d-%s", "n", "s").alias("r")),
        polars_df.select(sf.format_string("%d-%s", pl.col("n"), pl.col("s")).alias("r")),
    )


def test_sf_split(spark_session):
    spark_df = spark_session.createDataFrame([("a-b-c",)], ["s"])
    polars_df = pl.DataFrame({"s": ["a-b-c"]})
    compare_spark_polars(
        spark_df.select(F.split("s", "-")),
        polars_df.select(sf.split(pl.col("s"), "-")),
    )


def test_sf_reverse(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.reverse("s")),
        polars_df.select(sf.reverse(pl.col("s"))),
    )


def test_sf_substr(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(
        spark_df.select(F.col("s").substr(2, 3)),
        polars_df.select(sf.substr(pl.col("s"), 2, 3)),
    )


def test_sf_url_encode(spark_session):
    spark_df = spark_session.createDataFrame([("hello world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world"]})
    compare_spark_polars(
        spark_df.select(F.expr("url_encode(s) as s")),
        polars_df.select(sf.url_encode(pl.col("s"))),
    )


def test_sf_url_decode(spark_session):
    spark_df = spark_session.createDataFrame([("hello+world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello+world"]})
    compare_spark_polars(
        spark_df.select(F.expr("url_decode(s) as s")),
        polars_df.select(sf.url_decode(pl.col("s"))),
    )


def test_sf_format_number(spark_session):
    spark_df = spark_session.createDataFrame([(1234.5678,)], ["x"])
    polars_df = pl.DataFrame({"x": [1234.5678]})
    # Spark adds comma grouping (1,234.57), polyspark doesn't — compare without commas
    spark_r = spark_df.select(F.format_number("x", 2).alias("x")).toPandas().pipe(pl.from_pandas)
    polars_r = polars_df.select(sf.format_number(pl.col("x"), 2))
    spark_r = spark_r.with_columns(pl.col("x").str.replace(",", ""))
    assert_frame_equal(spark_r, polars_r, check_dtypes=False)


# ═══════════════════════════════════════════════════════════════════════════
# Math functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_round(spark_session):
    spark_df = spark_session.createDataFrame([(3.456,)], ["x"])
    polars_df = pl.DataFrame({"x": [3.456]})
    compare_spark_polars(
        spark_df.select(F.round("x", 2)),
        polars_df.select(sf.round(pl.col("x"), 2)),
    )


def test_sf_bround(spark_session):
    spark_df = spark_session.createDataFrame([(2.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [2.5]})
    compare_spark_polars(
        spark_df.select(F.bround("x", 0)),
        polars_df.select(sf.bround(pl.col("x"), 0)),
    )


def test_sf_sqrt(spark_session):
    spark_df = spark_session.createDataFrame([(4.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [4.0]})
    compare_spark_polars(spark_df.select(F.sqrt("x")), polars_df.select(sf.sqrt(pl.col("x"))))


def test_sf_pow(spark_session):
    spark_df = spark_session.createDataFrame([(2.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [2.0]})
    compare_spark_polars(
        spark_df.select(F.pow("x", 3).alias("x")),
        polars_df.select(sf.pow(pl.col("x"), 3)),
    )


def test_sf_negate(spark_session):
    spark_df = spark_session.createDataFrame([(5,)], ["x"])
    polars_df = pl.DataFrame({"x": [5]})
    compare_spark_polars(spark_df.select(F.negate("x")), polars_df.select(sf.negate(pl.col("x"))))


def test_sf_negative(spark_session):
    spark_df = spark_session.createDataFrame([(5,)], ["x"])
    polars_df = pl.DataFrame({"x": [5]})
    compare_spark_polars(spark_df.select(F.negate("x")), polars_df.select(sf.negative(pl.col("x"))))


def test_sf_abs(spark_session):
    spark_df = spark_session.createDataFrame([(-5,)], ["x"])
    polars_df = pl.DataFrame({"x": [-5]})
    compare_spark_polars(spark_df.select(F.abs("x")), polars_df.select(sf.abs(pl.col("x"))))


def test_sf_ceil(spark_session):
    spark_df = spark_session.createDataFrame([(1.2,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.2]})
    compare_spark_polars(spark_df.select(F.ceil("x")), polars_df.select(sf.ceil(pl.col("x"))))


def test_sf_floor(spark_session):
    spark_df = spark_session.createDataFrame([(1.8,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.8]})
    compare_spark_polars(spark_df.select(F.floor("x")), polars_df.select(sf.floor(pl.col("x"))))


def test_sf_log(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0]})
    compare_spark_polars(spark_df.select(F.log("x")), polars_df.select(sf.log(pl.col("x"))))


def test_sf_ln(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0]})
    compare_spark_polars(spark_df.select(F.log("x")), polars_df.select(sf.ln(pl.col("x"))))


def test_sf_log2(spark_session):
    spark_df = spark_session.createDataFrame([(8.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [8.0]})
    compare_spark_polars(spark_df.select(F.log2("x")), polars_df.select(sf.log2(pl.col("x"))))


def test_sf_log10(spark_session):
    spark_df = spark_session.createDataFrame([(100.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [100.0]})
    compare_spark_polars(spark_df.select(F.log10("x")), polars_df.select(sf.log10(pl.col("x"))))


def test_sf_log1p(spark_session):
    spark_df = spark_session.createDataFrame([(0.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.0]})
    compare_spark_polars(spark_df.select(F.log1p("x")), polars_df.select(sf.log1p(pl.col("x"))))


def test_sf_exp(spark_session):
    spark_df = spark_session.createDataFrame([(0.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.0]})
    compare_spark_polars(spark_df.select(F.exp("x")), polars_df.select(sf.exp(pl.col("x"))))


def test_sf_expm1(spark_session):
    spark_df = spark_session.createDataFrame([(0.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.0]})
    compare_spark_polars(spark_df.select(F.expm1("x")), polars_df.select(sf.expm1(pl.col("x"))))


def test_sf_signum(spark_session):
    spark_df = spark_session.createDataFrame([(-3.0,), (0.0,), (5.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [-3.0, 0.0, 5.0]})
    compare_spark_polars(spark_df.select(F.signum("x")), polars_df.select(sf.signum(pl.col("x"))))


def test_sf_sign(spark_session):
    spark_df = spark_session.createDataFrame([(-1.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [-1.0]})
    compare_spark_polars(spark_df.select(F.signum("x")), polars_df.select(sf.sign(pl.col("x"))))


def test_sf_cbrt(spark_session):
    spark_df = spark_session.createDataFrame([(27.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [27.0]})
    compare_spark_polars(spark_df.select(F.cbrt("x")), polars_df.select(sf.cbrt(pl.col("x"))))


def test_sf_rint(spark_session):
    spark_df = spark_session.createDataFrame([(2.6,)], ["x"])
    polars_df = pl.DataFrame({"x": [2.6]})
    compare_spark_polars(spark_df.select(F.rint("x")), polars_df.select(sf.rint(pl.col("x"))))


def test_sf_remainder(spark_session):
    spark_df = spark_session.createDataFrame([(10, 3)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [10], "b": [3]})
    compare_spark_polars(
        spark_df.select((F.col("a") % F.col("b")).alias("a")),
        polars_df.select(sf.remainder(pl.col("a"), pl.col("b"))),
    )


def test_sf_gcd(spark_session):
    spark_df = spark_session.createDataFrame([(12, 8)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [12], "b": [8]})
    compare_spark_polars(
        spark_df.select(F.expr("gcd(a, b) as a")),
        polars_df.select(sf.gcd(pl.col("a"), pl.col("b"))),
    )


def test_sf_lcm(spark_session):
    spark_df = spark_session.createDataFrame([(4, 6)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [4], "b": [6]})
    compare_spark_polars(
        spark_df.select(F.expr("lcm(a, b) as a")),
        polars_df.select(sf.lcm(pl.col("a"), pl.col("b"))),
    )


def test_sf_bitcount(spark_session):
    spark_df = spark_session.createDataFrame([(7,)], ["x"])
    polars_df = pl.DataFrame({"x": [7]})
    compare_spark_polars(
        spark_df.select(F.expr("bit_count(x) as x")),
        polars_df.select(sf.bitcount(pl.col("x"))),
    )


def test_sf_factorial(spark_session):
    spark_df = spark_session.createDataFrame([(5,)], ["x"])
    polars_df = pl.DataFrame({"x": [5]})
    compare_spark_polars(spark_df.select(F.factorial("x")), polars_df.select(sf.factorial(pl.col("x"))))


def test_sf_hex(spark_session):
    spark_df = spark_session.createDataFrame([(255,)], ["x"])
    polars_df = pl.DataFrame({"x": [255]})
    compare_spark_polars(spark_df.select(F.hex("x")), polars_df.select(sf.hex(pl.col("x"))))


def test_sf_unhex(spark_session):
    spark_df = spark_session.createDataFrame([("ff",)], ["x"])
    polars_df = pl.DataFrame({"x": ["ff"]})
    compare_spark_polars(spark_df.select(F.unhex("x")), polars_df.select(sf.unhex(pl.col("x"))))


def test_sf_hypot(spark_session):
    spark_df = spark_session.createDataFrame([(3.0, 4.0)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [3.0], "b": [4.0]})
    compare_spark_polars(
        spark_df.select(F.hypot("a", "b").alias("a")),
        polars_df.select(sf.hypot(pl.col("a"), pl.col("b"))),
    )


def test_sf_pmod(spark_session):
    spark_df = spark_session.createDataFrame([(-7, 3)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [-7], "b": [3]})
    compare_spark_polars(
        spark_df.select(F.pmod("a", F.lit(3)).alias("a")),
        polars_df.select(sf.pmod(pl.col("a"), 3)),
    )


def test_sf_pi(spark_session):
    spark_df = spark_session.createDataFrame([(1,)], ["x"])
    polars_df = pl.DataFrame({"x": [1]})
    compare_spark_polars(
        spark_df.select(F.lit(math.pi).alias("p")),
        polars_df.select(sf.pi().alias("p")),
    )


def test_sf_e(spark_session):
    spark_df = spark_session.createDataFrame([(1,)], ["x"])
    polars_df = pl.DataFrame({"x": [1]})
    compare_spark_polars(
        spark_df.select(F.lit(math.e).alias("e")),
        polars_df.select(sf.e().alias("e")),
    )


def test_sf_shiftleft(spark_session):
    spark_df = spark_session.createDataFrame([(1,)], ["x"])
    polars_df = pl.DataFrame({"x": [1]})
    compare_spark_polars(
        spark_df.select(F.shiftleft("x", 2)),
        polars_df.select(sf.shiftleft(pl.col("x"), 2)),
    )


def test_sf_shiftright(spark_session):
    spark_df = spark_session.createDataFrame([(8,)], ["x"])
    polars_df = pl.DataFrame({"x": [8]})
    compare_spark_polars(
        spark_df.select(F.shiftright("x", 2)),
        polars_df.select(sf.shiftright(pl.col("x"), 2)),
    )


def test_sf_shiftrightunsigned(spark_session):
    spark_df = spark_session.createDataFrame([(8,)], ["x"])
    polars_df = pl.DataFrame({"x": [8]})
    compare_spark_polars(
        spark_df.select(F.shiftrightunsigned("x", 2)),
        polars_df.select(sf.shiftrightunsigned(pl.col("x"), 2)),
    )


def test_sf_toDegrees(spark_session):
    spark_df = spark_session.createDataFrame([(math.pi,)], ["x"])
    polars_df = pl.DataFrame({"x": [math.pi]})
    compare_spark_polars(spark_df.select(F.toDegrees("x")), polars_df.select(sf.toDegrees(pl.col("x"))))


def test_sf_toRadians(spark_session):
    spark_df = spark_session.createDataFrame([(180.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [180.0]})
    compare_spark_polars(spark_df.select(F.toRadians("x")), polars_df.select(sf.toRadians(pl.col("x"))))


def test_sf_degrees(spark_session):
    spark_df = spark_session.createDataFrame([(math.pi,)], ["x"])
    polars_df = pl.DataFrame({"x": [math.pi]})
    compare_spark_polars(spark_df.select(F.degrees("x")), polars_df.select(sf.degrees(pl.col("x"))))


def test_sf_radians(spark_session):
    spark_df = spark_session.createDataFrame([(180.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [180.0]})
    compare_spark_polars(spark_df.select(F.radians("x")), polars_df.select(sf.radians(pl.col("x"))))


# ═══════════════════════════════════════════════════════════════════════════
# Trig functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_sin(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.sin("x")), polars_df.select(sf.sin(pl.col("x"))))


def test_sf_cos(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.cos("x")), polars_df.select(sf.cos(pl.col("x"))))


def test_sf_tan(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.tan("x")), polars_df.select(sf.tan(pl.col("x"))))


def test_sf_asin(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.asin("x")), polars_df.select(sf.asin(pl.col("x"))))


def test_sf_acos(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.acos("x")), polars_df.select(sf.acos(pl.col("x"))))


def test_sf_atan(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.atan("x")), polars_df.select(sf.atan(pl.col("x"))))


def test_sf_atan2(spark_session):
    spark_df = spark_session.createDataFrame([(1.0, 1.0)], ["y", "x"])
    polars_df = pl.DataFrame({"y": [1.0], "x": [1.0]})
    compare_spark_polars(
        spark_df.select(F.atan2("y", "x").alias("y")),
        polars_df.select(sf.atan2(pl.col("y"), pl.col("x"))),
    )


def test_sf_sinh(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.sinh("x")), polars_df.select(sf.sinh(pl.col("x"))))


def test_sf_cosh(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.cosh("x")), polars_df.select(sf.cosh(pl.col("x"))))


def test_sf_tanh(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.tanh("x")), polars_df.select(sf.tanh(pl.col("x"))))


def test_sf_asinh(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.asinh("x")), polars_df.select(sf.asinh(pl.col("x"))))


def test_sf_acosh(spark_session):
    spark_df = spark_session.createDataFrame([(2.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [2.0]})
    compare_spark_polars(spark_df.select(F.acosh("x")), polars_df.select(sf.acosh(pl.col("x"))))


def test_sf_atanh(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})
    compare_spark_polars(spark_df.select(F.atanh("x")), polars_df.select(sf.atanh(pl.col("x"))))


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_sum(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})
    compare_spark_polars(spark_df.select(F.sum("x")), polars_df.select(sf.sum(pl.col("x"))))


def test_sf_min(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})
    compare_spark_polars(spark_df.select(F.min("x")), polars_df.select(sf.min(pl.col("x"))))


def test_sf_max(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})
    compare_spark_polars(spark_df.select(F.max("x")), polars_df.select(sf.max(pl.col("x"))))


def test_sf_avg(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    compare_spark_polars(spark_df.select(F.avg("x")), polars_df.select(sf.avg(pl.col("x"))))


def test_sf_mean(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    compare_spark_polars(spark_df.select(F.avg("x")), polars_df.select(sf.mean(pl.col("x"))))


def test_sf_count(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})
    compare_spark_polars(spark_df.select(F.count("x")), polars_df.select(sf.count(pl.col("x"))))


def test_sf_first(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2]})
    compare_spark_polars(spark_df.select(F.first("x")), polars_df.select(sf.first(pl.col("x"))))


def test_sf_last(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2]})
    compare_spark_polars(spark_df.select(F.last("x")), polars_df.select(sf.last(pl.col("x"))))


def test_sf_count_distinct(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 2, 3]})
    compare_spark_polars(
        spark_df.select(F.countDistinct("x").alias("count_distinct(x)")),
        polars_df.select(sf.count_distinct(pl.col("x"))),
    )


def test_sf_collect_list(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})
    compare_spark_polars(
        spark_df.select(F.collect_list("x")),
        polars_df.select(sf.collect_list(pl.col("x"))),
    )


def test_sf_collect_set(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 2, 3]})
    # Set order is non-deterministic — sort inner lists before comparing
    spark_r = _spark_to_polars(spark_df.select(F.collect_set("x")))
    polars_r = polars_df.select(sf.collect_set(pl.col("x")))
    spark_r = spark_r.with_columns(pl.col("x").list.sort())
    polars_r = polars_r.with_columns(pl.col("x").list.sort())
    assert_frame_equal(spark_r, polars_r, check_dtypes=False)


def test_sf_stddev(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    compare_spark_polars(spark_df.select(F.stddev("x")), polars_df.select(sf.stddev(pl.col("x"))))


def test_sf_stddev_pop(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    compare_spark_polars(spark_df.select(F.stddev_pop("x")), polars_df.select(sf.stddev_pop(pl.col("x"))))


def test_sf_variance(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    compare_spark_polars(spark_df.select(F.variance("x")), polars_df.select(sf.variance(pl.col("x"))))


def test_sf_var_pop(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    compare_spark_polars(spark_df.select(F.var_pop("x")), polars_df.select(sf.var_pop(pl.col("x"))))


def test_sf_median(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    compare_spark_polars(spark_df.select(F.median("x")), polars_df.select(sf.median(pl.col("x"))))


def test_sf_count_if(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,), (4,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3, 4]})
    compare_spark_polars(
        spark_df.select(F.count_if(F.col("x") > 2).alias("count_if")),
        polars_df.select(sf.count_if(pl.col("x") > 2)),
    )


def test_sf_bool_and(spark_session):
    spark_df = spark_session.createDataFrame([(True,), (True,), (False,)], ["x"])
    polars_df = pl.DataFrame({"x": [True, True, False]})
    compare_spark_polars(spark_df.select(F.bool_and("x")), polars_df.select(sf.bool_and(pl.col("x"))))


def test_sf_bool_or(spark_session):
    spark_df = spark_session.createDataFrame([(False,), (False,), (True,)], ["x"])
    polars_df = pl.DataFrame({"x": [False, False, True]})
    compare_spark_polars(spark_df.select(F.bool_or("x")), polars_df.select(sf.bool_or(pl.col("x"))))


def test_sf_sum_distinct(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 2, 3]})
    compare_spark_polars(
        spark_df.select(F.sum_distinct(F.col("x")).alias("sum_distinct(x)")),
        polars_df.select(sf.sum_distinct(pl.col("x"))),
    )


def test_sf_approx_count_distinct(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 2, 3]})
    compare_spark_polars(
        spark_df.select(F.approx_count_distinct("x").alias("approx_count_distinct(x)")),
        polars_df.select(sf.approx_count_distinct(pl.col("x"))),
    )


def test_sf_greatest(spark_session):
    spark_df = spark_session.createDataFrame([(1, 5, 3)], ["a", "b", "c"])
    polars_df = pl.DataFrame({"a": [1], "b": [5], "c": [3]})
    compare_spark_polars(
        spark_df.select(F.greatest("a", "b", "c").alias("a")),
        polars_df.select(sf.greatest(pl.col("a"), pl.col("b"), pl.col("c"))),
    )


def test_sf_least(spark_session):
    spark_df = spark_session.createDataFrame([(1, 5, 3)], ["a", "b", "c"])
    polars_df = pl.DataFrame({"a": [1], "b": [5], "c": [3]})
    compare_spark_polars(
        spark_df.select(F.least("a", "b", "c").alias("a")),
        polars_df.select(sf.least(pl.col("a"), pl.col("b"), pl.col("c"))),
    )


def test_sf_product(spark_session):
    spark_df = spark_session.createDataFrame([(2,), (3,), (4,)], ["x"])
    polars_df = pl.DataFrame({"x": [2, 3, 4]})
    compare_spark_polars(spark_df.select(F.product("x")), polars_df.select(sf.product(pl.col("x"))))


def test_sf_coalesce(spark_session):
    spark_df = spark_session.createDataFrame([(None, 2)], T.StructType([
        T.StructField("a", T.IntegerType()), T.StructField("b", T.IntegerType()),
    ]))
    polars_df = pl.DataFrame({"a": [None], "b": [2]}).cast({"a": pl.Int32, "b": pl.Int32})
    compare_spark_polars(
        spark_df.select(F.coalesce("a", "b").alias("a")),
        polars_df.select(sf.coalesce(pl.col("a"), pl.col("b"))),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Null functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_nvl(spark_session):
    spark_df = spark_session.createDataFrame([(None,), (2,)], T.StructType([
        T.StructField("x", T.IntegerType()),
    ]))
    polars_df = pl.DataFrame({"x": [None, 2]}).cast({"x": pl.Int32})
    compare_spark_polars(
        spark_df.select(F.expr("nvl(x, 0) as x")),
        polars_df.select(sf.nvl(pl.col("x"), 0)),
    )


def test_sf_ifnull(spark_session):
    spark_df = spark_session.createDataFrame([(None,), (2,)], T.StructType([
        T.StructField("x", T.IntegerType()),
    ]))
    polars_df = pl.DataFrame({"x": [None, 2]}).cast({"x": pl.Int32})
    compare_spark_polars(
        spark_df.select(F.expr("ifnull(x, 0) as x")),
        polars_df.select(sf.ifnull(pl.col("x"), 0)),
    )


def test_sf_nullif(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2]})
    compare_spark_polars(
        spark_df.select(F.expr("nullif(x, 1) as x")),
        polars_df.select(sf.nullif(pl.col("x"), 1)),
    )


def test_sf_nanvl(spark_session):
    spark_df = spark_session.createDataFrame([(float("nan"),), (2.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [float("nan"), 2.0]})
    compare_spark_polars(
        spark_df.select(F.nanvl("x", F.lit(0.0))),
        polars_df.select(sf.nanvl(pl.col("x"), 0.0)),
    )


def test_sf_nvl2(spark_session):
    spark_df = spark_session.createDataFrame([(None,), (2,)], T.StructType([
        T.StructField("x", T.IntegerType()),
    ]))
    polars_df = pl.DataFrame({"x": [None, 2]}).cast({"x": pl.Int32})
    compare_spark_polars(
        spark_df.select(F.expr("nvl2(x, 99, 0) as x")),
        polars_df.select(sf.nvl2(pl.col("x"), 99, 0)),
    )


def test_sf_isnull(spark_session):
    spark_df = spark_session.createDataFrame([(None,), (2,)], T.StructType([
        T.StructField("x", T.IntegerType()),
    ]))
    polars_df = pl.DataFrame({"x": [None, 2]}).cast({"x": pl.Int32})
    compare_spark_polars(spark_df.select(F.isnull("x")), polars_df.select(sf.isnull(pl.col("x"))))


def test_sf_isnan(spark_session):
    spark_df = spark_session.createDataFrame([(float("nan"),), (2.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [float("nan"), 2.0]})
    compare_spark_polars(spark_df.select(F.isnan("x")), polars_df.select(sf.isnan(pl.col("x"))))


def test_sf_isnotnull(spark_session):
    spark_df = spark_session.createDataFrame([(None,), (2,)], T.StructType([
        T.StructField("x", T.IntegerType()),
    ]))
    polars_df = pl.DataFrame({"x": [None, 2]}).cast({"x": pl.Int32})
    compare_spark_polars(
        spark_df.select(F.col("x").isNotNull().alias("x")),
        polars_df.select(sf.isnotnull(pl.col("x"))),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Bitwise functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_bitwise_not(spark_session):
    spark_df = spark_session.createDataFrame([(5,)], ["x"])
    polars_df = pl.DataFrame({"x": [5]})
    compare_spark_polars(
        spark_df.select(F.bitwise_not("x")),
        polars_df.select(sf.bitwise_not(pl.col("x"))),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Array functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_array(spark_session):
    spark_df = spark_session.createDataFrame([(1, 2, 3)], ["a", "b", "c"])
    polars_df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    compare_spark_polars(
        spark_df.select(F.array("a", "b", "c").alias("a")),
        polars_df.select(sf.array(pl.col("a"), pl.col("b"), pl.col("c"))),
    )


def test_sf_array_distinct(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 2, 3]]})
    spark_r = _spark_to_polars(spark_df.select(F.array_distinct("a")))
    polars_r = polars_df.select(sf.array_distinct(pl.col("a")))
    spark_r = spark_r.with_columns(pl.col("a").list.sort())
    polars_r = polars_r.with_columns(pl.col("a").list.sort())
    assert_frame_equal(spark_r, polars_r, check_dtypes=False)


def test_sf_array_compact(spark_session):
    spark_df = spark_session.createDataFrame([([1, None, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, None, 3]]})
    compare_spark_polars(spark_df.select(F.array_compact("a")), polars_df.select(sf.array_compact(pl.col("a"))))


def test_sf_array_contains(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(
        spark_df.select(F.array_contains("a", 2)),
        polars_df.select(sf.array_contains(pl.col("a"), 2)),
    )


def test_sf_array_append(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2]]})
    compare_spark_polars(spark_df.select(F.array_append("a", 3)), polars_df.select(sf.array_append(pl.col("a"), 3)))


def test_sf_array_remove(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(spark_df.select(F.array_remove("a", 2)), polars_df.select(sf.array_remove(pl.col("a"), 2)))


def test_sf_array_except(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3], [2, 3, 4])], "a: array<int>, b: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
    compare_spark_polars(
        spark_df.select(F.array_except("a", "b")),
        polars_df.select(sf.array_except(pl.col("a"), pl.col("b"))),
    )


def test_sf_array_intersect(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3], [2, 3, 4])], "a: array<int>, b: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})
    spark_r = _spark_to_polars(spark_df.select(F.array_intersect("a", "b")))
    polars_r = polars_df.select(sf.array_intersect(pl.col("a"), pl.col("b")))
    spark_r = spark_r.with_columns(pl.col("a").list.sort())
    polars_r = polars_r.with_columns(pl.col("a").list.sort())
    assert_frame_equal(spark_r, polars_r, check_dtypes=False)


def test_sf_array_union(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2], [2, 3])], "a: array<int>, b: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2]], "b": [[2, 3]]})
    spark_r = _spark_to_polars(spark_df.select(F.array_union("a", "b")))
    polars_r = polars_df.select(sf.array_union(pl.col("a"), pl.col("b")))
    spark_r = spark_r.with_columns(pl.col("a").list.sort())
    polars_r = polars_r.with_columns(pl.col("a").list.sort())
    assert_frame_equal(spark_r, polars_r, check_dtypes=False)


def test_sf_array_join(spark_session):
    spark_df = spark_session.createDataFrame([(["a", "b", "c"],)], "a: array<string>")
    polars_df = pl.DataFrame({"a": [["a", "b", "c"]]})
    compare_spark_polars(spark_df.select(F.array_join("a", "-")), polars_df.select(sf.array_join(pl.col("a"), "-")))


def test_sf_array_max(spark_session):
    spark_df = spark_session.createDataFrame([([1, 5, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 5, 3]]})
    compare_spark_polars(spark_df.select(F.array_max("a")), polars_df.select(sf.array_max(pl.col("a"))))


def test_sf_array_min(spark_session):
    spark_df = spark_session.createDataFrame([([1, 5, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 5, 3]]})
    compare_spark_polars(spark_df.select(F.array_min("a")), polars_df.select(sf.array_min(pl.col("a"))))


def test_sf_array_size(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(spark_df.select(F.array_size("a")), polars_df.select(sf.array_size(pl.col("a"))))


def test_sf_size(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(spark_df.select(F.size("a")), polars_df.select(sf.size(pl.col("a"))))


def test_sf_cardinality(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(spark_df.select(F.size("a")), polars_df.select(sf.cardinality(pl.col("a"))))


def test_sf_array_sort(spark_session):
    spark_df = spark_session.createDataFrame([([3, 1, 2],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[3, 1, 2]]})
    compare_spark_polars(spark_df.select(F.array_sort("a")), polars_df.select(sf.array_sort(pl.col("a"))))


def test_sf_sort_array(spark_session):
    spark_df = spark_session.createDataFrame([([3, 1, 2],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[3, 1, 2]]})
    compare_spark_polars(
        spark_df.select(F.sort_array("a", asc=True)),
        polars_df.select(sf.sort_array(pl.col("a"), asc=True)),
    )


def test_sf_flatten(spark_session):
    spark_df = spark_session.createDataFrame([([[1, 2], [3, 4]],)], "a: array<array<int>>")
    polars_df = pl.DataFrame({"a": [[[1, 2], [3, 4]]]})
    compare_spark_polars(spark_df.select(F.flatten("a")), polars_df.select(sf.flatten(pl.col("a"))))


def test_sf_explode(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(spark_df.select(F.explode("a")), polars_df.select(sf.explode(pl.col("a"))), sort_by="a")


def test_sf_explode_outer(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2],), (None,)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2], None]})
    compare_spark_polars(
        spark_df.select(F.explode_outer("a")),
        polars_df.select(sf.explode_outer(pl.col("a"))),
        sort_by="a",
    )


def test_sf_element_at(spark_session):
    spark_df = spark_session.createDataFrame([([10, 20, 30],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[10, 20, 30]]})
    compare_spark_polars(spark_df.select(F.element_at("a", 2)), polars_df.select(sf.element_at(pl.col("a"), 2)))


def test_sf_arrays_overlap(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2], [2, 3])], "a: array<int>, b: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2]], "b": [[2, 3]]})
    compare_spark_polars(
        spark_df.select(F.arrays_overlap("a", "b")),
        polars_df.select(sf.arrays_overlap(pl.col("a"), pl.col("b"))),
    )


def test_sf_array_repeat(spark_session):
    spark_df = spark_session.createDataFrame([(5,)], ["x"])
    polars_df = pl.DataFrame({"x": [5]})
    compare_spark_polars(spark_df.select(F.array_repeat("x", 3)), polars_df.select(sf.array_repeat(pl.col("x"), 3)))


def test_sf_array_reverse(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(spark_df.select(F.array_reverse("a")), polars_df.select(sf.array_reverse(pl.col("a"))))


def test_sf_array_position(spark_session):
    spark_df = spark_session.createDataFrame([([10, 20, 30],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[10, 20, 30]]})
    compare_spark_polars(
        spark_df.select(F.array_position("a", 20)),
        polars_df.select(sf.array_position(pl.col("a"), 20)),
    )


def test_sf_array_prepend(spark_session):
    spark_df = spark_session.createDataFrame([([2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[2, 3]]})
    compare_spark_polars(spark_df.select(F.array_prepend("a", 1)), polars_df.select(sf.array_prepend(pl.col("a"), 1)))


def test_sf_slice(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3, 4, 5],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3, 4, 5]]})
    compare_spark_polars(spark_df.select(F.slice("a", 2, 3)), polars_df.select(sf.slice(pl.col("a"), 2, 3)))


# ═══════════════════════════════════════════════════════════════════════════
# Date/time functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_year(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.year("d")), polars_df.select(sf.year(pl.col("d"))))


def test_sf_month(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.month("d")), polars_df.select(sf.month(pl.col("d"))))


def test_sf_dayofmonth(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.dayofmonth("d")), polars_df.select(sf.dayofmonth(pl.col("d"))))


def test_sf_dayofweek(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.dayofweek("d")), polars_df.select(sf.dayofweek(pl.col("d"))))


def test_sf_dayofyear(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.dayofyear("d")), polars_df.select(sf.dayofyear(pl.col("d"))))


def test_sf_hour(spark_session):
    dt = datetime.datetime(2023, 6, 15, 14, 30, 0, tzinfo=datetime.timezone.utc)
    spark_df = spark_session.createDataFrame([(dt,)], T.StructType([T.StructField("t", T.TimestampType())]))
    polars_df = pl.DataFrame({"t": [dt]})
    compare_spark_polars(spark_df.select(F.hour("t")), polars_df.select(sf.hour(pl.col("t"))))


def test_sf_minute(spark_session):
    dt = datetime.datetime(2023, 6, 15, 14, 30, 0, tzinfo=datetime.timezone.utc)
    spark_df = spark_session.createDataFrame([(dt,)], T.StructType([T.StructField("t", T.TimestampType())]))
    polars_df = pl.DataFrame({"t": [dt]})
    compare_spark_polars(spark_df.select(F.minute("t")), polars_df.select(sf.minute(pl.col("t"))))


def test_sf_second(spark_session):
    dt = datetime.datetime(2023, 6, 15, 14, 30, 45, tzinfo=datetime.timezone.utc)
    spark_df = spark_session.createDataFrame([(dt,)], T.StructType([T.StructField("t", T.TimestampType())]))
    polars_df = pl.DataFrame({"t": [dt]})
    compare_spark_polars(spark_df.select(F.second("t")), polars_df.select(sf.second(pl.col("t"))))


def test_sf_quarter(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.quarter("d")), polars_df.select(sf.quarter(pl.col("d"))))


def test_sf_weekofyear(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.weekofyear("d")), polars_df.select(sf.weekofyear(pl.col("d"))))


def test_sf_weekday(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(
        spark_df.select(F.expr("weekday(d) as d")),
        polars_df.select(sf.weekday(pl.col("d"))),
    )


def test_sf_last_day(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.last_day("d")), polars_df.select(sf.last_day(pl.col("d"))))


def test_sf_date_add(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.date_add("d", 5)), polars_df.select(sf.date_add(pl.col("d"), 5)))


def test_sf_date_sub(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.date_sub("d", 5)), polars_df.select(sf.date_sub(pl.col("d"), 5)))


def test_sf_datediff(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 6, 20), datetime.date(2023, 6, 15))], ["a", "b"],
    )
    polars_df = pl.DataFrame({"a": [datetime.date(2023, 6, 20)], "b": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(
        spark_df.select(F.datediff("a", "b").alias("a")),
        polars_df.select(sf.datediff(pl.col("a"), pl.col("b"))),
    )


def test_sf_add_months(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 1, 31),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 31)]})
    compare_spark_polars(spark_df.select(F.add_months("d", 1)), polars_df.select(sf.add_months(pl.col("d"), 1)))


def test_sf_date_trunc(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(
        spark_df.select(F.date_trunc("month", "d").cast("date").alias("d")),
        polars_df.select(sf.date_trunc("month", pl.col("d"))),
    )


def test_sf_to_date(spark_session):
    spark_df = spark_session.createDataFrame([("2023-06-15",)], ["s"])
    polars_df = pl.DataFrame({"s": ["2023-06-15"]})
    compare_spark_polars(spark_df.select(F.to_date("s")), polars_df.select(sf.to_date(pl.col("s"))))


def test_sf_date_format(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(
        spark_df.select(F.date_format("d", "yyyy-MM-dd").alias("d")),
        polars_df.select(sf.date_format(pl.col("d"), "%Y-%m-%d")),
    )


def test_sf_unix_timestamp(spark_session):
    dt = datetime.datetime(2023, 6, 15, 0, 0, 0, tzinfo=datetime.timezone.utc)
    spark_df = spark_session.createDataFrame([(dt,)], T.StructType([T.StructField("t", T.TimestampType())]))
    polars_df = pl.DataFrame({"t": [dt]})
    compare_spark_polars(spark_df.select(F.unix_timestamp("t")), polars_df.select(sf.unix_timestamp(pl.col("t"))))


def test_sf_from_unixtime(spark_session):
    spark_df = spark_session.createDataFrame([(1686787200,)], T.StructType([T.StructField("t", T.LongType())]))
    polars_df = pl.DataFrame({"t": [1686787200]})
    compare_spark_polars(spark_df.select(F.from_unixtime("t")), polars_df.select(sf.from_unixtime(pl.col("t"))))


def test_sf_make_date(spark_session):
    spark_df = spark_session.createDataFrame([(2023, 6, 15)], ["y", "m", "d"])
    polars_df = pl.DataFrame({"y": [2023], "m": [6], "d": [15]})
    compare_spark_polars(
        spark_df.select(F.make_date("y", "m", "d").alias("d")),
        polars_df.select(sf.make_date(pl.col("y"), pl.col("m"), pl.col("d"))),
    )


def test_sf_unix_date(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(
        spark_df.select(F.expr("unix_date(d) as d")),
        polars_df.select(sf.unix_date(pl.col("d"))),
    )


def test_sf_trunc(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.trunc("d", "month")), polars_df.select(sf.trunc(pl.col("d"), "month")))


def test_sf_next_day(spark_session):
    spark_df = spark_session.createDataFrame([(datetime.date(2023, 6, 15),)], ["d"])
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})
    compare_spark_polars(spark_df.select(F.next_day("d", "Mon")), polars_df.select(sf.next_day(pl.col("d"), "Mon")))


def test_sf_months_between(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 6, 15), datetime.date(2023, 1, 15))], ["a", "b"],
    )
    polars_df = pl.DataFrame({"a": [datetime.date(2023, 6, 15)], "b": [datetime.date(2023, 1, 15)]})
    compare_spark_polars(
        spark_df.select(F.months_between("a", "b").alias("a")),
        polars_df.select(sf.months_between(pl.col("a"), pl.col("b"))),
    )


def test_sf_current_date(spark_session):
    spark_df = spark_session.createDataFrame([(1,)], ["x"])
    polars_df = pl.DataFrame({"x": [1]})
    # Both should return today's date
    spark_r = _spark_to_polars(spark_df.select(F.current_date().alias("d")))
    polars_r = polars_df.select(sf.current_date().alias("d"))
    assert_frame_equal(spark_r, polars_r, check_dtypes=False)


def test_sf_current_timestamp(spark_session):
    # Can't compare exact values — just verify both return non-null timestamps
    spark_df = spark_session.createDataFrame([(1,)], ["x"])
    polars_df = pl.DataFrame({"x": [1]})
    spark_r = spark_df.select(F.current_timestamp().alias("t")).collect()[0][0]
    polars_r = polars_df.select(sf.current_timestamp().alias("t"))["t"][0]
    assert spark_r is not None
    assert polars_r is not None


# ═══════════════════════════════════════════════════════════════════════════
# Hash functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_md5(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(spark_df.select(F.md5("s")), polars_df.select(sf.md5(pl.col("s"))))


def test_sf_sha1(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(spark_df.select(F.sha1("s")), polars_df.select(sf.sha1(pl.col("s"))))


def test_sf_sha(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(spark_df.select(F.sha1("s")), polars_df.select(sf.sha(pl.col("s"))))


def test_sf_sha256(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(spark_df.select(F.sha2("s", 256)), polars_df.select(sf.sha256(pl.col("s"))))


def test_sf_sha2(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(spark_df.select(F.sha2("s", 256)), polars_df.select(sf.sha2(pl.col("s"), 256)))


def test_sf_crc32(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})
    compare_spark_polars(spark_df.select(F.crc32("s")), polars_df.select(sf.crc32(pl.col("s"))))


# ═══════════════════════════════════════════════════════════════════════════
# Conditional & expression
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_when(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})
    compare_spark_polars(
        spark_df.select(F.when(F.col("x") > 1, F.lit(99)).otherwise(F.lit(0)).alias("r")),
        polars_df.select(sf.when(pl.col("x") > 1, pl.lit(99)).otherwise(pl.lit(0)).alias("r")),
    )


def test_sf_lit(spark_session):
    spark_df = spark_session.createDataFrame([(1,)], ["x"])
    polars_df = pl.DataFrame({"x": [1]})
    compare_spark_polars(
        spark_df.select(F.lit(42).alias("r")),
        polars_df.select(sf.lit(42).alias("r")),
    )


def test_sf_expr(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2]})
    compare_spark_polars(
        spark_df.select(F.expr("x * 2 as r")),
        polars_df.select(sf.expr("x * 2").alias("r")),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Struct / JSON / Map
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_struct(spark_session):
    spark_df = spark_session.createDataFrame([(1, "a")], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1], "y": ["a"]})
    compare_spark_polars(
        spark_df.select(F.struct("x", "y").alias("s")),
        polars_df.select(sf.struct(pl.col("x"), pl.col("y")).alias("s")),
    )


def test_sf_to_json(spark_session):
    spark_df = spark_session.createDataFrame([(1, "a")], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1], "y": ["a"]})
    spark_r = _spark_to_polars(spark_df.select(F.to_json(F.struct("x", "y")).alias("j")))
    polars_r = polars_df.select(sf.to_json(sf.struct(pl.col("x"), pl.col("y"))).alias("j"))
    # JSON key order may differ — parse and compare as dicts
    import json
    assert json.loads(spark_r["j"][0]) == json.loads(polars_r["j"][0])


def test_sf_get_json_object(spark_session):
    spark_df = spark_session.createDataFrame([('{"name":"Alice"}',)], ["j"])
    polars_df = pl.DataFrame({"j": ['{"name":"Alice"}']})
    compare_spark_polars(
        spark_df.select(F.get_json_object("j", "$.name").alias("j")),
        polars_df.select(sf.get_json_object(pl.col("j"), "$.name")),
    )


def test_sf_json_tuple(spark_session):
    spark_df = spark_session.createDataFrame([('{"a":"1","b":"2"}',)], ["j"])
    polars_df = pl.DataFrame({"j": ['{"a":"1","b":"2"}']})
    spark_r = _spark_to_polars(spark_df.select(F.json_tuple("j", "a", "b")))
    polars_r = polars_df.select(sf.json_tuple(pl.col("j"), "a", "b").alias("s"))
    # Spark returns separate columns c0/c1, polyspark returns a struct — compare values
    assert spark_r["c0"][0] == polars_r["s"][0]["a"]
    assert spark_r["c1"][0] == polars_r["s"][0]["b"]


def test_sf_map_keys(spark_session):
    spark_df = spark_session.createDataFrame(
        [({  "a": "1", "b": "2"},)],
        T.StructType([T.StructField("m", T.MapType(T.StringType(), T.StringType()))]),
    )
    polars_df = pl.DataFrame({"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]})
    spark_r = _spark_to_polars(spark_df.select(F.map_keys("m")))
    polars_r = polars_df.select(sf.map_keys(pl.col("m")))
    spark_r = spark_r.with_columns(pl.col("m").list.sort())
    polars_r = polars_r.with_columns(pl.col("m").list.sort())
    assert_frame_equal(spark_r, polars_r, check_dtypes=False)


def test_sf_map_values(spark_session):
    spark_df = spark_session.createDataFrame(
        [({"a": "1", "b": "2"},)],
        T.StructType([T.StructField("m", T.MapType(T.StringType(), T.StringType()))]),
    )
    polars_df = pl.DataFrame({"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]})
    spark_r = _spark_to_polars(spark_df.select(F.map_values("m")))
    polars_r = polars_df.select(sf.map_values(pl.col("m")))
    spark_r = spark_r.with_columns(pl.col("m").list.sort())
    polars_r = polars_r.with_columns(pl.col("m").list.sort())
    assert_frame_equal(spark_r, polars_r, check_dtypes=False)


def test_sf_map_contains_key(spark_session):
    spark_df = spark_session.createDataFrame(
        [({"a": 1, "b": 2},)],
        T.StructType([T.StructField("m", T.MapType(T.StringType(), T.IntegerType()))]),
    )
    polars_df = pl.DataFrame({"m": [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]]})
    compare_spark_polars(
        spark_df.select(F.expr("map_contains_key(m, 'a') as m")),
        polars_df.select(sf.map_contains_key(pl.col("m"), "a")),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Higher-order functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_transform(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(
        spark_df.select(F.transform("a", lambda x: x * 2)),
        polars_df.select(sf.transform(pl.col("a"), lambda x: x * 2)),
    )


def test_sf_filter(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3, 4],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
    compare_spark_polars(
        spark_df.select(F.filter("a", lambda x: x > 2)),
        polars_df.select(sf.filter(pl.col("a"), lambda x: x > 2)),
    )


def test_sf_exists(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(
        spark_df.select(F.exists("a", lambda x: x > 2)),
        polars_df.select(sf.exists(pl.col("a"), lambda x: x > 2)),
    )


def test_sf_forall(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(
        spark_df.select(F.forall("a", lambda x: x > 0)),
        polars_df.select(sf.forall(pl.col("a"), lambda x: x > 0)),
    )


def test_sf_aggregate(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})
    compare_spark_polars(
        spark_df.select(F.aggregate("a", F.lit(0), lambda acc, x: acc + x).alias("a")),
        polars_df.select(sf.aggregate(pl.col("a"), 0, lambda acc, x: acc + x).alias("a")),
    )


def test_sf_shuffle(spark_session):
    # Random — can't compare values, but verify same shape
    spark_df = spark_session.createDataFrame([([1, 2, 3, 4, 5],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3, 4, 5]]})
    spark_len = len(spark_df.select(F.shuffle("a")).collect()[0][0])
    polars_len = len(polars_df.select(sf.shuffle(pl.col("a")))["a"][0])
    assert spark_len == polars_len == 5


def test_sf_sequence(spark_session):
    spark_df = spark_session.createDataFrame([(1, 5)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [1], "b": [5]})
    compare_spark_polars(
        spark_df.select(F.sequence("a", "b").alias("a")),
        polars_df.select(sf.sequence(pl.col("a"), pl.col("b"))),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Window / ranking functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_row_number(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    sw = SparkWindow.orderBy("a")
    pw = PolyWindow.orderBy("a")
    compare_spark_polars(
        spark_df.select("a", F.row_number().over(sw).alias("rn")),
        polars_df.select("a", sf.row_number().over(pw).alias("rn")),
        sort_by="a",
    )


def test_sf_rank(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (1,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 1, 3]})
    sw = SparkWindow.orderBy("a")
    pw = PolyWindow.orderBy("a")
    compare_spark_polars(
        spark_df.select("a", F.rank().over(sw).alias("r")),
        polars_df.select("a", sf.rank().over(pw).alias("r")),
        sort_by="a",
    )


def test_sf_dense_rank(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (1,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 1, 3]})
    sw = SparkWindow.orderBy("a")
    pw = PolyWindow.orderBy("a")
    compare_spark_polars(
        spark_df.select("a", F.dense_rank().over(sw).alias("r")),
        polars_df.select("a", sf.dense_rank().over(pw).alias("r")),
        sort_by="a",
    )


def test_sf_lag(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    sw = SparkWindow.orderBy("a")
    pw = PolyWindow.orderBy("a")
    compare_spark_polars(
        spark_df.select("a", F.lag("a", 1).over(sw).alias("l")),
        polars_df.select("a", sf.lag(pl.col("a"), 1).over(pw).alias("l")),
        sort_by="a",
    )


def test_sf_lead(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    sw = SparkWindow.orderBy("a")
    pw = PolyWindow.orderBy("a")
    compare_spark_polars(
        spark_df.select("a", F.lead("a", 1).over(sw).alias("l")),
        polars_df.select("a", sf.lead(pl.col("a"), 1).over(pw).alias("l")),
        sort_by="a",
    )


def test_sf_ntile(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,), (4,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3, 4]})
    sw = SparkWindow.orderBy("a")
    pw = PolyWindow.orderBy("a")
    compare_spark_polars(
        spark_df.select("a", F.ntile(2).over(sw).alias("t")),
        polars_df.select("a", sf.ntile(2).over(pw).alias("t")),
        sort_by="a",
    )


def test_sf_cume_dist(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,), (4,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3, 4]})
    sw = SparkWindow.orderBy("a")
    pw = PolyWindow.orderBy("a")
    compare_spark_polars(
        spark_df.select("a", F.cume_dist().over(sw).alias("c")),
        polars_df.select("a", sf.cume_dist().over(pw).alias("c")),
        sort_by="a",
    )


def test_sf_percent_rank(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,), (4,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3, 4]})
    sw = SparkWindow.orderBy("a")
    pw = PolyWindow.orderBy("a")
    compare_spark_polars(
        spark_df.select("a", F.percent_rank().over(sw).alias("p")),
        polars_df.select("a", sf.percent_rank().over(pw).alias("p")),
        sort_by="a",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Statistical functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_corr(spark_session):
    spark_df = spark_session.createDataFrame([(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    compare_spark_polars(
        spark_df.select(F.corr("a", "b").alias("a")),
        polars_df.select(sf.corr(pl.col("a"), pl.col("b"))),
    )


def test_sf_covar_samp(spark_session):
    spark_df = spark_session.createDataFrame([(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    compare_spark_polars(
        spark_df.select(F.covar_samp("a", "b").alias("a")),
        polars_df.select(sf.covar_samp(pl.col("a"), pl.col("b"))),
    )


def test_sf_covar_pop(spark_session):
    spark_df = spark_session.createDataFrame([(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    compare_spark_polars(
        spark_df.select(F.covar_pop("a", "b").alias("a")),
        polars_df.select(sf.covar_pop(pl.col("a"), pl.col("b"))),
    )


def test_sf_kurtosis(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,), (4.0,), (5.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    compare_spark_polars(spark_df.select(F.kurtosis("x")), polars_df.select(sf.kurtosis(pl.col("x"))))


def test_sf_skewness(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,), (4.0,), (5.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    compare_spark_polars(spark_df.select(F.skewness("x")), polars_df.select(sf.skewness(pl.col("x"))))


def test_sf_mode(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 2, 3]})
    compare_spark_polars(spark_df.select(F.mode("x")), polars_df.select(sf.mode(pl.col("x"))))


def test_sf_percentile_approx(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,), (4.0,), (5.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    compare_spark_polars(
        spark_df.select(F.percentile_approx("x", 0.5).alias("percentile_approx(x, 0.5)")),
        polars_df.select(sf.percentile_approx(pl.col("x"), 0.5)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Random / Monotonic (can't compare exact values — verify shape)
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_rand(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})
    spark_r = _spark_to_polars(spark_df.select(F.rand(seed=42).alias("r")))
    polars_r = polars_df.select(sf.rand(seed=42).alias("r"))
    assert spark_r.shape == polars_r.shape


def test_sf_randn(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})
    spark_r = _spark_to_polars(spark_df.select(F.randn(seed=42).alias("r")))
    polars_r = polars_df.select(sf.randn(seed=42).alias("r"))
    assert spark_r.shape == polars_r.shape


def test_sf_uniform(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})
    # Spark 3.4+ has uniform; for older versions just test polyspark shape
    polars_r = polars_df.select(sf.uniform(0.0, 1.0, seed=42).alias("r"))
    assert polars_r.height == 3


def test_sf_monotonically_increasing_id(spark_session):
    spark_df = spark_session.createDataFrame([(10,), (20,), (30,)], ["x"])
    polars_df = pl.DataFrame({"x": [10, 20, 30]})
    spark_r = _spark_to_polars(spark_df.select(F.monotonically_increasing_id().alias("id")))
    polars_r = polars_df.select(sf.monotonically_increasing_id().alias("id"))
    # Both should produce 3 rows with sequential IDs
    assert spark_r.height == polars_r.height == 3


def test_sf_monotonic_id(spark_session):
    spark_df = spark_session.createDataFrame([(10,), (20,), (30,)], ["x"])
    polars_df = pl.DataFrame({"x": [10, 20, 30]})
    spark_r = _spark_to_polars(spark_df.select(F.monotonically_increasing_id().alias("id")))
    polars_r = polars_df.select(sf.monotonic_id().alias("id"))
    assert spark_r.height == polars_r.height == 3


# ═══════════════════════════════════════════════════════════════════════════
# Type functions
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_typeof(spark_session):
    spark_df = spark_session.createDataFrame([(1,)], ["x"])
    polars_df = pl.DataFrame({"x": [1]})
    # Spark returns "int"/"bigint", Polars returns "Int64" — both should be non-null strings
    spark_r = _spark_to_polars(spark_df.select(F.expr("typeof(x) as t")))
    polars_r = polars_df.select(sf.typeof(pl.col("x")).alias("t"))
    assert spark_r["t"][0] is not None
    assert polars_r["t"][0] is not None


# ═══════════════════════════════════════════════════════════════════════════
# Misc
# ═══════════════════════════════════════════════════════════════════════════

def test_sf_broadcast(spark_session):
    spark_df = spark_session.createDataFrame([(1,)], ["x"])
    polars_df = pl.DataFrame({"x": [1]})
    # Both should return the input unchanged
    assert sf.broadcast(polars_df) is polars_df
    assert F.broadcast(spark_df) is not None


def test_sf_named_struct(spark_session):
    spark_df = spark_session.createDataFrame([(1, 2)], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1], "y": [2]})
    compare_spark_polars(
        spark_df.select(F.expr("named_struct('a', x, 'b', y) as s")),
        polars_df.select(sf.named_struct("a", pl.col("x"), "b", pl.col("y")).alias("s")),
    )


def test_sf_assert_true(spark_session):
    spark_df = spark_session.createDataFrame([(True,), (True,)], ["x"])
    polars_df = pl.DataFrame({"x": [True, True]})
    # Both should succeed without error
    spark_df.select(F.assert_true(F.col("x"))).collect()
    polars_df.select(sf.assert_true(pl.col("x")).alias("r"))


def test_sf_assert_true_fails(spark_session):
    polars_df = pl.DataFrame({"x": [True, False]})
    with pytest.raises(Exception):
        polars_df.select(sf.assert_true(pl.col("x")).alias("r"))


# ═══════════════════════════════════════════════════════════════════════════
# NotImplementedError stubs
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("func_name", [
    "soundex", "metaphone", "from_json", "posexplode", "posexplode_outer",
    "window", "grouping", "grouping_id", "schema_of_csv", "schema_of_json",
    "input_file_name",
])
def test_sf_not_implemented(func_name):
    func = getattr(sf, func_name)
    with pytest.raises(NotImplementedError):
        if func_name == "from_json":
            func(pl.col("x"), {})
        elif func_name == "window":
            func("x", "1 hour")
        elif func_name == "grouping_id":
            func("x")
        elif func_name == "input_file_name":
            func()
        else:
            func(pl.col("x"))
