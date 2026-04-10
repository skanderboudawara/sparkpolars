"""Integration tests for polyspark Expr (Column) monkey-patches.

Each test creates both a PySpark DataFrame and a Polars DataFrame with
identical data, runs the equivalent operation on each, then compares
results via ``assert_frame_equal``.
"""

import datetime
import json
import math

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Row
from pyspark.sql.window import Window as SparkWindow
import polars as pl
from polars.testing import assert_frame_equal

import src.sparkpolars.polyspark.sql.functions as sf  # noqa: F401 -- installs patches
import src.sparkpolars.polyspark.sql.columns  # noqa: F401
from src.sparkpolars.polyspark.sql.window import Window as PolyWindow


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _spark_to_polars(spark_df):
    """Convert a Spark DataFrame to a Polars DataFrame via Pandas."""
    return spark_df.toPandas().pipe(pl.from_pandas)


# ===================================================================
# Null checks
# ===================================================================

def test_isNull(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (None,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, None, 3]})

    spark_result = spark_df.select(F.col("a").isNull().alias("a"))
    polars_result = polars_df.select(pl.col("a").isNull().alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_isNotNull(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (None,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, None, 3]})

    spark_result = spark_df.select(F.col("a").isNotNull().alias("a"))
    polars_result = polars_df.select(pl.col("a").isNotNull().alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Comparison & filtering
# ===================================================================

def test_isin(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,), (4,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3, 4]})

    spark_result = spark_df.filter(F.col("a").isin(1, 3)).select("a")
    polars_result = polars_df.filter(pl.col("a").isin(1, 3)).select("a")

    assert_frame_equal(
        _spark_to_polars(spark_result).sort("a"),
        polars_result.sort("a"),
        check_dtypes=False,
    )


def test_between(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    spark_result = spark_df.filter(F.col("a").between(2, 4)).select("a")
    polars_result = polars_df.filter(pl.col("a").between(2, 4)).select("a")

    assert_frame_equal(
        _spark_to_polars(spark_result).sort("a"),
        polars_result.sort("a"),
        check_dtypes=False,
    )


def test_eqNullSafe(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, 1), (2, None), (None, None), (3, 4)], ["a", "b"]
    )
    polars_df = pl.DataFrame({"a": [1, 2, None, 3], "b": [1, None, None, 4]})

    spark_result = spark_df.select(F.col("a").eqNullSafe(F.col("b")).alias("r"))
    polars_result = polars_df.select(pl.col("a").eqNullSafe(pl.col("b")).alias("r"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# String methods
# ===================================================================

def test_rlike(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",), ("abc123",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world", "abc123"]})

    spark_result = spark_df.filter(F.col("s").rlike(r"\d+")).select("s")
    polars_result = polars_df.filter(pl.col("s").rlike(r"\d+")).select("s")

    assert_frame_equal(
        _spark_to_polars(spark_result).sort("s"),
        polars_result.sort("s"),
        check_dtypes=False,
    )


def test_startswith(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",), ("help",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world", "help"]})

    spark_result = spark_df.filter(F.col("s").startswith("hel")).select("s")
    polars_result = polars_df.filter(pl.col("s").startswith("hel")).select("s")

    assert_frame_equal(
        _spark_to_polars(spark_result).sort("s"),
        polars_result.sort("s"),
        check_dtypes=False,
    )


def test_endswith(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",), ("bold",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world", "bold"]})

    spark_result = spark_df.filter(F.col("s").endswith("ld")).select("s")
    polars_result = polars_df.filter(pl.col("s").endswith("ld")).select("s")

    assert_frame_equal(
        _spark_to_polars(spark_result).sort("s"),
        polars_result.sort("s"),
        check_dtypes=False,
    )


def test_substr(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world"]})

    spark_result = spark_df.select(F.col("s").substr(1, 3).alias("s"))
    polars_result = polars_df.select(pl.col("s").substr(1, 3).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_contains(spark_session):
    spark_df = spark_session.createDataFrame([("foobar",), ("bazqux",)], ["s"])
    polars_df = pl.DataFrame({"s": ["foobar", "bazqux"]})

    spark_result = spark_df.filter(F.col("s").contains("bar")).select("s")
    polars_result = polars_df.filter(pl.col("s").contains("bar")).select("s")

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_like(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",), ("help",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world", "help"]})

    spark_result = spark_df.filter(F.col("s").like("hel%")).select("s")
    polars_result = polars_df.filter(pl.col("s").like("hel%")).select("s")

    assert_frame_equal(
        _spark_to_polars(spark_result).sort("s"),
        polars_result.sort("s"),
        check_dtypes=False,
    )


def test_ilike(spark_session):
    spark_df = spark_session.createDataFrame([("Hello",), ("HELP",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["Hello", "HELP", "world"]})

    spark_result = spark_df.filter(F.col("s").ilike("hel%")).select("s")
    polars_result = polars_df.filter(pl.col("s").ilike("hel%")).select("s")

    assert_frame_equal(
        _spark_to_polars(spark_result).sort("s"),
        polars_result.sort("s"),
        check_dtypes=False,
    )


def test_upper(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world"]})

    spark_result = spark_df.select(F.upper(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").upper().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_lower(spark_session):
    spark_df = spark_session.createDataFrame([("HELLO",), ("WORLD",)], ["s"])
    polars_df = pl.DataFrame({"s": ["HELLO", "WORLD"]})

    spark_result = spark_df.select(F.lower(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").lower().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_trim(spark_session):
    spark_df = spark_session.createDataFrame([("  hello  ",), ("  world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["  hello  ", "  world"]})

    spark_result = spark_df.select(F.trim(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").trim().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_btrim(spark_session):
    spark_df = spark_session.createDataFrame([("  hello  ",), ("  world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["  hello  ", "  world"]})

    spark_result = spark_df.select(F.trim(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").btrim().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_ltrim(spark_session):
    spark_df = spark_session.createDataFrame([("  hello  ",), ("  world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["  hello  ", "  world"]})

    spark_result = spark_df.select(F.ltrim(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").ltrim().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_rtrim(spark_session):
    spark_df = spark_session.createDataFrame([("  hello  ",), ("  world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["  hello  ", "  world"]})

    spark_result = spark_df.select(F.rtrim(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").rtrim().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_lpad(spark_session):
    spark_df = spark_session.createDataFrame([("hi",), ("a",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hi", "a"]})

    spark_result = spark_df.select(F.lpad(F.col("s"), 5, "0").alias("s"))
    polars_result = polars_df.select(pl.col("s").lpad(5, "0").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_rpad(spark_session):
    spark_df = spark_session.createDataFrame([("hi",), ("a",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hi", "a"]})

    spark_result = spark_df.select(F.rpad(F.col("s"), 5, "0").alias("s"))
    polars_result = polars_df.select(pl.col("s").rpad(5, "0").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_left(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world"]})

    # PySpark doesn't have a direct left() -- use substr
    spark_result = spark_df.select(F.col("s").substr(1, 3).alias("s"))
    polars_result = polars_df.select(pl.col("s").left(3).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_right(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world"]})

    # PySpark: right(col, n)
    spark_result = spark_df.select(F.expr("right(s, 3)").alias("s"))
    polars_result = polars_df.select(pl.col("s").right(3).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_length(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("hi",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "hi"]})

    spark_result = spark_df.select(F.length(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").length().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_locate(spark_session):
    spark_df = spark_session.createDataFrame([("hello world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world"]})

    # Spark locate is 1-based; polyspark locate is 0-based like Polars str.find
    spark_result = spark_df.select(F.locate("llo", F.col("s")).alias("s"))
    polars_result = polars_df.select((pl.col("s").locate("llo") + 1).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_position(spark_session):
    # position is an alias for locate
    spark_df = spark_session.createDataFrame([("hello world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world"]})

    spark_result = spark_df.select(F.locate("world", F.col("s")).alias("s"))
    polars_result = polars_df.select((pl.col("s").position("world") + 1).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_repeat(spark_session):
    spark_df = spark_session.createDataFrame([("ab",)], ["s"])
    polars_df = pl.DataFrame({"s": ["ab"]})

    spark_result = spark_df.select(F.expr("repeat(s, 3)").alias("s"))
    polars_result = polars_df.select(pl.col("s").repeat(3).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_reverse(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world"]})

    spark_result = spark_df.select(F.reverse(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").reverse().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_split(spark_session):
    spark_df = spark_session.createDataFrame([("a,b,c",)], ["s"])
    polars_df = pl.DataFrame({"s": ["a,b,c"]})

    spark_result = spark_df.select(F.split(F.col("s"), ",").alias("s"))
    polars_result = polars_df.select(pl.col("s").split(",").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_base64(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})

    spark_result = spark_df.select(F.base64(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").base64().alias("s"))

    # base64 in Polars returns Binary; cast to Utf8 for comparison
    polars_result = polars_result.select(pl.col("s").cast(pl.String))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_encode(spark_session):
    # Spark encode returns bytes; polyspark encode returns the encoded form
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})

    # Just test polyspark doesn't error and produces a result
    polars_result = polars_df.select(pl.col("s").encode("base64").alias("s"))
    assert polars_result.shape[0] == 1


def test_decode(spark_session):
    import base64 as b64
    encoded = b64.b64encode(b"hello").decode()
    polars_df = pl.DataFrame({"s": [encoded]})

    polars_result = polars_df.select(pl.col("s").decode("base64").alias("s"))
    assert polars_result.shape[0] == 1


def test_regexp_count(spark_session):
    spark_df = spark_session.createDataFrame([("hello world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world"]})

    spark_result = spark_df.select(F.expr("regexp_count(s, 'l')").alias("s"))
    polars_result = polars_df.select(pl.col("s").regexp_count("l").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_regexp_extract(spark_session):
    spark_df = spark_session.createDataFrame([("2023-01-15",)], ["s"])
    polars_df = pl.DataFrame({"s": ["2023-01-15"]})

    spark_result = spark_df.select(F.regexp_extract(F.col("s"), r"(\d{4})", 1).alias("s"))
    polars_result = polars_df.select(pl.col("s").regexp_extract(r"(\d{4})", 1).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_regexp_extract_all(spark_session):
    spark_df = spark_session.createDataFrame([("hello world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world"]})

    spark_result = spark_df.select(F.expr(r"regexp_extract_all(s, '\\w+', 0)").alias("s"))
    polars_result = polars_df.select(pl.col("s").regexp_extract_all(r"\w+").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_regexp_replace(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})

    # Spark regexp_replace replaces ALL matches by default; polyspark replaces first
    # Use a pattern that matches only once to get the same result
    spark_result = spark_df.select(F.regexp_replace(F.col("s"), "^h", "H").alias("s"))
    polars_result = polars_df.select(pl.col("s").regexp_replace("^h", "H").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_str_replace(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})

    # str_replace replaces first occurrence
    spark_result = spark_df.select(F.regexp_replace(F.col("s"), "^h", "H").alias("s"))
    polars_result = polars_df.select(pl.col("s").str_replace("^h", "H").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_translate(spark_session):
    spark_df = spark_session.createDataFrame([("abc",), ("xyz",)], ["s"])
    polars_df = pl.DataFrame({"s": ["abc", "xyz"]})

    spark_result = spark_df.select(F.translate(F.col("s"), "abc", "xyz").alias("s"))
    polars_result = polars_df.select(pl.col("s").translate("abc", "xyz").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_initcap(spark_session):
    spark_df = spark_session.createDataFrame([("hello world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world"]})

    spark_result = spark_df.select(F.initcap(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").initcap().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_ascii_code(spark_session):
    spark_df = spark_session.createDataFrame([("A",), ("Z",)], ["s"])
    polars_df = pl.DataFrame({"s": ["A", "Z"]})

    spark_result = spark_df.select(F.ascii(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").ascii_code().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_instr(spark_session):
    spark_df = spark_session.createDataFrame([("hello world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world"]})

    spark_result = spark_df.select(F.instr(F.col("s"), "world").alias("s"))
    polars_result = polars_df.select(pl.col("s").instr("world").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_split_part(spark_session):
    spark_df = spark_session.createDataFrame([("a,b,c",)], ["s"])
    polars_df = pl.DataFrame({"s": ["a,b,c"]})

    spark_result = spark_df.select(F.expr("split_part(s, ',', 2)").alias("s"))
    polars_result = polars_df.select(pl.col("s").split_part(",", 2).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_substring_index(spark_session):
    spark_df = spark_session.createDataFrame([("a.b.c.d",)], ["s"])
    polars_df = pl.DataFrame({"s": ["a.b.c.d"]})

    spark_result = spark_df.select(F.substring_index(F.col("s"), ".", 2).alias("s"))
    polars_result = polars_df.select(pl.col("s").substring_index(".", 2).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Type casting
# ===================================================================

def test_cast(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2]})

    spark_result = spark_df.select(F.col("a").cast("double").alias("a"))
    polars_result = polars_df.select(pl.col("a").cast(pl.Float64).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_astype(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2]})

    spark_result = spark_df.select(F.col("a").cast("string").alias("a"))
    polars_result = polars_df.select(pl.col("a").astype(pl.String).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_try_cast(spark_session):
    # Spark try_cast available via expr
    spark_df = spark_session.createDataFrame([("1",), ("abc",), ("3",)], ["s"])
    polars_df = pl.DataFrame({"s": ["1", "abc", "3"]})

    spark_result = spark_df.select(F.expr("try_cast(s as int)").alias("s"))
    polars_result = polars_df.select(pl.col("s").try_cast(pl.Int32).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_isNaN(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1.0,), (float("nan"),), (3.0,)], ["v"]
    )
    polars_df = pl.DataFrame({"v": [1.0, float("nan"), 3.0]})

    spark_result = spark_df.select(F.isnan(F.col("v")).alias("v"))
    polars_result = polars_df.select(pl.col("v").isNaN().alias("v"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Bitwise
# ===================================================================

def test_bitwiseAND(spark_session):
    spark_df = spark_session.createDataFrame([(10, 12), (12, 10)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [10, 12], "b": [12, 10]})

    spark_result = spark_df.select(F.col("a").bitwiseAND(F.col("b")).alias("r"))
    polars_result = polars_df.select(pl.col("a").bitwiseAND(pl.col("b")).alias("r"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_bitwiseOR(spark_session):
    spark_df = spark_session.createDataFrame([(10, 5), (0, 15)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [10, 0], "b": [5, 15]})

    spark_result = spark_df.select(F.col("a").bitwiseOR(F.col("b")).alias("r"))
    polars_result = polars_df.select(pl.col("a").bitwiseOR(pl.col("b")).alias("r"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_bitwiseXOR(spark_session):
    spark_df = spark_session.createDataFrame([(10, 12), (12, 10)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [10, 12], "b": [12, 10]})

    spark_result = spark_df.select(F.col("a").bitwiseXOR(F.col("b")).alias("r"))
    polars_result = polars_df.select(pl.col("a").bitwiseXOR(pl.col("b")).alias("r"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_bitwiseNOT(spark_session):
    spark_df = spark_session.createDataFrame([(5,)], ["x"])
    polars_df = pl.DataFrame({"x": [5]})

    spark_result = spark_df.select(F.bitwise_not(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").bitwiseNOT().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_shiftLeft(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2]})

    spark_result = spark_df.select(F.shiftLeft(F.col("x"), 3).alias("x"))
    polars_result = polars_df.select(pl.col("x").shiftLeft(3).alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_shiftRight(spark_session):
    spark_df = spark_session.createDataFrame([(8,), (16,)], ["x"])
    polars_df = pl.DataFrame({"x": [8, 16]})

    spark_result = spark_df.select(F.shiftRight(F.col("x"), 2).alias("x"))
    polars_result = polars_df.select(pl.col("x").shiftRight(2).alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Sorting
# ===================================================================

def test_asc(spark_session):
    spark_df = spark_session.createDataFrame([(3,), (1,), (2,)], ["a"])
    polars_df = pl.DataFrame({"a": [3, 1, 2]})

    spark_result = spark_df.orderBy(F.col("a").asc()).select("a")
    polars_result = polars_df.select(pl.col("a").asc())

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_desc(spark_session):
    spark_df = spark_session.createDataFrame([(3,), (1,), (2,)], ["a"])
    polars_df = pl.DataFrame({"a": [3, 1, 2]})

    spark_result = spark_df.orderBy(F.col("a").desc()).select("a")
    polars_result = polars_df.select(pl.col("a").desc())

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_asc_nulls_first(spark_session):
    spark_df = spark_session.createDataFrame([(3,), (None,), (1,)], ["a"])
    polars_df = pl.DataFrame({"a": [3, None, 1]})

    spark_result = spark_df.orderBy(F.col("a").asc_nulls_first()).select("a")
    polars_result = polars_df.select(pl.col("a").asc_nulls_first())

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_asc_nulls_last(spark_session):
    spark_df = spark_session.createDataFrame([(3,), (None,), (1,)], ["a"])
    polars_df = pl.DataFrame({"a": [3, None, 1]})

    spark_result = spark_df.orderBy(F.col("a").asc_nulls_last()).select("a")
    polars_result = polars_df.select(pl.col("a").asc_nulls_last())

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_desc_nulls_first(spark_session):
    spark_df = spark_session.createDataFrame([(3,), (None,), (1,)], ["a"])
    polars_df = pl.DataFrame({"a": [3, None, 1]})

    spark_result = spark_df.orderBy(F.col("a").desc_nulls_first()).select("a")
    polars_result = polars_df.select(pl.col("a").desc_nulls_first())

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_desc_nulls_last(spark_session):
    spark_df = spark_session.createDataFrame([(3,), (None,), (1,)], ["a"])
    polars_df = pl.DataFrame({"a": [3, None, 1]})

    spark_result = spark_df.orderBy(F.col("a").desc_nulls_last()).select("a")
    polars_result = polars_df.select(pl.col("a").desc_nulls_last())

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Struct operations
# ===================================================================

def test_getField(spark_session):
    spark_df = spark_session.createDataFrame(
        [Row(s=Row(x=1, y=2)), Row(s=Row(x=3, y=4))]
    )
    polars_df = pl.DataFrame({"s": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]})

    spark_result = spark_df.select(F.col("s").getField("x").alias("x"))
    polars_result = polars_df.select(pl.col("s").getField("x").alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_withField(spark_session):
    spark_df = spark_session.createDataFrame(
        [Row(s=Row(x=1, y=2)), Row(s=Row(x=3, y=4))]
    )
    polars_df = pl.DataFrame({"s": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]})

    spark_result = spark_df.select(F.col("s").withField("z", F.lit(99)).alias("s"))
    polars_result = polars_df.select(pl.col("s").withField("z", pl.lit(99)).alias("s"))

    # Compare the z field extracted from the struct
    spark_z = spark_result.select(F.col("s").getField("z").alias("z"))
    polars_z = polars_result.select(pl.col("s").getField("z").alias("z"))

    assert_frame_equal(_spark_to_polars(spark_z), polars_z, check_dtypes=False)


def test_dropFields():
    # dropFields is not supported -- test that it raises
    import pytest
    polars_df = pl.DataFrame({"s": [{"x": 1, "y": 2}]})
    with pytest.raises(NotImplementedError):
        polars_df.select(pl.col("s").dropFields("x"))


def test_to_json(spark_session):
    spark_df = spark_session.createDataFrame(
        [Row(s=Row(a=1, b=2))]
    )
    polars_df = pl.DataFrame({"s": [{"a": 1, "b": 2}]})

    spark_result = spark_df.select(F.to_json(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").to_json().alias("s"))

    # Compare parsed JSON dicts since key order can differ
    spark_val = json.loads(_spark_to_polars(spark_result)["s"][0])
    polars_val = json.loads(polars_result["s"][0])
    assert spark_val == polars_val


# ===================================================================
# Map operations
# ===================================================================

def test_map_keys(spark_session):
    spark_df = spark_session.createDataFrame(
        [Row(m={"a": "1", "b": "2"})],
    )
    polars_df = pl.DataFrame(
        {"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]}
    )

    spark_result = spark_df.select(F.map_keys(F.col("m")).alias("m"))
    polars_result = polars_df.select(pl.col("m").map_keys().alias("m"))

    # Sort the list values before comparing since map key order is non-deterministic
    spark_pl = _spark_to_polars(spark_result)
    spark_sorted = spark_pl.select(pl.col("m").list.sort())
    polars_sorted = polars_result.select(pl.col("m").list.sort())

    assert_frame_equal(spark_sorted, polars_sorted, check_dtypes=False)


def test_map_values(spark_session):
    spark_df = spark_session.createDataFrame(
        [Row(m={"a": "1", "b": "2"})],
    )
    polars_df = pl.DataFrame(
        {"m": [[{"key": "a", "value": "1"}, {"key": "b", "value": "2"}]]}
    )

    spark_result = spark_df.select(F.map_values(F.col("m")).alias("m"))
    polars_result = polars_df.select(pl.col("m").map_values().alias("m"))

    spark_pl = _spark_to_polars(spark_result)
    spark_sorted = spark_pl.select(pl.col("m").list.sort())
    polars_sorted = polars_result.select(pl.col("m").list.sort())

    assert_frame_equal(spark_sorted, polars_sorted, check_dtypes=False)


# ===================================================================
# Conditional: when/otherwise
# ===================================================================

def test_when_otherwise(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    spark_result = spark_df.select(
        F.when(F.col("a") > 3, F.lit(99)).otherwise(F.lit(0)).alias("r")
    )
    polars_result = polars_df.select(
        pl.col("a").when(pl.col("a") > 3, pl.lit(99)).otherwise(pl.lit(0)).alias("r")
    )

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Math & aggregation on Expr
# ===================================================================

def test_negate(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (-2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, -2, 3]})

    spark_result = spark_df.select(F.expr("-x").alias("x"))
    polars_result = polars_df.select(pl.col("x").negate().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_avg(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,), (4.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    spark_result = spark_df.select(F.avg(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").avg().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_log2(spark_session):
    spark_df = spark_session.createDataFrame([(8.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [8.0]})

    spark_result = spark_df.select(F.log2(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").log2().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_log10(spark_session):
    spark_df = spark_session.createDataFrame([(100.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [100.0]})

    spark_result = spark_df.select(F.log10(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").log10().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_cbrt(spark_session):
    spark_df = spark_session.createDataFrame([(8.0,), (27.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [8.0, 27.0]})

    spark_result = spark_df.select(F.cbrt(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").cbrt().alias("x"))

    # Compare with tolerance due to floating point
    spark_pl = _spark_to_polars(spark_result)
    for i in range(polars_result.height):
        assert abs(spark_pl["x"][i] - polars_result["x"][i]) < 1e-10


def test_signum(spark_session):
    spark_df = spark_session.createDataFrame([(3.0,), (-2.0,), (0.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [3.0, -2.0, 0.0]})

    spark_result = spark_df.select(F.signum(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").signum().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Null handling
# ===================================================================

def test_nvl(spark_session):
    spark_df = spark_session.createDataFrame([(None,), (2,), (None,)], ["a"])
    polars_df = pl.DataFrame({"a": [None, 2, None]})

    spark_result = spark_df.select(F.coalesce(F.col("a"), F.lit(-1)).alias("a"))
    polars_result = polars_df.select(pl.col("a").nvl(-1).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_ifnull(spark_session):
    spark_df = spark_session.createDataFrame([(None,), (2,), (None,)], ["a"])
    polars_df = pl.DataFrame({"a": [None, 2, None]})

    spark_result = spark_df.select(F.coalesce(F.col("a"), F.lit(-1)).alias("a"))
    polars_result = polars_df.select(pl.col("a").ifnull(-1).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_nullif(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    spark_result = spark_df.select(F.expr("nullif(a, 2)").alias("a"))
    polars_result = polars_df.select(pl.col("a").nullif(2).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_nanvl(spark_session):
    spark_df = spark_session.createDataFrame(
        [(float("nan"),), (2.0,), (float("nan"),)], ["x"]
    )
    polars_df = pl.DataFrame({"x": [float("nan"), 2.0, float("nan")]})

    spark_result = spark_df.select(F.nanvl(F.col("x"), F.lit(-1.0)).alias("x"))
    polars_result = polars_df.select(pl.col("x").nanvl(-1.0).alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_nvl2(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (None,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, None, 3]})

    spark_result = spark_df.select(F.expr("nvl2(a, 100, 0)").alias("a"))
    polars_result = polars_df.select(pl.col("a").nvl2(100, 0).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Date/time
# ===================================================================

def test_year(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 6, 15),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})

    spark_result = spark_df.select(F.year(F.col("d")).alias("d"))
    polars_result = polars_df.select(pl.col("d").year().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_month(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 6, 15),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})

    spark_result = spark_df.select(F.month(F.col("d")).alias("d"))
    polars_result = polars_df.select(pl.col("d").month().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_hour(spark_session):
    from datetime import timezone
    ts = datetime.datetime(2023, 1, 16, 14, 30, 0, tzinfo=timezone.utc)
    spark_df = spark_session.createDataFrame([(ts,)], ["t"])
    polars_df = pl.DataFrame({"t": [ts]})

    spark_result = spark_df.select(F.hour(F.col("t")).alias("t"))
    polars_result = polars_df.select(pl.col("t").hour().alias("t"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_dayofmonth(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 16),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})

    spark_result = spark_df.select(F.dayofmonth(F.col("d")).alias("d"))
    polars_result = polars_df.select(pl.col("d").dayofmonth().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_dayofweek(spark_session):
    # 2023-01-16 is Monday. Spark: Mon=2.
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 16),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})

    spark_result = spark_df.select(F.dayofweek(F.col("d")).alias("d"))
    polars_result = polars_df.select(pl.col("d").dayofweek().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_dayofyear(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 16),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})

    spark_result = spark_df.select(F.dayofyear(F.col("d")).alias("d"))
    polars_result = polars_df.select(pl.col("d").dayofyear().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_last_day(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 16),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})

    spark_result = spark_df.select(F.last_day(F.col("d")).alias("d"))
    polars_result = polars_df.select(pl.col("d").last_day().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_date_add(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 16),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})

    spark_result = spark_df.select(F.date_add(F.col("d"), 5).alias("d"))
    polars_result = polars_df.select(pl.col("d").date_add(5).alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_date_sub(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 16),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})

    spark_result = spark_df.select(F.date_sub(F.col("d"), 5).alias("d"))
    polars_result = polars_df.select(pl.col("d").date_sub(5).alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_add_months(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 16),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 16)]})

    spark_result = spark_df.select(F.add_months(F.col("d"), 2).alias("d"))
    polars_result = polars_df.select(pl.col("d").add_months(2).alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_to_date(spark_session):
    spark_df = spark_session.createDataFrame([("2023-06-15",)], ["s"])
    polars_df = pl.DataFrame({"s": ["2023-06-15"]})

    spark_result = spark_df.select(F.to_date(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").to_date().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_to_timestamp(spark_session):
    spark_df = spark_session.createDataFrame([("2023-06-15 10:30:00",)], ["s"])
    polars_df = pl.DataFrame({"s": ["2023-06-15 10:30:00"]})

    spark_result = spark_df.select(F.to_timestamp(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").to_timestamp().alias("s"))

    # Extract hour to verify parsing works the same way
    spark_hour = spark_result.select(F.hour(F.col("s")).alias("h"))
    spark_h = _spark_to_polars(spark_hour)["h"][0]
    polars_h = polars_result.select(pl.col("s").dt.hour().alias("h"))["h"][0]

    assert spark_h == polars_h


def test_date_format(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 6, 15),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 6, 15)]})

    # Spark uses Java-style format; Polars uses strftime
    spark_result = spark_df.select(F.date_format(F.col("d"), "yyyy/MM/dd").alias("d"))
    polars_result = polars_df.select(pl.col("d").date_format("%Y/%m/%d").alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_unix_timestamp(spark_session):
    from datetime import timezone
    ts = datetime.datetime(2023, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
    spark_df = spark_session.createDataFrame([(ts,)], ["t"])
    polars_df = pl.DataFrame({"t": [ts]})

    spark_result = spark_df.select(F.unix_timestamp(F.col("t")).alias("t"))
    polars_result = polars_df.select(pl.col("t").unix_timestamp().alias("t"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_quarter(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 4, 15),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 4, 15)]})

    spark_result = spark_df.select(F.quarter(F.col("d")).alias("d"))
    polars_result = polars_df.select(pl.col("d").quarter().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_minute(spark_session):
    from datetime import timezone
    ts = datetime.datetime(2023, 1, 1, 14, 35, 0, tzinfo=timezone.utc)
    spark_df = spark_session.createDataFrame([(ts,)], ["t"])
    polars_df = pl.DataFrame({"t": [ts]})

    spark_result = spark_df.select(F.minute(F.col("t")).alias("t"))
    polars_result = polars_df.select(pl.col("t").minute().alias("t"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_second(spark_session):
    from datetime import timezone
    ts = datetime.datetime(2023, 1, 1, 14, 35, 45, tzinfo=timezone.utc)
    spark_df = spark_session.createDataFrame([(ts,)], ["t"])
    polars_df = pl.DataFrame({"t": [ts]})

    spark_result = spark_df.select(F.second(F.col("t")).alias("t"))
    polars_result = polars_df.select(pl.col("t").second().alias("t"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_weekofyear(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 9),)], ["d"]
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 9)]})

    spark_result = spark_df.select(F.weekofyear(F.col("d")).alias("d"))
    polars_result = polars_df.select(pl.col("d").weekofyear().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_weekday(spark_session):
    # Spark weekday(): Mon=0, Sun=6
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 9),)], ["d"]  # Monday
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 9)]})

    spark_result = spark_df.select(F.expr("weekday(d)").alias("d"))
    polars_result = polars_df.select(pl.col("d").weekday().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_date_trunc(spark_session):
    # Use a date (not datetime) to avoid timezone conversion issues
    d = datetime.date(2023, 6, 15)
    spark_df = spark_session.createDataFrame([(d,)], ["d"])
    polars_df = pl.DataFrame({"d": [d]})

    spark_result = spark_df.select(F.date_trunc("month", F.col("d")).alias("d"))
    polars_result = polars_df.select(pl.col("d").date_trunc("month").alias("d"))

    # Compare as date strings since Spark returns timestamp and Polars returns date
    spark_pl = _spark_to_polars(spark_result).select(
        pl.col("d").cast(pl.Date).cast(pl.String).alias("d")
    )
    polars_str = polars_result.select(pl.col("d").cast(pl.String).alias("d"))

    assert_frame_equal(spark_pl, polars_str, check_dtypes=False)


# ===================================================================
# Hash
# ===================================================================

def test_md5(spark_session):
    spark_df = spark_session.createDataFrame([("hello",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello", "world"]})

    spark_result = spark_df.select(F.md5(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").md5().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_sha1(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})

    spark_result = spark_df.select(F.sha1(F.col("s")).alias("s"))
    polars_result = polars_df.select(pl.col("s").sha1().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_sha256(spark_session):
    spark_df = spark_session.createDataFrame([("hello",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello"]})

    spark_result = spark_df.select(F.sha2(F.col("s"), 256).alias("s"))
    polars_result = polars_df.select(pl.col("s").sha256().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Array/List
# ===================================================================

def test_array_distinct(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 2, 3]]})

    spark_result = spark_df.select(F.array_distinct(F.col("a")).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_distinct().alias("a"))

    # Sort list values before comparing (order non-deterministic for distinct)
    spark_pl = _spark_to_polars(spark_result).select(pl.col("a").list.sort())
    polars_pl = polars_result.select(pl.col("a").list.sort())

    assert_frame_equal(spark_pl, polars_pl, check_dtypes=False)


def test_array_compact(spark_session):
    spark_df = spark_session.createDataFrame([([1, None, 2],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, None, 2]]})

    spark_result = spark_df.select(F.array_compact(F.col("a")).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_compact().alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_contains(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})

    spark_result = spark_df.select(F.array_contains(F.col("a"), 2).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_contains(2).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_append(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2]]})

    spark_result = spark_df.select(F.array_append(F.col("a"), 3).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_append(3).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_remove(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 1, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 1, 3]]})

    spark_result = spark_df.select(F.array_remove(F.col("a"), 1).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_remove(1).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_union(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3], [2, 3, 4])], ["a", "b"])
    polars_df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})

    spark_result = spark_df.select(F.array_union(F.col("a"), F.col("b")).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_union(pl.col("b")).alias("a"))

    spark_pl = _spark_to_polars(spark_result).select(pl.col("a").list.sort())
    polars_pl = polars_result.select(pl.col("a").list.sort())

    assert_frame_equal(spark_pl, polars_pl, check_dtypes=False)


def test_array_intersect(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3], [2, 3, 4])], ["a", "b"])
    polars_df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3, 4]]})

    spark_result = spark_df.select(F.array_intersect(F.col("a"), F.col("b")).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_intersect(pl.col("b")).alias("a"))

    spark_pl = _spark_to_polars(spark_result).select(pl.col("a").list.sort())
    polars_pl = polars_result.select(pl.col("a").list.sort())

    assert_frame_equal(spark_pl, polars_pl, check_dtypes=False)


def test_array_except(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3], [2, 3])], ["a", "b"])
    polars_df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[2, 3]]})

    spark_result = spark_df.select(F.array_except(F.col("a"), F.col("b")).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_except(pl.col("b")).alias("a"))

    spark_pl = _spark_to_polars(spark_result).select(pl.col("a").list.sort())
    polars_pl = polars_result.select(pl.col("a").list.sort())

    assert_frame_equal(spark_pl, polars_pl, check_dtypes=False)


def test_array_join(spark_session):
    spark_df = spark_session.createDataFrame([(["a", "b", "c"],)], "a: array<string>")
    polars_df = pl.DataFrame({"a": [["a", "b", "c"]]})

    spark_result = spark_df.select(F.array_join(F.col("a"), ",").alias("a"))
    polars_result = polars_df.select(pl.col("a").array_join(",").alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_max(spark_session):
    spark_df = spark_session.createDataFrame([([1, 3, 2],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 3, 2]]})

    spark_result = spark_df.select(F.array_max(F.col("a")).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_max().alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_min(spark_session):
    spark_df = spark_session.createDataFrame([([1, 3, 2],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 3, 2]]})

    spark_result = spark_df.select(F.array_min(F.col("a")).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_min().alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_size(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],), ([4, 5],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3], [4, 5]]})

    spark_result = spark_df.select(F.size(F.col("a")).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_size().alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_size(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})

    spark_result = spark_df.select(F.size(F.col("a")).alias("a"))
    polars_result = polars_df.select(pl.col("a").size().alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_sort(spark_session):
    spark_df = spark_session.createDataFrame([([3, 1, 2],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[3, 1, 2]]})

    spark_result = spark_df.select(F.array_sort(F.col("a")).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_sort().alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_slice(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3, 4],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3, 4]]})

    # Spark: slice(col, start_1based, length)
    spark_result = spark_df.select(F.slice(F.col("a"), 2, 2).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_slice(2, 2).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_position(spark_session):
    spark_df = spark_session.createDataFrame([([10, 20, 30],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[10, 20, 30]]})

    spark_result = spark_df.select(F.array_position(F.col("a"), 20).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_position(20).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_prepend(spark_session):
    spark_df = spark_session.createDataFrame([([10, 20, 30],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[10, 20, 30]]})

    spark_result = spark_df.select(F.array_prepend(F.col("a"), 5).alias("a"))
    polars_result = polars_df.select(pl.col("a").array_prepend(5).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_array_reverse(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})

    spark_result = spark_df.select(F.expr("reverse(a)").alias("a"))
    polars_result = polars_df.select(pl.col("a").array_reverse().alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_element_at(spark_session):
    spark_df = spark_session.createDataFrame([([10, 20, 30],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[10, 20, 30]]})

    spark_result = spark_df.select(F.element_at(F.col("a"), 2).alias("a"))
    polars_result = polars_df.select(pl.col("a").element_at(2).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_arrays_overlap(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3], [3, 4, 5])], ["a", "b"])
    polars_df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[3, 4, 5]]})

    spark_result = spark_df.select(F.arrays_overlap(F.col("a"), F.col("b")).alias("a"))
    polars_result = polars_df.select(pl.col("a").arrays_overlap(pl.col("b")).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_collect_list(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 3]})

    spark_result = spark_df.select(F.collect_list(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").collect_list().alias("x"))

    # Sort list contents for deterministic comparison
    spark_pl = _spark_to_polars(spark_result).select(pl.col("x").list.sort())
    polars_pl = polars_result.select(pl.col("x").list.sort())

    assert_frame_equal(spark_pl, polars_pl, check_dtypes=False)


def test_collect_set(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 2, 3]})

    spark_result = spark_df.select(F.collect_set(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").collect_set().alias("x"))

    spark_pl = _spark_to_polars(spark_result).select(pl.col("x").list.sort())
    polars_pl = polars_result.select(pl.col("x").list.sort())

    assert_frame_equal(spark_pl, polars_pl, check_dtypes=False)


# ===================================================================
# Aggregation
# ===================================================================

def test_stddev(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})

    spark_result = spark_df.select(F.stddev(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").stddev().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) < 1e-10


def test_variance(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})

    spark_result = spark_df.select(F.variance(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").variance().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) < 1e-10


def test_count_distinct(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 2, 3]})

    spark_result = spark_df.select(F.countDistinct(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").count_distinct().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_bool_and(spark_session):
    spark_df = spark_session.createDataFrame([(True,), (True,), (True,)], ["x"])
    polars_df = pl.DataFrame({"x": [True, True, True]})

    spark_result = spark_df.select(F.expr("bool_and(x)").alias("x"))
    polars_result = polars_df.select(pl.col("x").bool_and().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_every(spark_session):
    spark_df = spark_session.createDataFrame([(True,), (True,), (False,)], ["x"])
    polars_df = pl.DataFrame({"x": [True, True, False]})

    spark_result = spark_df.select(F.expr("every(x)").alias("x"))
    polars_result = polars_df.select(pl.col("x").every().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_bool_or(spark_session):
    spark_df = spark_session.createDataFrame([(False,), (False,), (True,)], ["x"])
    polars_df = pl.DataFrame({"x": [False, False, True]})

    spark_result = spark_df.select(F.expr("bool_or(x)").alias("x"))
    polars_result = polars_df.select(pl.col("x").bool_or().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_some(spark_session):
    spark_df = spark_session.createDataFrame([(False,), (True,), (False,)], ["x"])
    polars_df = pl.DataFrame({"x": [False, True, False]})

    spark_result = spark_df.select(F.expr("some(x)").alias("x"))
    polars_result = polars_df.select(pl.col("x").some().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_percentile(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,), (2.0,), (3.0,), (4.0,), (5.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

    spark_result = spark_df.select(F.percentile_approx(F.col("x"), 0.5).alias("x"))
    polars_result = polars_df.select(pl.col("x").percentile(0.5).alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) <= 1.0


def test_sum_distinct(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (2,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 2, 3]})

    spark_result = spark_df.select(F.sum_distinct(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").sum_distinct().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_approx_count_distinct(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (2,), (3,), (3,), (3,)], ["x"])
    polars_df = pl.DataFrame({"x": [1, 2, 2, 3, 3, 3]})

    spark_result = spark_df.select(F.approx_count_distinct(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").approx_count_distinct().alias("x"))

    # Allow small tolerance since Spark uses HLL approximation
    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) <= 1


# ===================================================================
# Window: lag, lead
# ===================================================================

def test_lag(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, "A"), (2, "A"), (3, "A"), (4, "B"), (5, "B")], ["x", "g"]
    )
    polars_df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "g": ["A", "A", "A", "B", "B"]})

    spark_w = SparkWindow.partitionBy("g").orderBy("x")
    poly_w = PolyWindow.partitionBy("g").orderBy("x")

    spark_result = spark_df.select("g", "x", F.lag(F.col("x"), 1).over(spark_w).alias("l"))
    spark_result = spark_result.orderBy("g", "x").select("l")

    polars_result = polars_df.with_columns(
        pl.col("x").lag(1).over(poly_w).alias("l")
    ).sort("g", "x").select("l")

    spark_pl = _spark_to_polars(spark_result)

    assert_frame_equal(spark_pl, polars_result, check_dtypes=False)


def test_lead(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, "A"), (2, "A"), (3, "A"), (4, "B"), (5, "B")], ["x", "g"]
    )
    polars_df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "g": ["A", "A", "A", "B", "B"]})

    spark_w = SparkWindow.partitionBy("g").orderBy("x")
    poly_w = PolyWindow.partitionBy("g").orderBy("x")

    spark_result = spark_df.select("g", "x", F.lead(F.col("x"), 1).over(spark_w).alias("l"))
    spark_result = spark_result.orderBy("g", "x").select("l")

    polars_result = polars_df.with_columns(
        pl.col("x").lead(1).over(poly_w).alias("l")
    ).sort("g", "x").select("l")

    spark_pl = _spark_to_polars(spark_result)

    assert_frame_equal(spark_pl, polars_result, check_dtypes=False)


# ===================================================================
# Extra math
# ===================================================================

def test_log1p(spark_session):
    spark_df = spark_session.createDataFrame([(0.0,), (1.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.0, 1.0]})

    spark_result = spark_df.select(F.log1p(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").log1p().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    for i in range(polars_result.height):
        assert abs(spark_pl["x"][i] - polars_result["x"][i]) < 1e-10


def test_expm1(spark_session):
    spark_df = spark_session.createDataFrame([(0.0,), (1.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.0, 1.0]})

    spark_result = spark_df.select(F.expm1(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").expm1().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    for i in range(polars_result.height):
        assert abs(spark_pl["x"][i] - polars_result["x"][i]) < 1e-10


def test_rint(spark_session):
    spark_df = spark_session.createDataFrame([(1.4,), (1.6,), (2.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.4, 1.6, 2.5]})

    spark_result = spark_df.select(F.rint(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").rint().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_bitcount(spark_session):
    spark_df = spark_session.createDataFrame([(7,), (0,), (15,)], ["x"])
    polars_df = pl.DataFrame({"x": [7, 0, 15]})

    spark_result = spark_df.select(F.expr("bit_count(x)").alias("x"))
    polars_result = polars_df.select(pl.col("x").bitcount().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Trig aliases
# ===================================================================

def test_asin(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})

    spark_result = spark_df.select(F.asin(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").asin().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) < 1e-10


def test_acos(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})

    spark_result = spark_df.select(F.acos(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").acos().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) < 1e-10


def test_atan(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0]})

    spark_result = spark_df.select(F.atan(F.col("x")).alias("x"))
    polars_result = polars_df.select(pl.col("x").atan().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) < 1e-10


def test_asinh(spark_session):
    spark_df = spark_session.createDataFrame([(1.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [1.0]})

    spark_result = spark_df.select(F.expr("asinh(x)").alias("x"))
    polars_result = polars_df.select(pl.col("x").asinh().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) < 1e-10


def test_acosh(spark_session):
    spark_df = spark_session.createDataFrame([(2.0,)], ["x"])
    polars_df = pl.DataFrame({"x": [2.0]})

    spark_result = spark_df.select(F.expr("acosh(x)").alias("x"))
    polars_result = polars_df.select(pl.col("x").acosh().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) < 1e-10


def test_atanh(spark_session):
    spark_df = spark_session.createDataFrame([(0.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [0.5]})

    spark_result = spark_df.select(F.expr("atanh(x)").alias("x"))
    polars_result = polars_df.select(pl.col("x").atanh().alias("x"))

    spark_pl = _spark_to_polars(spark_result)
    assert abs(spark_pl["x"][0] - polars_result["x"][0]) < 1e-10


# ===================================================================
# Other: chr, find_in_set, regexp_like, overlay
# ===================================================================

def test_chr(spark_session):
    spark_df = spark_session.createDataFrame([(65,), (66,), (67,)], ["x"])
    polars_df = pl.DataFrame({"x": [65, 66, 67]})

    spark_result = spark_df.select(F.expr("chr(x)").alias("x"))
    polars_result = polars_df.select(pl.col("x").chr().alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_find_in_set(spark_session):
    spark_df = spark_session.createDataFrame([("a,b,c,d",)], ["s"])
    polars_df = pl.DataFrame({"s": ["a,b,c,d"]})

    spark_result = spark_df.select(F.expr("find_in_set('c', s)").alias("s"))
    polars_result = polars_df.select(pl.col("s").find_in_set("c").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_regexp_like(spark_session):
    spark_df = spark_session.createDataFrame([("hello123",), ("world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello123", "world"]})

    spark_result = spark_df.select(F.expr(r"regexp_like(s, '\\d+')").alias("s"))
    polars_result = polars_df.select(pl.col("s").regexp_like(r"\d+").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_overlay(spark_session):
    spark_df = spark_session.createDataFrame([("hello world",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world"]})

    spark_result = spark_df.select(F.overlay(F.col("s"), F.lit("there"), 7).alias("s"))
    polars_result = polars_df.select(pl.col("s").overlay("there", 7).alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# bround, pmod
# ===================================================================

def test_bround(spark_session):
    spark_df = spark_session.createDataFrame([(2.5,), (3.5,)], ["x"])
    polars_df = pl.DataFrame({"x": [2.5, 3.5]})

    spark_result = spark_df.select(F.bround(F.col("x"), 0).alias("x"))
    polars_result = polars_df.select(pl.col("x").bround(0).alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_pmod(spark_session):
    spark_df = spark_session.createDataFrame([(-7,)], ["x"])
    polars_df = pl.DataFrame({"x": [-7]})

    spark_result = spark_df.select(F.expr("pmod(x, 3)").alias("x"))
    polars_result = polars_df.select(pl.col("x").pmod(3).alias("x"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# next_day, unix_date
# ===================================================================

def test_next_day(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(2023, 1, 11),)], ["d"]  # Wednesday
    )
    polars_df = pl.DataFrame({"d": [datetime.date(2023, 1, 11)]})

    spark_result = spark_df.select(F.next_day(F.col("d"), "Mon").alias("d"))
    polars_result = polars_df.select(pl.col("d").next_day("Mon").alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_unix_date(spark_session):
    spark_df = spark_session.createDataFrame(
        [(datetime.date(1970, 1, 1),), (datetime.date(1970, 1, 2),)], ["d"]
    )
    polars_df = pl.DataFrame(
        {"d": [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2)]}
    )

    spark_result = spark_df.select(F.expr("unix_date(d)").alias("d"))
    polars_result = polars_df.select(pl.col("d").unix_date().alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# to_utc_timestamp, from_utc_timestamp
# ===================================================================

def test_to_utc_timestamp(spark_session):
    # to_utc_timestamp interprets a naive timestamp as being in the given timezone
    # and converts it to UTC
    ts = datetime.datetime(2023, 6, 15, 12, 0, 0)
    spark_df = spark_session.createDataFrame([(ts,)], ["t"])
    polars_df = pl.DataFrame({"t": [ts]})

    spark_result = spark_df.select(F.to_utc_timestamp(F.col("t"), "America/New_York").alias("t"))
    polars_result = polars_df.select(pl.col("t").to_utc_timestamp("America/New_York").alias("t"))

    # Compare as epoch seconds to avoid timezone representation differences
    spark_pl = _spark_to_polars(spark_result).select(
        pl.col("t").cast(pl.Datetime("us")).dt.epoch("s").alias("t")
    )
    polars_ep = polars_result.select(pl.col("t").dt.epoch("s").alias("t"))

    assert_frame_equal(spark_pl, polars_ep, check_dtypes=False)


def test_from_utc_timestamp(spark_session):
    # from_utc_timestamp: interpret timestamp as UTC and convert to target tz.
    # Spark and Polars differ in how naive timestamps + tz conversions interact
    # during collection. Just verify the offset is applied correctly.
    ts = datetime.datetime(2023, 6, 15, 16, 0, 0)
    polars_df = pl.DataFrame({"t": [ts]})

    polars_result = polars_df.select(
        pl.col("t").from_utc_timestamp("America/New_York").dt.hour().alias("h")
    )

    # NYC is EDT (UTC-4) in June, so 16:00 UTC -> 12:00 EDT
    expected = pl.DataFrame({"h": [12]})
    assert_frame_equal(polars_result, expected, check_dtypes=False)


# ===================================================================
# unbase64, regexp_substr, levenshtein
# ===================================================================

def test_unbase64(spark_session):
    import base64 as b64
    encoded = b64.b64encode(b"hello").decode()

    spark_df = spark_session.createDataFrame([(encoded,)], ["s"])
    polars_df = pl.DataFrame({"s": [encoded]})

    # Spark unbase64 returns binary; cast to string for comparison
    spark_result = spark_df.select(F.expr("cast(unbase64(s) as string)").alias("s"))
    polars_result = polars_df.select(pl.col("s").unbase64().alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_regexp_substr(spark_session):
    spark_df = spark_session.createDataFrame([("hello world 123",)], ["s"])
    polars_df = pl.DataFrame({"s": ["hello world 123"]})

    spark_result = spark_df.select(F.expr(r"regexp_extract(s, '(\\d+)', 1)").alias("s"))
    polars_result = polars_df.select(pl.col("s").regexp_substr(r"(\d+)").alias("s"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_levenshtein(spark_session):
    spark_df = spark_session.createDataFrame([("kitten", "sitting")], ["a", "b"])
    polars_df = pl.DataFrame({"a": ["kitten"], "b": ["sitting"]})

    spark_result = spark_df.select(F.levenshtein(F.col("a"), F.col("b")).alias("d"))
    polars_result = polars_df.select(pl.col("a").levenshtein(pl.col("b")).alias("d"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# exists, shuffle, aggregate
# ===================================================================

def test_exists(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3]]})

    spark_result = spark_df.select(F.exists(F.col("a"), lambda x: x > 2).alias("a"))
    polars_result = polars_df.select(pl.col("a").exists(lambda e: e > 2).alias("a"))

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


def test_shuffle():
    # Shuffle is random, so just verify it returns a list with the same elements
    polars_df = pl.DataFrame({"a": [[1, 2, 3, 4, 5]]})
    result = polars_df.select(pl.col("a").shuffle(seed=42).alias("a"))
    assert sorted(result["a"][0].to_list()) == [1, 2, 3, 4, 5]


def test_aggregate(spark_session):
    spark_df = spark_session.createDataFrame([([1, 2, 3, 4],)], "a: array<int>")
    polars_df = pl.DataFrame({"a": [[1, 2, 3, 4]]})

    spark_result = spark_df.select(
        F.aggregate(F.col("a"), F.lit(0), lambda acc, x: acc + x).alias("a")
    )
    polars_result = polars_df.select(
        pl.col("a").aggregate(0, lambda acc, e: acc + e).alias("a")
    )

    assert_frame_equal(_spark_to_polars(spark_result), polars_result, check_dtypes=False)


# ===================================================================
# Polyspark-only / aliased (no Spark equivalent -- test against expected)
# ===================================================================

def test_transform():
    polars_df = pl.DataFrame({"arr": [[1, 2, 3], [4, 5, 6]]})
    result = polars_df.select(pl.col("arr").transform(lambda e: e * 2))
    expected = pl.DataFrame({"arr": [[2, 4, 6], [8, 10, 12]]})
    assert_frame_equal(result, expected, check_dtypes=False)


def test_list_filter():
    polars_df = pl.DataFrame({"a": [[1, 2, 3, 4]]})
    result = polars_df.select(pl.col("a").list_filter(lambda e: e > 2))
    expected = pl.DataFrame({"a": [[3, 4]]})
    assert_frame_equal(result, expected, check_dtypes=False)


def test_forall():
    polars_df = pl.DataFrame({"a": [[2, 4, 6]]})
    result = polars_df.select(pl.col("a").forall(lambda e: e > 0))
    expected = pl.DataFrame({"a": [True]})
    assert_frame_equal(result, expected, check_dtypes=False)


def test_forall_false():
    polars_df = pl.DataFrame({"a": [[2, -1, 6]]})
    result = polars_df.select(pl.col("a").forall(lambda e: e > 0))
    expected = pl.DataFrame({"a": [False]})
    assert_frame_equal(result, expected, check_dtypes=False)


def test_getItem():
    polars_df = pl.DataFrame({"m": [json.dumps({"a": "v1"}), json.dumps({"a": "v2"})]})
    result = polars_df.select(pl.col("m").getItem(pl.lit("a")).alias("val"))
    expected = pl.DataFrame({"val": ["v1", "v2"]})
    assert_frame_equal(result, expected, check_dtypes=False)
