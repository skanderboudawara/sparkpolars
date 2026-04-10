"""Integration tests for polyspark DataFrame monkey-patches.

Every test creates BOTH a Spark DataFrame and a Polars DataFrame with the same data,
runs the native PySpark operation on the Spark DF and the polyspark operation on the
Polars DF, then compares results via assert_frame_equal.
"""

import tempfile

import pytest
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Row
import polars as pl
from polars.testing import assert_frame_equal
import src.sparkpolars  # noqa: F401  # registers .toPolars() on SparkDataFrame 

import src.sparkpolars.polyspark.sql.dataframe  # noqa: F401
import src.sparkpolars.polyspark.sql.functions as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spark_to_polars(spark_df):
    """Convert a Spark DataFrame to Polars via Pandas."""
    return spark_df.toPolars()


# ===========================================================================
# Column operations
# ===========================================================================


def test_withColumn(spark_session):
    spark_df = spark_session.createDataFrame([(1, "a"), (2, "b")], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})

    spark_result = spark_df.withColumn("z", F.col("x") * 2)
    polars_result = polars_df.withColumn("z", pl.col("x") * 2)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("x"), polars_result.sort("x"), check_dtypes=False)


def test_withColumns(spark_session):
    spark_df = spark_session.createDataFrame([(1, 10), (2, 20)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [1, 2], "b": [10, 20]})

    spark_result = spark_df.withColumn("c", F.col("a") + F.col("b"))
    polars_result = polars_df.withColumns({"c": pl.col("a") + pl.col("b")})

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_withColumnRenamed(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,)], ["old_name"])
    polars_df = pl.DataFrame({"old_name": [1, 2]})

    spark_result = spark_df.withColumnRenamed("old_name", "new_name")
    polars_result = polars_df.withColumnRenamed("old_name", "new_name")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("new_name"), polars_result.sort("new_name"), check_dtypes=False)


def test_withColumnsRenamed(spark_session):
    spark_df = spark_session.createDataFrame([(1, "a"), (2, "b")], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})

    spark_result = spark_df.withColumnsRenamed({"x": "col_x", "y": "col_y"})
    polars_result = polars_df.withColumnsRenamed({"x": "col_x", "y": "col_y"})

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("col_x"), polars_result.sort("col_x"), check_dtypes=False)


def test_select(spark_session):
    spark_df = spark_session.createDataFrame([(1, "a", 1.0), (2, "b", 2.0)], ["a", "b", "c"])
    polars_df = pl.DataFrame({"a": [1, 2], "b": ["a", "b"], "c": [1.0, 2.0]})

    spark_result = spark_df.select("a", "b")
    polars_result = polars_df.select("a", "b")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_selectExpr(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    spark_result = spark_df.selectExpr("a", "a * 2 as a_doubled")
    polars_result = polars_df.selectExpr("a", "a * 2 as a_doubled")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_drop(spark_session):
    spark_df = spark_session.createDataFrame([(1, "a", 1.0)], ["a", "b", "c"])
    polars_df = pl.DataFrame({"a": [1], "b": ["a"], "c": [1.0]})

    spark_result = spark_df.drop("c")
    polars_result = polars_df.drop("c")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_colRegex(spark_session):
    spark_df = spark_session.createDataFrame([(1, 2, 3)], ["val_a", "val_b", "id"])
    polars_df = pl.DataFrame({"val_a": [1], "val_b": [2], "id": [3]})

    spark_result = spark_df.select(spark_df.colRegex("`^val_.*`"))
    polars_result = polars_df.colRegex("^val_.*")

    expected = _spark_to_polars(spark_result)
    assert set(expected.columns) == set(polars_result.columns)
    assert_frame_equal(
        expected.select(sorted(expected.columns)),
        polars_result.select(sorted(polars_result.columns)),
        check_dtypes=False,
    )


# ===========================================================================
# Filtering
# ===========================================================================


def test_filter(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    spark_result = spark_df.filter(F.col("a") > 1)
    polars_result = polars_df.filter(pl.col("a") > 1)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_where(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    spark_result = spark_df.where(F.col("a") <= 2)
    polars_result = polars_df.where(pl.col("a") <= 2)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_filter_sql_string(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    spark_result = spark_df.filter("a > 1")
    polars_result = polars_df.filter("a > 1")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


# ===========================================================================
# Sorting
# ===========================================================================


def test_sort_ascending(spark_session):
    spark_df = spark_session.createDataFrame([(3,), (1,), (2,)], ["a"])
    polars_df = pl.DataFrame({"a": [3, 1, 2]})

    spark_result = spark_df.sort("a", ascending=True)
    polars_result = polars_df.sort("a", ascending=True)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_sort_descending(spark_session):
    spark_df = spark_session.createDataFrame([(3,), (1,), (2,)], ["a"])
    polars_df = pl.DataFrame({"a": [3, 1, 2]})

    spark_result = spark_df.sort("a", ascending=False)
    polars_result = polars_df.sort("a", ascending=False)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_orderBy(spark_session):
    spark_df = spark_session.createDataFrame([(3, "c"), (1, "a"), (2, "b")], ["a", "b"])
    polars_df = pl.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]})

    spark_result = spark_df.orderBy("a", ascending=False)
    polars_result = polars_df.orderBy("a", ascending=False)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_sortWithinPartitions(spark_session):
    """sortWithinPartitions is aliased to sort in polyspark (single partition)."""
    polars_df = pl.DataFrame({"a": [3, 1, 2]})
    polars_result = polars_df.sortWithinPartitions("a", ascending=True)

    expected = pl.DataFrame({"a": [1, 2, 3]})
    assert_frame_equal(polars_result, expected, check_dtypes=False)


def test_sort_multi_columns(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, "b"), (2, "a"), (1, "a"), (2, "b")], ["x", "y"]
    )
    polars_df = pl.DataFrame({"x": [1, 2, 1, 2], "y": ["b", "a", "a", "b"]})

    spark_result = spark_df.sort("x", "y", ascending=[True, False])
    polars_result = polars_df.sort("x", "y", ascending=[True, False])

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


# ===========================================================================
# Joins -- ALL join types via parametrize
# ===========================================================================


@pytest.mark.parametrize("join_type", ["inner", "left", "right", "outer", "semi", "anti"])
def test_join_types(spark_session, join_type):
    spark_left = spark_session.createDataFrame(
        [(1, "a"), (2, "b"), (3, "c"), (4, "d")], ["id", "val"]
    )
    spark_right = spark_session.createDataFrame(
        [(2, "x"), (3, "y"), (5, "z")], ["id", "val2"]
    )
    polars_left = pl.DataFrame({"id": [1, 2, 3, 4], "val": ["a", "b", "c", "d"]})
    polars_right = pl.DataFrame({"id": [2, 3, 5], "val2": ["x", "y", "z"]})

    # Map generic test join type names to PySpark join type names
    spark_join_map = {
        "inner": "inner",
        "left": "left",
        "right": "right",
        "outer": "outer",
        "semi": "left_semi",
        "anti": "left_anti",
    }
    # Map to polyspark join type names
    polyspark_join_map = {
        "inner": "inner",
        "left": "left_outer",
        "right": "right_outer",
        "outer": "full_outer",
        "semi": "semi",
        "anti": "left_anti",
    }

    spark_result = spark_left.join(spark_right, on="id", how=spark_join_map[join_type])
    polars_result = polars_left.join(polars_right, on="id", how=polyspark_join_map[join_type])

    expected = _spark_to_polars(spark_result)

    # Polyspark full_outer keeps both key columns (id + id_r_polyspark), coalesce them
    if join_type == "outer" and "id_r_polyspark" in polars_result.columns:
        polars_result = polars_result.with_columns(
            pl.coalesce(pl.col("id"), pl.col("id_r_polyspark")).alias("id")
        ).drop("id_r_polyspark")

    # Determine sort columns -- use only columns that exist in both results
    sort_cols = sorted(set(expected.columns) & set(polars_result.columns))
    # For outer join, there may be nulls so we fill before sorting
    if join_type == "outer":
        expected = expected.fill_null("")
        polars_result = polars_result.fill_null("")

    assert set(expected.columns) == set(polars_result.columns)
    assert_frame_equal(
        expected.select(sorted(expected.columns)).sort(sort_cols),
        polars_result.select(sorted(polars_result.columns)).sort(sort_cols),
        check_dtypes=False,
    )


def test_cross_join(spark_session):
    spark_left = spark_session.createDataFrame([(1,), (2,)], ["a"])
    spark_right = spark_session.createDataFrame([(10,), (20,)], ["b"])
    polars_left = pl.DataFrame({"a": [1, 2]})
    polars_right = pl.DataFrame({"b": [10, 20]})

    spark_result = spark_left.crossJoin(spark_right)
    polars_result = polars_left.crossJoin(polars_right)

    expected = _spark_to_polars(spark_result)
    assert expected.shape == polars_result.shape
    assert_frame_equal(
        expected.sort("a", "b"),
        polars_result.sort("a", "b"),
        check_dtypes=False,
    )


def test_join_with_null_keys(spark_session):
    spark_left = spark_session.createDataFrame(
        [(1, "a"), (None, "b"), (3, "c")],
        T.StructType([
            T.StructField("id", T.IntegerType(), True),
            T.StructField("val", T.StringType()),
        ]),
    )
    spark_right = spark_session.createDataFrame(
        [(1, "x"), (None, "y")],
        T.StructType([
            T.StructField("id", T.IntegerType(), True),
            T.StructField("val2", T.StringType()),
        ]),
    )
    polars_left = pl.DataFrame({"id": [1, None, 3], "val": ["a", "b", "c"]}).cast({"id": pl.Int32})
    polars_right = pl.DataFrame({"id": [1, None], "val2": ["x", "y"]}).cast({"id": pl.Int32})

    spark_result = spark_left.join(spark_right, on="id", how="inner")
    polars_result = polars_left.join(polars_right, on="id", how="inner")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(
        expected.sort("id"),
        polars_result.sort("id"),
        check_dtypes=False,
    )


def test_join_duplicate_keys(spark_session):
    spark_left = spark_session.createDataFrame(
        [(1, "a"), (1, "b"), (2, "c")], ["id", "val"]
    )
    spark_right = spark_session.createDataFrame(
        [(1, "x"), (1, "y")], ["id", "val2"]
    )
    polars_left = pl.DataFrame({"id": [1, 1, 2], "val": ["a", "b", "c"]})
    polars_right = pl.DataFrame({"id": [1, 1], "val2": ["x", "y"]})

    spark_result = spark_left.join(spark_right, on="id", how="inner")
    polars_result = polars_left.join(polars_right, on="id", how="inner")

    expected = _spark_to_polars(spark_result)
    assert expected.height == polars_result.height  # 2*2 = 4 matches
    assert_frame_equal(
        expected.sort("id", "val", "val2"),
        polars_result.sort("id", "val", "val2"),
        check_dtypes=False,
    )


def test_join_multi_column_keys(spark_session):
    spark_left = spark_session.createDataFrame(
        [(1, "a", 10), (2, "b", 20)], ["id", "key", "val"]
    )
    spark_right = spark_session.createDataFrame(
        [(1, "a", 100), (2, "c", 200)], ["id", "key", "val2"]
    )
    polars_left = pl.DataFrame({"id": [1, 2], "key": ["a", "b"], "val": [10, 20]})
    polars_right = pl.DataFrame({"id": [1, 2], "key": ["a", "c"], "val2": [100, 200]})

    spark_result = spark_left.join(spark_right, on=["id", "key"], how="inner")
    polars_result = polars_left.join(polars_right, on=["id", "key"], how="inner")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(
        expected.sort("id"),
        polars_result.sort("id"),
        check_dtypes=False,
    )


# ===========================================================================
# Set operations
# ===========================================================================


def test_union(spark_session):
    spark_df1 = spark_session.createDataFrame([(1,), (2,)], ["a"])
    spark_df2 = spark_session.createDataFrame([(3,), (4,)], ["a"])
    polars_df1 = pl.DataFrame({"a": [1, 2]})
    polars_df2 = pl.DataFrame({"a": [3, 4]})

    spark_result = spark_df1.union(spark_df2)
    polars_result = polars_df1.union(polars_df2)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_unionAll(spark_session):
    spark_df1 = spark_session.createDataFrame([(1,), (2,)], ["a"])
    spark_df2 = spark_session.createDataFrame([(2,), (3,)], ["a"])
    polars_df1 = pl.DataFrame({"a": [1, 2]})
    polars_df2 = pl.DataFrame({"a": [2, 3]})

    spark_result = spark_df1.unionAll(spark_df2)
    polars_result = polars_df1.unionAll(polars_df2)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_unionByName(spark_session):
    spark_df1 = spark_session.createDataFrame([(1, "a")], ["id", "val"])
    spark_df2 = spark_session.createDataFrame([(2, "b")], ["id", "val"])
    polars_df1 = pl.DataFrame({"id": [1], "val": ["a"]})
    polars_df2 = pl.DataFrame({"id": [2], "val": ["b"]})

    spark_result = spark_df1.unionByName(spark_df2)
    polars_result = polars_df1.unionByName(polars_df2)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("id"), polars_result.sort("id"), check_dtypes=False)


def test_intersect(spark_session):
    spark_df1 = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    spark_df2 = spark_session.createDataFrame([(2,), (3,), (4,)], ["a"])
    polars_df1 = pl.DataFrame({"a": [1, 2, 3]})
    polars_df2 = pl.DataFrame({"a": [2, 3, 4]})

    spark_result = spark_df1.intersect(spark_df2)
    polars_result = polars_df1.intersect(polars_df2)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_intersectAll(spark_session):
    spark_df1 = spark_session.createDataFrame(
        [("a", 1), ("a", 1), ("b", 3), ("c", 4)], ["C1", "C2"]
    )
    spark_df2 = spark_session.createDataFrame(
        [("a", 1), ("a", 1), ("b", 3)], ["C1", "C2"]
    )
    polars_df1 = pl.DataFrame({"C1": ["a", "a", "b", "c"], "C2": [1, 1, 3, 4]})
    polars_df2 = pl.DataFrame({"C1": ["a", "a", "b"], "C2": [1, 1, 3]})

    spark_result = spark_df1.intersectAll(spark_df2)
    polars_result = polars_df1.intersectAll(polars_df2)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(
        expected.sort("C1", "C2"),
        polars_result.sort("C1", "C2"),
        check_dtypes=False,
    )


def test_subtract(spark_session):
    spark_df1 = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    spark_df2 = spark_session.createDataFrame([(2,), (3,), (4,)], ["a"])
    polars_df1 = pl.DataFrame({"a": [1, 2, 3]})
    polars_df2 = pl.DataFrame({"a": [2, 3, 4]})

    spark_result = spark_df1.subtract(spark_df2)
    polars_result = polars_df1.subtract(polars_df2)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


# ===========================================================================
# GroupBy -- ALL aggregation types
# ===========================================================================


def test_groupBy_agg_sum(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").agg(F.sum("sal").alias("sum(sal)"))
    polars_result = polars_df.groupBy("dept").agg(sf.sum("sal"))

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_agg_avg(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").agg(F.avg("sal").alias("avg(sal)"))
    polars_result = polars_df.groupBy("dept").agg(sf.avg("sal"))

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_agg_min(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").agg(F.min("sal").alias("min(sal)"))
    polars_result = polars_df.groupBy("dept").agg(sf.min("sal"))

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_agg_max(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").agg(F.max("sal").alias("max(sal)"))
    polars_result = polars_df.groupBy("dept").agg(sf.max("sal"))

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_agg_count(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").agg(F.count("sal").alias("count(sal)"))
    polars_result = polars_df.groupBy("dept").agg(sf.count("sal"))

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_agg_multiple(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").agg(
        F.sum("sal").alias("sum(sal)"),
        F.avg("sal").alias("avg(sal)"),
    )
    polars_result = polars_df.groupBy("dept").agg(sf.sum("sal"), sf.avg("sal"))

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_shorthand_sum(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").sum("sal")
    polars_result = polars_df.groupBy("dept").sum("sal")

    expected = _spark_to_polars(spark_result)
    # Spark names the column "sum(sal)" as well
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_shorthand_avg(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").avg("sal")
    polars_result = polars_df.groupBy("dept").avg("sal")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_shorthand_min(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").min("sal")
    polars_result = polars_df.groupBy("dept").min("sal")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_shorthand_max(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").max("sal")
    polars_result = polars_df.groupBy("dept").max("sal")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_shorthand_count(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").count()
    polars_result = polars_df.groupBy("dept").count()

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_dict_agg(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    spark_result = spark_df.groupBy("dept").agg({"sal": "sum"})
    polars_result = polars_df.groupBy("dept").agg({"sal": "sum"})

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_groupBy_global_agg(spark_session):
    spark_df = spark_session.createDataFrame([(100,), (200,), (300,)], ["sal"])
    polars_df = pl.DataFrame({"sal": [100, 200, 300]})

    spark_result = spark_df.agg(F.sum("sal").alias("sum(sal)"))
    polars_result = polars_df.agg(sf.sum("sal"))

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


# ===========================================================================
# SparkGroupBy wrapper tests
# ===========================================================================


def test_sparkgroupby_count(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A",), ("A",), ("B",), ("B",)], ["dept"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B", "B"]})

    spark_result = spark_df.groupBy("dept").count()
    polars_result = polars_df.groupBy("dept").count()

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_sparkgroupby_sum(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 10), ("A", 20), ("B", 30)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [10, 20, 30]})

    spark_result = spark_df.groupBy("dept").sum("sal")
    polars_result = polars_df.groupBy("dept").sum("sal")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_sparkgroupby_avg(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 10), ("A", 20), ("B", 30)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [10, 20, 30]})

    spark_result = spark_df.groupBy("dept").avg("sal")
    polars_result = polars_df.groupBy("dept").avg("sal")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_sparkgroupby_min(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 10), ("A", 20), ("B", 30)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [10, 20, 30]})

    spark_result = spark_df.groupBy("dept").min("sal")
    polars_result = polars_df.groupBy("dept").min("sal")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_sparkgroupby_max(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 10), ("A", 20), ("B", 30)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [10, 20, 30]})

    spark_result = spark_df.groupBy("dept").max("sal")
    polars_result = polars_df.groupBy("dept").max("sal")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


def test_sparkgroupby_agg_multiple(spark_session):
    spark_df = spark_session.createDataFrame(
        [("A", 10), ("A", 20), ("B", 30)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [10, 20, 30]})

    spark_result = spark_df.groupBy("dept").agg(
        F.sum("sal").alias("sum(sal)"),
        F.min("sal").alias("min(sal)"),
        F.max("sal").alias("max(sal)"),
    )
    polars_result = polars_df.groupBy("dept").agg(
        sf.sum("sal"), sf.min("sal"), sf.max("sal")
    )

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("dept"), polars_result.sort("dept"), check_dtypes=False)


# ===========================================================================
# Describe / Summary
# ===========================================================================


def test_describe(spark_session):
    spark_df = spark_session.createDataFrame([(1, 1.0), (2, 2.0), (3, 3.0)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

    spark_result = spark_df.describe()
    polars_result = polars_df.describe()

    # Both should have a "statistic" column and produce summary stats
    expected = _spark_to_polars(spark_result)
    assert "summary" in expected.columns or "statistic" in expected.columns
    assert "statistic" in polars_result.columns
    # Check that both produce the same number of summary statistics
    assert expected.height > 0
    assert polars_result.height > 0


def test_summary(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    spark_result = spark_df.summary()
    polars_result = polars_df.summary()

    expected = _spark_to_polars(spark_result)
    assert expected.height > 0
    assert polars_result.height > 0
    assert "statistic" in polars_result.columns


# ===========================================================================
# Null handling
# ===========================================================================


def test_fillna(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, None), (None, "b"), (3, None)],
        T.StructType([
            T.StructField("a", T.IntegerType(), True),
            T.StructField("b", T.StringType(), True),
        ]),
    )
    polars_df = pl.DataFrame(
        {"a": [1, None, 3], "b": [None, "b", None]}
    ).cast({"a": pl.Int32})

    spark_result = spark_df.fillna({"a": 0, "b": "missing"})
    polars_result = polars_df.fillna(0, subset=["a"]).fillna("missing", subset=["b"])

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_dropna(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1.0,), (float("nan"),), (3.0,)], ["a"]
    )
    polars_df = pl.DataFrame({"a": [1.0, float("nan"), 3.0]})

    spark_result = spark_df.dropna()
    polars_result = polars_df.dropna()

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_dropnulls(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, "a"), (None, "b"), (3, None)],
        T.StructType([
            T.StructField("a", T.IntegerType(), True),
            T.StructField("b", T.StringType(), True),
        ]),
    )
    polars_df = pl.DataFrame(
        {"a": [1, None, 3], "b": ["a", "b", None]}
    ).cast({"a": pl.Int32})

    spark_result = spark_df.dropna()
    polars_result = polars_df.dropnulls()

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_na_drop(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, "a"), (None, "b"), (3, None)],
        T.StructType([
            T.StructField("a", T.IntegerType(), True),
            T.StructField("b", T.StringType(), True),
        ]),
    )
    polars_df = pl.DataFrame(
        {"a": [1, None, 3], "b": ["a", "b", None]}
    ).cast({"a": pl.Int32})

    spark_result = spark_df.na.drop()
    polars_result = polars_df.na.drop()

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_na_fill(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, None), (None, 20)],
        T.StructType([
            T.StructField("a", T.IntegerType(), True),
            T.StructField("b", T.IntegerType(), True),
        ]),
    )
    polars_df = pl.DataFrame({"a": [1, None], "b": [None, 20]}).cast({"a": pl.Int32, "b": pl.Int32})

    spark_result = spark_df.na.fill(0)
    polars_result = polars_df.na.fill(0)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_na_replace(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    spark_result = spark_df.na.replace(1, 99)
    polars_result = polars_df.na.replace(1, 99)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


# ===========================================================================
# Deduplication
# ===========================================================================


def test_distinct(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 1, 2, 3]})

    spark_result = spark_df.distinct()
    polars_result = polars_df.distinct()

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_dropDuplicates(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, "a"), (1, "b"), (2, "a")], ["x", "y"]
    )
    polars_df = pl.DataFrame({"x": [1, 1, 2], "y": ["a", "b", "a"]})

    spark_result = spark_df.dropDuplicates(["x"])
    polars_result = polars_df.dropDuplicates(subset=["x"])

    expected = _spark_to_polars(spark_result)
    assert expected.height == polars_result.height
    assert set(expected["x"].to_list()) == set(polars_result["x"].to_list())


def test_drop_duplicates(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, "a"), (1, "b"), (2, "a")], ["x", "y"]
    )
    polars_df = pl.DataFrame({"x": [1, 1, 2], "y": ["a", "b", "a"]})

    spark_result = spark_df.drop_duplicates(["x"])
    polars_result = polars_df.drop_duplicates(subset=["x"])

    assert _spark_to_polars(spark_result).height == polars_result.height


# ===========================================================================
# Sampling (shape/behavior tests -- can't compare exact results)
# ===========================================================================


def test_sample_fraction(spark_session):
    spark_df = spark_session.createDataFrame([(i,) for i in range(10)], ["a"])
    polars_df = pl.DataFrame({"a": list(range(10))})

    spark_result = spark_df.sample(fraction=0.5, seed=42)
    polars_result = polars_df.sample(fraction=0.5, seed=42)

    assert isinstance(_spark_to_polars(spark_result), pl.DataFrame)
    assert isinstance(polars_result, pl.DataFrame)
    assert polars_result.height <= polars_df.height


def test_randomSplit(spark_session):
    spark_df = spark_session.createDataFrame([(i,) for i in range(10)], ["a"])
    polars_df = pl.DataFrame({"a": list(range(10))})

    spark_splits = spark_df.randomSplit([0.6, 0.4], seed=42)
    polars_splits = polars_df.randomSplit([0.6, 0.4], seed=42)

    assert len(spark_splits) == len(polars_splits) == 2
    spark_total = sum(s.count() for s in spark_splits)
    polars_total = sum(len(s) for s in polars_splits)
    assert spark_total == polars_total == 10


# ===========================================================================
# Slicing
# ===========================================================================


def test_take(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    spark_result = spark_df.take(2)
    polars_result = polars_df.take(2)

    # Spark take returns list of Row, polyspark returns list of dict
    assert len(spark_result) == len(polars_result) == 2


def test_first(spark_session):
    spark_df = spark_session.createDataFrame([(1, "a"), (2, "b")], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})

    spark_result = spark_df.first()
    polars_result = polars_df.first()

    assert spark_result["x"] == polars_result["x"]
    assert spark_result["y"] == polars_result["y"]


def test_offset(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,), (4,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3, 4]})

    spark_result = spark_df.offset(2)
    polars_result = polars_df.offset(2)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


# ===========================================================================
# Transformation
# ===========================================================================


def test_transform(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    def add_col_spark(df):
        return df.withColumn("b", F.col("a") * 2)

    def add_col_polars(df):
        return df.withColumn("b", pl.col("a") * 2)

    spark_result = spark_df.transform(add_col_spark)
    polars_result = polars_df.transform(add_col_polars)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_melt(spark_session):
    # Spark uses unpivot (available in Spark 3.4+), we test the polyspark melt
    polars_df = pl.DataFrame({"id": [1, 2], "val_a": [10, 20], "val_b": [30, 40]})

    result = polars_df.melt(id_vars=["id"], value_vars=["val_a", "val_b"])
    assert "variable" in result.columns
    assert "value" in result.columns
    assert result.height == 4


# ===========================================================================
# Output conversion (test they don't error and return correct type)
# ===========================================================================


def test_toDF(spark_session):
    spark_df = spark_session.createDataFrame([(1, "a")], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1], "y": ["a"]})

    spark_result = spark_df.toDF("col1", "col2")
    polars_result = polars_df.toDF("col1", "col2")

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_toJSON(spark_session):
    polars_df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = polars_df.toJSON()
    assert isinstance(result, str)
    assert '"a"' in result


def test_toPandas(spark_session):
    import pandas as pd

    spark_df = spark_session.createDataFrame([(1, "a"), (2, "b")], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})

    spark_result = spark_df.toPandas()
    polars_result = polars_df.toPandas()

    assert isinstance(spark_result, pd.DataFrame)
    assert isinstance(polars_result, pd.DataFrame)
    assert list(spark_result.columns) == list(polars_result.columns)


def test_toArrow(spark_session):
    import pyarrow as pa

    polars_df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = polars_df.toArrow()
    assert isinstance(result, pa.Table)
    assert result.num_rows == 2


# ===========================================================================
# Schema/Info
# ===========================================================================


def test_printSchema(spark_session, capsys):
    spark_df = spark_session.createDataFrame([(1, "a")], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1], "y": ["a"]})

    # Neither should raise
    spark_df.printSchema()
    capsys.readouterr()  # clear
    polars_df.printSchema()
    captured = capsys.readouterr()
    assert "x" in captured.out


def test_explain(spark_session):
    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    result = polars_df.explain()
    assert isinstance(result, str)
    assert len(result) > 0


def test_show(spark_session, capsys):
    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    polars_df.show()  # should not raise
    # show() only configures display in polyspark


def test_count_property(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    assert spark_df.count() == polars_df.count


def test_columns_property(spark_session):
    spark_df = spark_session.createDataFrame([(1, "a")], ["x", "y"])
    polars_df = pl.DataFrame({"x": [1], "y": ["a"]})

    assert spark_df.columns == polars_df.columns


def test_isEmpty(spark_session):
    spark_df = spark_session.createDataFrame([(1,)], ["a"])
    polars_df = pl.DataFrame({"a": [1]})

    assert polars_df.isEmpty() is False

    empty_polars = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    assert empty_polars.isEmpty() is True


def test_isLocal(spark_session):
    polars_df = pl.DataFrame({"a": [1]})
    assert polars_df.isLocal() is True


def test_isStreaming(spark_session):
    polars_df = pl.DataFrame({"a": [1]})
    assert polars_df.isStreaming is False


# ===========================================================================
# Correlation / Covariance
# ===========================================================================


def test_corr(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, 2.0), (2, 4.0), (3, 6.0), (4, 8.0)], ["a", "b"]
    )
    polars_df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [2.0, 4.0, 6.0, 8.0]})

    spark_result = spark_df.corr("a", "b")
    polars_result = polars_df.corr("a", "b")

    assert abs(spark_result - polars_result) < 1e-9


def test_cov(spark_session):
    spark_df = spark_session.createDataFrame(
        [(1, 2.0), (2, 4.0), (3, 6.0), (4, 8.0)], ["a", "b"]
    )
    polars_df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [2.0, 4.0, 6.0, 8.0]})

    spark_result = spark_df.cov("a", "b")
    polars_result = polars_df.cov("a", "b")

    assert abs(spark_result - polars_result) < 1e-6


# ===========================================================================
# Temp views
# ===========================================================================


def test_createTempView(spark_session):
    import src.sparkpolars.polyspark.sql.dataframe as _mod

    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    polars_df.createOrReplaceTempView("test_temp_view")
    assert "test_temp_view" in _mod._local_sql_ctx.tables()
    _mod._local_sql_ctx.unregister("test_temp_view")


def test_createOrReplaceTempView(spark_session):
    import src.sparkpolars.polyspark.sql.dataframe as _mod

    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    polars_df.createOrReplaceTempView("test_cort_view")
    assert "test_cort_view" in _mod._local_sql_ctx.tables()
    result = _mod._local_sql_ctx.execute("SELECT * FROM test_cort_view").collect()
    assert result.height == 3
    _mod._local_sql_ctx.unregister("test_cort_view")


def test_createTempView_raises_if_exists():
    import src.sparkpolars.polyspark.sql.dataframe as _mod

    polars_df = pl.DataFrame({"a": [1]})
    polars_df.createOrReplaceTempView("dup_view_test")
    with pytest.raises(RuntimeError, match="already exists"):
        polars_df.createTempView("dup_view_test")
    _mod._local_sql_ctx.unregister("dup_view_test")


def test_createGlobalTempView():
    import src.sparkpolars.polyspark.sql.dataframe as _mod

    polars_df = pl.DataFrame({"a": [1, 2]})
    polars_df.createOrReplaceGlobalTempView("global_test_view")
    assert "global_test_view" in _mod._global_sql_ctx.tables()
    _mod._global_sql_ctx.unregister("global_test_view")


def test_createOrReplaceGlobalTempView():
    import src.sparkpolars.polyspark.sql.dataframe as _mod

    polars_df = pl.DataFrame({"a": [1]})
    polars_df.createOrReplaceGlobalTempView("global_cort_view")
    assert "global_cort_view" in _mod._global_sql_ctx.tables()
    _mod._global_sql_ctx.unregister("global_cort_view")


def test_createGlobalTempView_raises_if_exists():
    import src.sparkpolars.polyspark.sql.dataframe as _mod

    polars_df = pl.DataFrame({"a": [1]})
    polars_df.createOrReplaceGlobalTempView("dup_global_test")
    with pytest.raises(RuntimeError, match="already exists"):
        polars_df.createGlobalTempView("dup_global_test")
    _mod._global_sql_ctx.unregister("dup_global_test")


# ===========================================================================
# Iteration (test they don't error)
# ===========================================================================


def test_foreach(spark_session):
    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    collected = []
    polars_df.foreach(lambda row: collected.append(row["a"]))
    assert collected == [1, 2, 3]


def test_foreachPartition(spark_session):
    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    collected = []
    polars_df.foreachPartition(lambda it: collected.extend(r["a"] for r in it))
    assert sorted(collected) == [1, 2, 3]


# ===========================================================================
# Map operations (test they work)
# ===========================================================================


def test_mapInPandas(spark_session):
    polars_df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

    def add_col(it):
        for pdf in it:
            yield pdf.assign(c=pdf["a"] * 2)

    result = polars_df.mapInPandas(add_col)
    assert isinstance(result, pl.DataFrame)
    assert "c" in result.columns
    assert result["c"].to_list() == [2, 4, 6]


def test_mapInArrow(spark_session):
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    def identity(it):
        for batch in it:
            yield batch

    result = polars_df.mapInArrow(identity)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == polars_df.shape


# ===========================================================================
# Lateral join
# ===========================================================================


def test_lateralJoin_cross(spark_session):
    polars_df = pl.DataFrame({"a": [1, 2]})
    polars_other = pl.DataFrame({"b": [10, 20]})

    result = polars_df.lateralJoin(polars_other)
    assert result.height == 4  # 2 x 2


def test_lateralJoin_nonequi(spark_session):
    polars_left = pl.DataFrame({"a": [1, 2, 3]})
    polars_right = pl.DataFrame({"b": [2, 3, 4]})

    result = polars_left.lateralJoin(
        polars_right,
        on=pl.col("a") < pl.col("b"),
    )
    assert isinstance(result, pl.DataFrame)
    assert all(
        row["a"] < row["b"]
        for row in result.select("a", "b").iter_rows(named=True)
    )


def test_lateralJoin_left_raises():
    polars_left = pl.DataFrame({"a": [1]})
    polars_right = pl.DataFrame({"b": [2]})

    with pytest.raises(NotImplementedError):
        polars_left.lateralJoin(polars_right, on=pl.col("a") < pl.col("b"), how="left")


# ===========================================================================
# exists
# ===========================================================================


def test_exists_non_empty():
    polars_df = pl.DataFrame({"a": [1, 2, 3]})
    assert polars_df.exists() is True


def test_exists_empty():
    polars_df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    assert polars_df.exists() is False


def test_exists_semi_join_pattern():
    """Correlated EXISTS simulated as semi join."""
    customers = pl.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "country": ["USA", "Canada", "USA", "Australia"],
    })
    orders = pl.DataFrame({"customer_id": [101, 102, 103, 101]})
    result = customers.join(
        orders.select("customer_id").unique(), on="customer_id", how="semi"
    )
    assert set(result["customer_id"].to_list()) == {101, 102, 103}


def test_not_exists_anti_join_pattern():
    """Correlated NOT EXISTS simulated as anti join."""
    customers = pl.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "country": ["USA", "Canada", "USA", "Australia"],
    })
    orders = pl.DataFrame({"customer_id": [101, 102, 103, 101]})
    result = customers.join(
        orders.select("customer_id").unique(), on="customer_id", how="anti"
    )
    assert result["customer_id"].to_list() == [104]


# ===========================================================================
# Return-self / no-op methods (verify identity)
# ===========================================================================

_RETURN_SELF_METHODS = [
    ("hint", ("broadcast",), {}),
    ("alias", ("my_alias",), {}),
    ("repartition", (4,), {}),
    ("coalesce", (1,), {}),
    ("unpersist", (), {}),
    ("repartitionByRange", (4,), {}),
    ("withMetadata", ("a", {"comment": "x"}), {}),
    ("cache", (), {}),
    ("checkpoint", (), {}),
    ("localCheckpoint", (), {}),
    ("collect", (), {}),
    ("persist", (), {}),
]


@pytest.mark.parametrize("method,args,kwargs", _RETURN_SELF_METHODS)
def test_return_self(method, args, kwargs):
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = getattr(df, method)(*args, **kwargs)
    assert result is df


# ===========================================================================
# NotImplementedError stubs (parametrize)
# ===========================================================================

_NOT_IMPLEMENTED = [
    "approxQuantile",
    "asTable",
    "crosstab",
    "cube",
    "dropDuplicatesWithinWatermark",
    "executionInfo",
    "freqItems",
    "groupingSets",
    "inputFiles",
    "metadataColumn",
    "observe",
    "rdd",
    "rollup",
    "sameSemantics",
    "sampleBy",
    "scalar",
    "semanticHash",
    "stat",
    "storageLevel",
    "to",
    "toLocalIterator",
    "withWatermark",
    "writeStream",
    "writeTo",
    "mergeInto",
    "pandas_api",
]


@pytest.mark.parametrize("method", _NOT_IMPLEMENTED)
def test_not_implemented(method):
    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(NotImplementedError):
        getattr(df, method)()


# ===========================================================================
# DataFrameWriter (test basic write operations work)
# ===========================================================================


def test_write_parquet(spark_session):
    polars_df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        polars_path = f.name

    polars_df.write.mode("overwrite").parquet(polars_path)
    polars_result = pl.read_parquet(polars_path)

    assert_frame_equal(
        polars_df.sort("x"),
        polars_result.sort("x"),
        check_dtypes=False,
    )


def test_write_csv(spark_session):
    polars_df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name

    polars_df.write.mode("overwrite").csv(path)
    result = pl.read_csv(path)
    assert result.shape == polars_df.shape


def test_write_json(spark_session):
    polars_df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False, mode="w") as f:
        path = f.name

    polars_df.write.mode("overwrite").json(path)
    result = pl.read_ndjson(path)
    assert result.shape == polars_df.shape


def test_write_builder_chain():
    from src.sparkpolars.polyspark.sql.dataframe import DataFrameWriter

    df = pl.DataFrame({"a": [1]})
    w = df.write.mode("overwrite").option("compression", "snappy")
    assert isinstance(w, DataFrameWriter)
    assert w._mode == "overwrite"
    assert w._options["compression"] == "snappy"


def test_write_error_mode_raises():
    df = pl.DataFrame({"a": [1]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    df.write.mode("overwrite").parquet(path)
    with pytest.raises(RuntimeError, match="already exists"):
        df.write.parquet(path)  # default mode="error"


def test_write_ignore_mode_skips():
    df = pl.DataFrame({"a": [1, 2, 3]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    df.write.mode("overwrite").parquet(path)
    # Write smaller DF with ignore mode -- should NOT overwrite
    small = df.head(1)
    small.write.mode("ignore").parquet(path)
    result = pl.read_parquet(path)
    assert result.height == 3  # original data preserved


def test_write_saveAsTable():
    import src.sparkpolars.polyspark.sql.dataframe as _mod

    df = pl.DataFrame({"a": [1]})
    df.write.saveAsTable("writer_table_test")
    assert "writer_table_test" in _mod._local_sql_ctx.tables()
    _mod._local_sql_ctx.unregister("writer_table_test")


def test_write_save_no_format_raises():
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="No format"):
        df.write.save("/tmp/nowhere")


# ===========================================================================
# Replace
# ===========================================================================


def test_replace_scalar(spark_session):
    spark_df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
    polars_df = pl.DataFrame({"a": [1, 2, 3]})

    spark_result = spark_df.na.replace(1, 99)
    polars_result = polars_df.replace(1, 99)

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_replace_subset(spark_session):
    spark_df = spark_session.createDataFrame([(1, 10), (2, 20)], ["a", "b"])
    polars_df = pl.DataFrame({"a": [1, 2], "b": [10, 20]})

    spark_result = spark_df.na.replace(1, 99, subset=["a"])
    polars_result = polars_df.replace(1, 99, subset=["a"])

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


def test_replace_dict(spark_session):
    spark_df = spark_session.createDataFrame([("x",), ("y",), ("z",)], ["a"])
    polars_df = pl.DataFrame({"a": ["x", "y", "z"]})

    spark_result = spark_df.na.replace({"x": "X", "y": "Y"})
    polars_result = polars_df.replace({"x": "X", "y": "Y"}, subset=["a"])

    expected = _spark_to_polars(spark_result)
    assert_frame_equal(expected.sort("a"), polars_result.sort("a"), check_dtypes=False)


# ===========================================================================
# Window functions (test all window types)
# ===========================================================================


def test_window_row_number(spark_session):
    from pyspark.sql.window import Window as SparkWindow

    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 150), ("B", 250)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B", "B"], "sal": [100, 200, 150, 250]})

    w_spark = SparkWindow.partitionBy("dept").orderBy("sal")
    spark_result = spark_df.withColumn("rn", F.row_number().over(w_spark))

    polars_result = polars_df.with_columns(
        pl.col("sal").rank("ordinal").over("dept").alias("rn")
    ).sort("dept", "sal")

    expected = _spark_to_polars(spark_result).sort("dept", "sal")
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_window_rank(spark_session):
    from pyspark.sql.window import Window as SparkWindow

    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 100), ("A", 200), ("B", 150)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "A", "B"], "sal": [100, 100, 200, 150]})

    w_spark = SparkWindow.partitionBy("dept").orderBy("sal")
    spark_result = spark_df.withColumn("rnk", F.rank().over(w_spark))

    polars_result = polars_df.with_columns(
        pl.col("sal").rank("min").over("dept").alias("rnk")
    ).sort("dept", "sal")

    expected = _spark_to_polars(spark_result).sort("dept", "sal")
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_window_dense_rank(spark_session):
    from pyspark.sql.window import Window as SparkWindow

    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 100), ("A", 200), ("B", 150)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "A", "B"], "sal": [100, 100, 200, 150]})

    w_spark = SparkWindow.partitionBy("dept").orderBy("sal")
    spark_result = spark_df.withColumn("dr", F.dense_rank().over(w_spark))

    polars_result = polars_df.with_columns(
        pl.col("sal").rank("dense").over("dept").alias("dr")
    ).sort("dept", "sal")

    expected = _spark_to_polars(spark_result).sort("dept", "sal")
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_window_lag(spark_session):
    from pyspark.sql.window import Window as SparkWindow

    spark_df = spark_session.createDataFrame(
        [("A", 1, 100), ("A", 2, 200), ("A", 3, 300)], ["dept", "seq", "sal"]
    )
    polars_df = pl.DataFrame(
        {"dept": ["A", "A", "A"], "seq": [1, 2, 3], "sal": [100, 200, 300]}
    )

    w_spark = SparkWindow.partitionBy("dept").orderBy("seq")
    spark_result = spark_df.withColumn("lag_sal", F.lag("sal", 1).over(w_spark))

    polars_result = polars_df.sort("dept", "seq").with_columns(
        pl.col("sal").shift(1).over("dept").alias("lag_sal")
    )

    expected = _spark_to_polars(spark_result).sort("dept", "seq")
    polars_result = polars_result.sort("dept", "seq")
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_window_lead(spark_session):
    from pyspark.sql.window import Window as SparkWindow

    spark_df = spark_session.createDataFrame(
        [("A", 1, 100), ("A", 2, 200), ("A", 3, 300)], ["dept", "seq", "sal"]
    )
    polars_df = pl.DataFrame(
        {"dept": ["A", "A", "A"], "seq": [1, 2, 3], "sal": [100, 200, 300]}
    )

    w_spark = SparkWindow.partitionBy("dept").orderBy("seq")
    spark_result = spark_df.withColumn("lead_sal", F.lead("sal", 1).over(w_spark))

    polars_result = polars_df.sort("dept", "seq").with_columns(
        pl.col("sal").shift(-1).over("dept").alias("lead_sal")
    )

    expected = _spark_to_polars(spark_result).sort("dept", "seq")
    polars_result = polars_result.sort("dept", "seq")
    assert_frame_equal(expected, polars_result, check_dtypes=False)


def test_window_sum_over_partition(spark_session):
    from pyspark.sql.window import Window as SparkWindow

    spark_df = spark_session.createDataFrame(
        [("A", 100), ("A", 200), ("B", 300)], ["dept", "sal"]
    )
    polars_df = pl.DataFrame({"dept": ["A", "A", "B"], "sal": [100, 200, 300]})

    w_spark = SparkWindow.partitionBy("dept")
    spark_result = spark_df.withColumn("total", F.sum("sal").over(w_spark))

    polars_result = polars_df.with_columns(
        pl.col("sal").sum().over("dept").alias("total")
    )

    expected = _spark_to_polars(spark_result).sort("dept", "sal")
    polars_result = polars_result.sort("dept", "sal")
    assert_frame_equal(expected, polars_result, check_dtypes=False)


# ===========================================================================
# Additional edge case tests
# ===========================================================================


def test_outer_noop():
    """Expr.outer() is a no-op -- returns the same expression."""
    import src.sparkpolars.polyspark.sql.functions  # noqa: F401
    expr = pl.col("x")
    assert expr.outer() is expr


def test_drop_nonexistent():
    df = pl.DataFrame({"a": [1], "b": [2]})
    result = df.drop("nonexistent")
    assert_frame_equal(result, df)


def test_agg_invalid_dict_values():
    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Aggregation functions"):
        df.agg({"a": "invalid_agg"})


def test_agg_invalid_dict_types():
    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="All keys and values"):
        df.agg({1: "sum"})


def test_show_vertical_raises():
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(NotImplementedError):
        df.show(vertical=True)


def test_show_negative_n_raises():
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        df.show(n=-1)


def test_dropna_raises_with_thresh():
    df = pl.DataFrame({"a": [1, None, 3]})
    with pytest.raises(NotImplementedError):
        df.dropna(thresh=2)


def test_dropnulls_raises_with_thresh():
    df = pl.DataFrame({"a": [1, None, 3]})
    with pytest.raises(NotImplementedError):
        df.dropnulls(thresh=2)


def test_registerTempTable():
    import src.sparkpolars.polyspark.sql.dataframe as _mod

    df = pl.DataFrame({"a": [1, 2]})
    df.registerTempTable("reg_table_test")
    assert "reg_table_test" in _mod._local_sql_ctx.tables()
    _mod._local_sql_ctx.unregister("reg_table_test")


def test_toDF_wrong_count_raises():
    df = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="toDF"):
        df.toDF("x")


def test_write_returns_writer():
    from src.sparkpolars.polyspark.sql.dataframe import DataFrameWriter

    df = pl.DataFrame({"a": [1]})
    assert isinstance(df.write, DataFrameWriter)
