import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

from src.sparkpolars.config import Config
from src.sparkpolars.sparkpolars import ModeMethod, to_spark, toPolars


def test_spark_to_polars_native(spark_session, spark_df, polars_df):
    assert_frame_equal(toPolars(spark_df, lazy=False), polars_df)
    assert_frame_equal(toPolars(spark_df, lazy=True), polars_df.lazy())

    data_spark = to_spark(polars_df, config=Config(map_elements=["cin"]))
    assert data_spark.schema == spark_df.schema
    assert data_spark.collect() == spark_df.collect()

    data_spark = to_spark(polars_df.lazy(), config=Config(map_elements=["cin"]))
    assert data_spark.schema == spark_df.schema
    assert data_spark.collect() == spark_df.collect()

    with pytest.raises(NotImplementedError, match=re.escape("Method not implemented.")):
        toPolars(spark_df, mode="not implemented")
    with pytest.raises(NotImplementedError, match=re.escape("Method not implemented.")):
        to_spark(polars_df, mode="not implemented")


def test_spark_to_polars_using_arrow(spark_session, spark_df, polars_df):
    pl_data = toPolars(
        spark_df,
        lazy=False,
        mode=ModeMethod.ARROW,
    )
    assert_frame_equal(pl_data, polars_df)


def test_spark_to_polars_using_pandas(spark_session, spark_df, polars_df):
    pl_data = toPolars(
        spark_df.drop("cin", "address"),  # Struct and Map types are not supported in pandas
        lazy=False,
        mode=ModeMethod.PANDAS,
    )
    assert_frame_equal(pl_data, polars_df.drop("cin", "address"))


def test_polars_to_spark_using_pandas(spark_session, spark_df, polars_df):
    data_spark = to_spark(
        polars_df.drop("cin"),  # pandas does not support Map type
        mode=ModeMethod.PANDAS,
    )
    assert data_spark.schema == spark_df.drop("cin").schema
    assert data_spark.collect() == spark_df.drop("cin").collect()


def test_polars_to_spark_using_arrow(spark_session, spark_df, polars_df):
    data_spark = to_spark(
        polars_df.drop("cin"),
        mode=ModeMethod.ARROW,
    )
    assert data_spark.schema == spark_df.drop("cin").schema
    assert data_spark.collect() == spark_df.drop("cin").collect()


def test_empty_spark_dataframe(spark_session):
    schema = StructType([StructField("a", IntegerType()), StructField("b", StringType())])
    df = spark_session.createDataFrame([], schema=schema)
    pl_df = df.toPolars()

    expected_polars = pl.DataFrame(
        schema={
            "a": pl.Int32,
            "b": pl.Utf8,
        },
        data={
            "a": [],
            "b": [],
        },
    )

    assert_frame_equal(pl_df, expected_polars)


def test_empty_polars_dataframe(spark_session):

    pl_df = pl.DataFrame(
        schema={
            "a": pl.Int32,
            "b": pl.Utf8,
        },
        data={
            "a": [],
            "b": [],
        },
    )
    df = pl_df.to_spark()

    schema = StructType([StructField("a", IntegerType()), StructField("b", StringType())])
    expected_df = spark_session.createDataFrame([], schema=schema)

    assert df.schema == expected_df.schema
    assert df.collect() == expected_df.collect()


def test_with_array_struct(spark_session):

    schema = StructType([
        StructField("a", StringType()),
        StructField("b", ArrayType(StructType([
            StructField("c", IntegerType()),
            StructField("d", StringType()),
    ])))])

    data = [
        ("a", [(1, "b"), (2, "c")]),
        ("b", [(3, "d")]),
    ]

    df = spark_session.createDataFrame(data, schema=schema)
    pl_df = df.toPolars()

    pl_expected = pl.DataFrame(
        schema={
            "a": pl.Utf8,
            "b": pl.List(pl.Struct({"c": pl.Int32, "d": pl.Utf8})),
        },
        data={
            "a": ["a", "b"],
            "b": [[(1, "b"), (2, "c")], [(3, "d")]],
        },
    )

    assert_frame_equal(pl_df, pl_expected)
