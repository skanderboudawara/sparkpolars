
import polars as pl
import pyspark.sql.types as T
import pytest

from src.sparkpolars._from_polars_to_spark import (
    _convert_array_to_list,
    _type_convert_polars_to_spark,
)
from src.sparkpolars._from_spark_to_polars import _type_convert_pyspark_to_polars
from src.sparkpolars.config import Config


@pytest.mark.parametrize(("pyspark_type", "expected_polars_type"), [
    (T.StringType(), pl.String),
    (T.IntegerType(), pl.Int32),
    (T.LongType(), pl.Int64),
    (T.ShortType(), pl.Int16),
    (T.ByteType(), pl.Int8),
    (T.BooleanType(), pl.Boolean),
    (T.BinaryType(), pl.Binary),
    (T.DecimalType(16, 10), pl.Decimal(16, 10)),
    (T.DecimalType(5, 5), pl.Decimal(5, 5)),
    (T.TimestampType(), pl.Datetime),
    (T.ArrayType(T.StringType()), pl.List(pl.String)),
    (T.ArrayType(T.ArrayType(T.StringType())), pl.List(pl.List(pl.String))),
    (T.MapType(T.StringType(), T.StringType()), pl.List(pl.Struct({"key": pl.String, "value": pl.String}))),
    (T.ArrayType(T.ArrayType(T.ArrayType(T.IntegerType()))), pl.List(pl.List(pl.List(pl.Int32)))),
    (T.StructType([T.StructField("street", T.StringType()), T.StructField("city", T.StringType()), T.StructField("state", T.StringType()), T.StructField("zip", T.LongType())]), pl.Struct({"street": pl.String, "city": pl.String, "state": pl.String, "zip": pl.Int64})),
    (T.StructType([T.StructField("street", T.StringType()), T.StructField("city", T.StringType()), T.StructField("state", T.ArrayType(T.ArrayType(T.StringType()))), T.StructField("zip", T.LongType())]), pl.Struct({"street": pl.String, "city": pl.String, "state": pl.List(pl.List(pl.String)), "zip": pl.Int64})),
])
def test__type_convert_pyspark_to_polars(pyspark_type, expected_polars_type):
    assert _type_convert_pyspark_to_polars(pyspark_type) == expected_polars_type


def test__type_convert_pyspark_to_polars_unknown():
    with pytest.raises(ValueError, match="Unsupported type: 'unknown'"):
        _type_convert_pyspark_to_polars("unknown")


def test__test_convert_spark_to_polars_datetime():
    assert _type_convert_pyspark_to_polars(T.TimestampType(), Config(time_unit="ns")) == pl.Datetime("ns")
    assert _type_convert_pyspark_to_polars(T.TimestampType(), tz="Africa/Tunis") == pl.Datetime(time_zone="Africa/Tunis")


@pytest.mark.parametrize(("is_map", "polar_type", "expected_spark_type"), [
    (False, pl.String, T.StringType()),
    (False, pl.Int32, T.IntegerType()),
    (False, pl.Int64, T.LongType()),
    (False, pl.Int16, T.ShortType()),
    (False, pl.Int8, T.ByteType()),
    (False, pl.UInt32, T.IntegerType()),
    (False, pl.UInt64, T.LongType()),
    (False, pl.UInt16, T.ShortType()),
    (False, pl.UInt8, T.ByteType()),
    (False, pl.Boolean, T.BooleanType()),
    (False, pl.Binary, T.BinaryType()),
    (False, pl.Datetime(), T.TimestampType()),
    (False, pl.Decimal(16, 10), T.DecimalType(16, 10)),
    (False, pl.Decimal(5, 5), T.DecimalType(5, 5)),
    (False, pl.List(pl.String), T.ArrayType(T.StringType())),
    (False, pl.List(pl.List(pl.String)), T.ArrayType(T.ArrayType(T.StringType()))),
    (False, pl.List(pl.List(pl.List(pl.Int32))), T.ArrayType(T.ArrayType(T.ArrayType(T.IntegerType())))),
    (None , pl.Struct({"street": pl.String, "city": pl.String, "state": pl.String, "zip": pl.Int64}), T.StructType([T.StructField("street", T.StringType()), T.StructField("city", T.StringType()), T.StructField("state", T.StringType()), T.StructField("zip", T.LongType())])),
    (False, pl.Struct({"street": pl.String, "city": pl.String, "state": pl.List(pl.List(pl.String)), "zip": pl.Int64}), T.StructType([T.StructField("street", T.StringType()), T.StructField("city", T.StringType()), T.StructField("state", T.ArrayType(T.ArrayType(T.StringType()))), T.StructField("zip", T.LongType())])),
    (True , pl.List(pl.Struct({"key": pl.String, "value": pl.String})), T.MapType(T.StringType(), T.StringType())),
])
def test__type_convert_polars_to_spark(is_map, polar_type, expected_spark_type):
    assert _type_convert_polars_to_spark(polar_type, is_map=is_map) == expected_spark_type


def test__type_convert_polars_to_spark_unknown():
    with pytest.raises(ValueError, match="Unsupported type: 'unknown'"):
        _type_convert_polars_to_spark("unknown")
    polar_type = pl.List(pl.Struct({"pey": pl.String, "value": pl.String}))
    with pytest.raises(ValueError, match=r"Only types List\(Struct\(key: Type, value: Type\)\) are supported for maps."):
        _type_convert_polars_to_spark(polar_type, is_map=True)
    polar_type = pl.List(pl.Struct({"key": pl.String, "falue": pl.String}))
    with pytest.raises(ValueError, match=r"Only types List\(Struct\(key: Type, value: Type\)\) are supported for maps."):
        _type_convert_polars_to_spark(polar_type, is_map=True)
    polar_type = pl.List(pl.Struct({"key": pl.String, "value": pl.String, "another": pl.String}))
    with pytest.raises(ValueError, match=r"Only types List\(Struct\(key: Type, value: Type\)\) are supported for maps."):
        _type_convert_polars_to_spark(polar_type, is_map=True)
    polar_type = pl.List(pl.List(pl.Int32))
    with pytest.raises(ValueError, match=r"Only types List\(Struct\(key: Type, value: Type\)\) are supported for maps."):
        _type_convert_polars_to_spark(polar_type, is_map=True)


def test_convert_array_to_list():
    df = pl.DataFrame(
        data={
            "foo": [[1, 2, 3]],
        },
        schema={
            "foo": pl.Array(pl.Int32, 3),
        },
    )
    df = _convert_array_to_list(df)
    schema = df.collect_schema()
    schema == {"foo": pl.List(pl.Int32)}  # noqa: B015
