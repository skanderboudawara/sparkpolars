from src.sparkpolars._from_polars_to_spark import _convert_schema_polars_to_spark
from src.sparkpolars._from_spark_to_polars import _convert_schema_spark_to_polars
from src.sparkpolars.config import Config


def test_schema_from_spark_to_polars(schema_spark, schema_polars):
    assert _convert_schema_polars_to_spark(schema_polars, Config(map_elements=["cin"])) == schema_spark
    assert _convert_schema_spark_to_polars(schema_spark, tz="UTC") == schema_polars
