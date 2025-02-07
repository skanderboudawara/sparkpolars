import re

import pytest
from polars.testing import assert_frame_equal

from src.config import Config
from src.sparkpolars import to_spark, toPolars


def test_spark_to_polars(spark_session, spark_df, polars_df):
    assert_frame_equal(toPolars(spark_df, lazy=False).drop("cin"), polars_df.drop("cin"))
    assert_frame_equal(toPolars(spark_df, lazy=True).drop("cin"), polars_df.lazy().drop("cin"))

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
