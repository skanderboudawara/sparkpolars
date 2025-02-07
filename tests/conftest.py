import pytest
from pyspark.sql import SparkSession, Row
import pyspark.sql.types as T
import polars as pl
from datetime import datetime
from zoneinfo import ZoneInfo

@pytest.fixture(scope="session")
def spark_session():

    spark = SparkSession.builder \
        .appName("pytest") \
        .config("spark.sql.timezone", "Europe/Paris") \
        .getOrCreate()

    yield spark
    spark.stop()
    
@pytest.fixture()
def spark_data():
    data =  [
       Row(name='Alice', age=20, is_student=True, gpa=3.5, courses=['Math', 'Physics'], address=Row(street='123 Main St', city='Springfield', state='IL', zip=62701), created_at=datetime(2020, 1, 1, 1, 1, 1), updated_at=datetime(2020, 1, 1, 1, 1, 1), hobby=[['Skiing', 'Hiking'], ['Reading', 'Writing']], cin={"hello": "world", "another": "fine"}),
       Row(name='Bob', age=21, is_student=False, gpa=3.0, courses=['Math', 'Physics'], address=Row(street='123 Main St', city='Springfield', state='IL', zip=62701), created_at=datetime(2020, 1, 1, 1, 1, 1), updated_at=datetime(2020, 1, 1, 1, 1, 1), hobby=[['Skiing', 'Hiking'], ['Reading', 'Writing']], cin={"hello": "world", "another": "fine"}),
    ]
    
    return data
    
@pytest.fixture()
def polars_data():
    data = [
       {
           'name': 'Alice',
           'age': 20,
           'is_student': True,
           'gpa': 3.5,
           'courses': ['Math','Physics'],
           'address': {'street': '123 Main St', 'city': 'Springfield','state': 'IL','zip': 62701},
           'created_at': datetime(2020, 1, 1, 1, 1, 1),
           'updated_at': datetime(2020, 1, 1, 1, 1, 1),
           'hobby': [['Skiing', 'Hiking'], ['Reading', 'Writing']],
           "cin": {"hello": "world", "another": "fine"},
       },
       {
           'name': 'Bob',
           'age': 21,
           'is_student': False,
           'gpa': 3.0,
           'courses': ['Math','Physics'],
           'address': {'street': '123 Main St', 'city': 'Springfield','state': 'IL','zip': 62701},
           'created_at': datetime(2020, 1, 1, 1, 1, 1),
           'updated_at': datetime(2020, 1, 1, 1, 1, 1),
           'hobby': [['Skiing', 'Hiking'], ['Reading', 'Writing']],
           "cin": {"hello": "world", "another": "fine"},
       },
    ]
    
    return data
@pytest.fixture()
def schema_spark():
    schema=T.StructType(
        [
            T.StructField("name", T.StringType()),
            T.StructField("age", T.IntegerType()),
            T.StructField("is_student", T.BooleanType()),
            T.StructField("gpa", T.FloatType()),
            T.StructField("courses", T.ArrayType(T.StringType())),
            T.StructField("address", T.StructType([
                T.StructField("street", T.StringType()),
                T.StructField("city", T.StringType()),
                T.StructField("state", T.StringType()),
                T.StructField("zip", T.LongType())
            ])),
            T.StructField("created_at", T.TimestampType()),
            T.StructField("updated_at", T.TimestampType()),
            T.StructField("hobby", T.ArrayType(T.ArrayType(T.StringType()))),
            T.StructField("cin", T.MapType(T.StringType(), T.StringType()))
        ]
    )
    return schema
    
@pytest.fixture()
def spark_df(spark_session, schema_spark, spark_data):
    df = spark_session.createDataFrame(
        schema=schema_spark,
        data = spark_data
    )

    return df

@pytest.fixture()
def schema_polars():
    schema={
        'name': pl.String,
        'age': pl.Int32,
        'is_student': pl.Boolean,
        'gpa': pl.Float32,
        'courses': pl.List(pl.String),
        'address': pl.Struct({'street': pl.String, 'city': pl.String, 'state': pl.String, 'zip': pl.Int64}),
        'created_at': pl.Datetime(),
        'updated_at': pl.Datetime(),
        'hobby': pl.List(pl.List(pl.String)),
        "cin": pl.Object(),
    }
    return schema

@pytest.fixture()
def polars_df(schema_polars, polars_data):
    df = pl.DataFrame(
        schema=schema_polars,
        data = polars_data
    )
    
    return df