# sparkpolars

This is a lightweight library that doesn't require any dependency to run. (Only when requested)

## Installation

`pip install sparkpolars`  *waiting for first release*

## Requirements

- Minimum python required 3.10
- Pyspark and Polars should be already installed if you are going to use this library
- Minimum spark version 3.3.0
- Minimum polars version 1.0

## Why This library exists ?
```python
# usually the conversion is done spark -> to pandas -> polars from pandas
# or
# polars -> to pandas -> spark from pandas
```
- **Removing any dependency** such us `Pandas` or `Pyarrow`. Using native functions such us `collect` and `schema` interpretations.
- When it comes to complex types such us `MapType`, `StructType`, and Nested `ArrayType` the conversion is hit or miss and inconsistent

## Features

- Convert a spark DataFrame to Polars DataFrame or LazyFrame
- Consistency in schema conversion, sometimes `LongType` can be converted to `Int32` relying on pandas interpretation of your data or it should be `Int64`. With `sparkpolars` you can be assured that `LongType` stays `Int64`
- 3 modes of conversion `NATIVE`, `ARROW`, `PANDAS`
- Using `NATIVE` will guarantee the conversion of `MapType`, `StructType`, and Nested `ArrayType`
- Using `ARROW`, `PANDAS` comes with the limitations of the library converting complex types
- Using `Config` to precise the columns to convert from polars `list(struct)` to spark `MapType`
- Using `Config` to precise the `time_zone` and `time_unit` for polars `Datetime`

## Usage:

1. from spark to polars DataFrame

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("example").getOrCreate()

df = spark.createDataFrame([(1, 2)], ["a", "b"])

polars_df = df.toPolars()
```

2. from spark to polars LazyFrame

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("example").getOrCreate()

df = spark.createDataFrame([(1, 2)], ["a", "b"])

polars_df = df.toPolars(lazy=True)
```

3. from polars DataFrame to spark

```python
from pyspark.sql import SparkSession
from polars import DataFrame

spark = SparkSession.builder.appName("example").getOrCreate()

df = spark.DataFrame({"a": [1], "b": [2]})  # It also be a LazyDataFrame

spark_df = df.to_spark(spark=spark)
# or 
spark_df = df.to_spark()  # It will try to get the spark ActiveSession
```

4. Using specific mode
   
```python

from sparkpolars import ModeMethod

spark_df = df.to_spark(mode=ModeMethod.NATIVE)
spark_df = df.to_spark(mode=ModeMethod.PANDAS)
spark_df = df.to_spark(mode=ModeMethod.ARROW)

polars_df = df.toPolars(mode=ModeMethod.NATIVE)
polars_df = df.toPolars(mode=ModeMethod.PANDAS)
polars_df = df.toPolars(mode=ModeMethod.ARROW)
```

4. Using Config
```python

from sparkpolars import Config

conf = Config(
    map_elements=["column_should_be_converted_to_map_type", ...],  # the list of columns that should be converted to MapType
    time_unit="ms", # Literal["ns", "us", "ms"] default to "us"
)
spark_df = df.to_spark(config=conf)

polars_df = df.toPolars(config=conf)
```

## Known limitations

### JVM Timezone:

The JVM Timezone can be different from the spark TimeZone 

When collecting Data to memory the spark data will be collected through the JVM and convert all the `TimestampType` to JVM timezone. If there is any discrepancy, then you should verify the timezone of your JVM 

### Memory:

As form using `pandas` or `collect` collecting spark data to memory comes with its known limitations. If the data collected exceeds the Driver Memory then you will have issues.

### MapType:

The main reason of this library existence is to handle MapType

#### From spark to polars
If you have in spark 

Type : `StructField("example", MapType(StringType(), IntegerType())}`
Data :  `{"a": 1, "b": 2}`

Then It will become on polars

Type : `{"example": List(Struct("key": String, "value": Int32))}`
Data : `[{"key": "a", "value": 1}, {"key: "b", "value": 2}]`

### from polars to spark
If you have in polars 

Type : `{"example": List(Struct("key": String, "value": Int32))}`
Data : `[{"key": "a", "value": 1}, {"key: "b", "value": 2}]`

Then It will become on polars without specifying any config (Default Behavior)

Type : `StructField("example", ArrayType(StructType(StructField("key", StringType())), StructField("value", IntegerType())))`
Data : `[{"key": "a", "value": 1}, {"key: "b", "value": 2}]`

If you want this data to be converted to MapType

```python
from sparkpolars import Config
conf = Config(
    map_elements=["example"]
)
```
Type : `StructField("example", MapType(StringType(), IntegerType())}`
Data :  `{"a": 1, "b": 2}`


## License
- pending

## Contribution
- pending