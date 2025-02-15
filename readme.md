# sparkpolars

**sparkpolars** is a lightweight library designed for seamless conversions between Apache Spark and Polars without unnecessary dependencies. (Dependencies are only required when explicitly requested.)

## Installation

```sh
pip install sparkpolars
# or
conda install skandev::sparkpolars
```

## Requirements

- **Python** ≥ 3.10
- **Apache Spark** ≥ 3.3.0 (must be pre-installed)
- **Polars** ≥ 1.0 (must be pre-installed)
- **Pyspark** must also be installed if you plan to use this library

## Why Does This Library Exist?

### The Problem

Typical conversions between Spark and Polars often involve an intermediate Pandas step:

```python
# Traditional approach:
# Spark -> Pandas -> Polars
# or
# Polars -> Pandas -> Spark
```

### The Solution

**sparkpolars** eliminates unnecessary dependencies like `pandas` and `pyarrow` by leveraging native functions such as `.collect()` and schema interpretation.

### Key Benefits

- 🚀 **No extra dependencies** – No need for Pandas or PyArrow
- ✅ **Reliable handling of complex types** – Provides better consistency for `MapType`, `StructType`, and nested `ArrayType`, where existing conversion methods can be unreliable

## Features

- Convert a Spark DataFrame to a Polars DataFrame or LazyFrame
- Ensures schema consistency: preserves `LongType` as `Int64` instead of mistakenly converting to `Int32`
- Three conversion modes: `NATIVE`, `ARROW`, `PANDAS`
- `NATIVE` mode properly converts `MapType`, `StructType`, and nested `ArrayType`
- `ARROW` and `PANDAS` modes may have limitations with complex types
- Configurable conversion settings for Polars `list(struct)` to Spark `MapType`
- Timezone and time unit customization for Polars `Datetime`

## Usage

### 0. Supercharge Polars and Spark DataFrame

In your `__init__.py` file at the root project you can do the following for ease of use

```python
from sparkpolars import toPolars, to_spark
from pyspark.sql import DataFrame as SparkDataFrame
from polars import DataFrame as PolarsDataFrame, LazyFrame as PolarsLazyFrame

__all__ = [
    "toPolars",
    "to_spark",
]

SparkDataFrame.toPolars = toPolars
PolarsDataFrame.to_spark = to_spark
PolarsLazyFrame.to_spark = to_spark
```

### 1. From Spark to Polars DataFrame

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("example").getOrCreate()

df = spark.createDataFrame([(1, 2)], ["a", "b"])

polars_df = df.toPolars()
```

### 2. From Spark to Polars LazyFrame

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("example").getOrCreate()

df = spark.createDataFrame([(1, 2)], ["a", "b"])

polars_df = df.toPolars(lazy=True)
```

### 3. From Polars DataFrame to Spark

```python
from pyspark.sql import SparkSession
from polars import DataFrame

spark = SparkSession.builder.appName("example").getOrCreate()

df = DataFrame({"a": [1], "b": [2]})  # It can also be a LazyDataFrame

spark_df = df.to_spark(spark=spark)
# or
spark_df = df.to_spark()  # It will try to get the Spark ActiveSession
```

### 4. Using Specific Mode

```python
from sparkpolars import ModeMethod

spark_df = df.to_spark(mode=ModeMethod.NATIVE)
spark_df = df.to_spark(mode=ModeMethod.PANDAS)
spark_df = df.to_spark(mode=ModeMethod.ARROW)

polars_df = df.toPolars(mode=ModeMethod.NATIVE)
polars_df = df.toPolars(mode=ModeMethod.PANDAS)
polars_df = df.toPolars(mode=ModeMethod.ARROW)
```

### 5. Using Config

```python
from sparkpolars import Config

conf = Config(
    map_elements=["column_should_be_converted_to_map_type", ...],  # Specify columns to convert to MapType
    time_unit="ms",  # Literal["ns", "us", "ms"], defaults to "us"
)
spark_df = df.to_spark(config=conf)

polars_df = df.toPolars(config=conf)
```

## Known Limitations

### JVM Timezone Discrepancy

Spark timestamps are collected via the JVM, which may differ from Spark’s timezone settings. If issues arise, verify the JVM timezone.

### Memory Constraints

Collecting large datasets into memory can exceed available driver memory, leading to failures. (as for pandas/arrow)

### Handling `MapType`:


#### From Spark to Polars
If you have in Spark:

Type: `StructField("example", MapType(StringType(), IntegerType()))`

Data:  `{"a": 1, "b": 2}`

Then it will become in Polars:

Type: `{"example": List(Struct("key": String, "value": Int32))}`

Data: `[{"key": "a", "value": 1}, {"key": "b", "value": 2}]`

#### From Polars to Spark
If you have in Polars:

Type: `{"example": List(Struct("key": String, "value": Int32))}`

Data: `[{"key": "a", "value": 1}, {"key": "b", "value": 2}]`

Then it will become in Spark without specifying any config (Default Behavior):

Type: `StructField("example", ArrayType(StructType(StructField("key", StringType())), StructField("value", IntegerType())))`

Data: `[{"key": "a", "value": 1}, {"key": "b", "value": 2}]`

If you want this data to be converted to MapType:

```python
from sparkpolars import Config
conf = Config(
    map_elements=["example"]
)
```

Type: `StructField("example", MapType(StringType(), IntegerType()))`

Data:  `{"a": 1, "b": 2}`

## License
- MIT License

## Contribution
- Create an associated issue, or assign yourself to an existing issue
- Fork the project
- Install all the dependencies `pip install ".[dev,lint,test]`
- Install pre-commit file `pre-commit install`
- Develop your feature
- Unit-test your feature
- Create a Pull request
