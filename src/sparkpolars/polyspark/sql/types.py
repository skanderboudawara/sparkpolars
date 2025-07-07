"""Polars SQL types for Polyspark."""

import polars as pl

# Create StringType alias for polars String type
StringType = pl.String
ByteType = pl.Int8
ShortType = pl.Int16
IntegerType = pl.Int32
LongType = pl.Int64
BooleanType = pl.Boolean
FloatType = pl.Float32
DateType = pl.Date
DoubleType = pl.Float64
BinaryType = pl.Binary
NullType = pl.Null
TimestampNTZType = pl.Datetime
