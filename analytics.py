from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, min, count

spark = SparkSession.builder \
    .appName("NYC_Taxi_Analytics") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.parquet("cleaned_taxi_data")

print("TOTAL TRIPS:", df.count())

print("\nAVERAGE TRIP DISTANCE:")
df.select(avg("trip_distance")).show()

print("\nAVERAGE FARE:")
df.select(avg("fare_amount")).show()

print("\nTRIP DURATION STATISTICS:")
df.select(
    avg("trip_duration"),
    min("trip_duration"),
    max("trip_duration")
).show()

print("\nPAYMENT TYPE ANALYSIS:")
df.groupBy("payment_type") \
    .agg(count("*").alias("total_trips")) \
    .orderBy("total_trips", ascending=False) \
    .show()

print("\nTOP PICKUP LOCATIONS:")
df.groupBy("PULocationID") \
    .count() \
    .orderBy("count", ascending=False) \
    .show(10)

spark.stop()
