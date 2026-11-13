from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp

spark = SparkSession.builder \
    .appName("NYC_Taxi_Cleaning") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.parquet("yellow_tripdata_2024-01.parquet")

# Remove missing and invalid records
df = df.dropna(subset=[
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount"
])

df = df.filter(
    (col("trip_distance") > 0) &
    (col("fare_amount") > 0) &
    (col("passenger_count") > 0)
)

# Calculate trip duration in minutes
df = df.withColumn(
    "trip_duration",
    (unix_timestamp("tpep_dropoff_datetime") -
     unix_timestamp("tpep_pickup_datetime")) / 60
)

# Remove unrealistic trip durations
df = df.filter(
    (col("trip_duration") > 0) &
    (col("trip_duration") <= 180)
)

print("CLEANED RECORD COUNT:", df.count())

print("\nCLEANED DATA:")
df.select(
    "trip_distance",
    "fare_amount",
    "trip_duration"
).show(10)

df.write.mode("overwrite").parquet("cleaned_taxi_data")

print("CLEANED DATA SAVED SUCCESSFULLY!")

spark.stop()
