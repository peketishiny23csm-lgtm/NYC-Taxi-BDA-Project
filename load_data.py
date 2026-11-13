from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("NYC_Taxi_BDA") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.parquet("yellow_tripdata_2024-01.parquet")

print("TOTAL RECORDS:", df.count())

print("\nDATASET COLUMNS:")
df.printSchema()

print("\nFIRST 5 RECORDS:")
df.show(5, truncate=False)

spark.stop()
