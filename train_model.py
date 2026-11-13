from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, dayofweek, month
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("NYC_Taxi_ML") \
    .master("local[*]") \
    .getOrCreate()

# Load cleaned dataset
df = spark.read.parquet("cleaned_taxi_data")

# Feature engineering
df = df.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
df = df.withColumn("pickup_day", dayofweek("tpep_pickup_datetime"))
df = df.withColumn("pickup_month", month("tpep_pickup_datetime"))

# Select features and target
data = df.select(
    "trip_distance",
    "fare_amount",
    "passenger_count",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "trip_duration"
)

# Limit training data for laptop performance
data = data.limit(300000)

assembler = VectorAssembler(
    inputCols=[
        "trip_distance",
        "fare_amount",
        "passenger_count",
        "pickup_hour",
        "pickup_day",
        "pickup_month"
    ],
    outputCol="features"
)

data = assembler.transform(data).select("features", "trip_duration")

train, test = data.randomSplit([0.8, 0.2], seed=42)

evaluator = RegressionEvaluator(
    labelCol="trip_duration",
    predictionCol="prediction",
    metricName="rmse"
)

# Linear Regression
lr = LinearRegression(
    featuresCol="features",
    labelCol="trip_duration"
)

lr_model = lr.fit(train)
lr_predictions = lr_model.transform(test)

lr_rmse = evaluator.evaluate(lr_predictions)

# Random Forest
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="trip_duration",
    numTrees=20,
    maxDepth=8,
    seed=42
)

rf_model = rf.fit(train)
rf_predictions = rf_model.transform(test)

rf_rmse = evaluator.evaluate(rf_predictions)

# Calculate MAE
mae_evaluator = RegressionEvaluator(
    labelCol="trip_duration",
    predictionCol="prediction",
    metricName="mae"
)

lr_mae = mae_evaluator.evaluate(lr_predictions)
rf_mae = mae_evaluator.evaluate(rf_predictions)

print("\n========== MODEL RESULTS ==========")
print("LINEAR REGRESSION RMSE:", lr_rmse)
print("LINEAR REGRESSION MAE:", lr_mae)
print("RANDOM FOREST RMSE:", rf_rmse)
print("RANDOM FOREST MAE:", rf_mae)

# Save models
lr_model.write().overwrite().save("linear_regression_model")
rf_model.write().overwrite().save("random_forest_model")

print("\nBOTH MODELS SAVED SUCCESSFULLY!")

if rf_rmse < lr_rmse:
    print("BEST MODEL: RANDOM FOREST REGRESSOR")
else:
    print("BEST MODEL: LINEAR REGRESSION")

spark.stop()
