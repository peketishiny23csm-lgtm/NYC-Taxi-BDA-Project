# NYC Taxi Trip Analytics and Duration Prediction

## Project Overview
This project analyzes NYC Yellow Taxi trip records for January 2024 using Apache Spark and applies machine learning models to estimate trip duration.

## Technologies Used
- Python
- Apache Spark (PySpark)
- Pandas
- Matplotlib
- Scikit-learn
- Java

## Dataset
NYC Yellow Taxi Trip Records – January 2024

Source: NYC Taxi and Limousine Commission (NYC TLC)

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Project Workflow
1. Load the taxi dataset using Apache Spark.
2. Clean the data.
3. Perform data analytics.
4. Extract useful features.
5. Train Linear Regression and Random Forest models.
6. Evaluate model performance using RMSE and MAE.
7. Visualize the model comparison.

## Results

| Model | RMSE | MAE |
|---|---:|---:|
| Linear Regression | 7.11 | 4.13 |
| Random Forest Regressor | 5.22 | 2.57 |

## Best Performing Model
Random Forest Regressor achieved lower RMSE and MAE values.

## Conclusion
Apache Spark was used to process the taxi dataset, and machine learning models were trained to estimate trip duration. Random Forest Regressor performed better than Linear Regression in this experiment.
