import matplotlib.pyplot as plt

models = ["Linear Regression", "Random Forest"]
rmse = [7.1115, 5.2248]
mae = [4.1266, 2.5704]

plt.figure(figsize=(8, 5))
plt.bar(models, rmse)
plt.title("NYC Taxi Model Comparison - RMSE")
plt.ylabel("RMSE (Minutes)")
plt.savefig("model_comparison.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(models, mae)
plt.title("NYC Taxi Model Comparison - MAE")
plt.ylabel("MAE (Minutes)")
plt.savefig("mae_comparison.png")
plt.show()

print("COMPARISON GRAPHS SAVED SUCCESSFULLY!")
