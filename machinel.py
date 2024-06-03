import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv(r"Salary_dataset.csv")
x_train = df["YearsExperience"].values.reshape(-1, 1)
y_train = df["Salary"].values

# Fit the model
model = LinearRegression()
model.fit(x_train, y_train)  # type: ignore
w = model.coef_[0]
b = model.intercept_

predictions = model.predict(x_train)

plt.scatter(x_train, y_train, marker="o", c="pink", label="Actual Values")  # type: ignore
plt.plot(x_train, predictions, c="green", label="Model Prediction")
plt.title("Salary Predictor Model")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (Money)")
plt.legend()
plt.show()

print("Coefficient (w):", w)
print("Intercept (b):", b)
