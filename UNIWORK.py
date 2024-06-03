import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Load dataset
data = pd.read_csv("Diabetespred.csv")

# Define features and target variable
X = data[
    [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
]
y = data["Outcome"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create a logistic regression model
model = LogisticRegression(
    random_state=42, C=0.5, multi_class="multinomial", solver="lbfgs", max_iter=200 # type: ignore
)

# Train the model
model.fit(X_train, y_train)

logits = model.decision_function(X_test)

# Calculate sigmoid probabilities manually (for demonstration)
sigmoid_probabilities = sigmoid(logits)

# Make predictions based on the maximum probability
predictions = (sigmoid_probabilities > 0.55).astype(int)


# Predict on the test set
# predictions = model.predict(X_test_imputed)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)


# def calculate_fuel_remaining(current_fuel, distance, fuel_efficiency):
#     fuel_used = distance / fuel_efficiency
#     remaining_fuel = current_fuel - fuel_used
#     return remaining_fuel

# def main():
#     current_fuel = float(input("Enter current fuel level: "))
#     fuel_efficiency = 11

#     while current_fuel > 0:
#         distance_to_next_station = float(input("Enter distance to next gas station: "))
#         min_distance = min(distance_to_next_station, current_fuel * fuel_efficiency)
#         max_distance = current_fuel * fuel_efficiency
#         current_fuel = calculate_fuel_remaining(current_fuel, min_distance, fuel_efficiency)

#         if current_fuel < 0:
#             print("Warning: Fuel is not enough to reach the next station!")
#             break
#         else:
#             print("Traveled distance:", min_distance)
#             print("Remaining fuel:", current_fuel)
#             if min_distance < distance_to_next_station:
#                 print("Warning: Fuel is not enough to reach the next station!")

# import pandas as pd
# from sklearn.datasets import load_wine

# data = load_wine()
# X = data.data  # Using all features
# y = data.target

# # Convert NumPy array to pandas DataFrame
# df = pd.DataFrame(X,columns=data.feature_names)

# # Display the column names
# print(df.columns)
