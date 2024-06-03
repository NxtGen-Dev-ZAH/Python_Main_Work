import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Load dataset
data = pd.read_csv("Social_Network_Ads.csv")

# Convert 'Gender' column to numeric
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

# Define features and target variable
X = data[["UserID", "Gender", "Age", "EstimatedSalary"]]
y = data["Purchased"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Impute missing values in the test set using mean strategy
imputer = SimpleImputer(strategy="mean")
X_test_imputed = imputer.fit_transform(X_test)

# Create a logistic regression model
model = LogisticRegression(
    random_state=42, C=0.5, multi_class="multinomial", solver="lbfgs"
)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test_imputed)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)

# # Predict probabilities on the test set
# logits = model.decision_function(X_test_imputed)

# # Calculate sigmoid probabilities manually
# sigmoid_probabilities = sigmoid(logits)
# # Reshape sigmoid_probabilities to ensure it's 2D
# sigmoid_probabilities = sigmoid_probabilities.reshape(-1, 1)

# # Make predictions based on the maximum probability
# predictions = np.argmax(sigmoid_probabilities, axis=1)

# print(predictions)


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.impute import SimpleImputer


# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))


# # Load dataset
# data = pd.read_csv("Diabetespred.csv")

# # Define features and target variable
# X = data[
#     [
#         "Pregnancies",
#         "Glucose",
#         "BloodPressure",
#         "SkinThickness",
#         "Insulin",
#         "BMI",
#         "DiabetesPedigreeFunction",
#         "Age",
#     ]
# ]
# y = data["Outcome"]

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )


# # Create a logistic regression model
# model = LogisticRegression(
#     random_state=42, C=0.5, multi_class="multinomial", solver="lbfgs",max_iter=200
# )

# # Train the model
# model.fit(X_train, y_train)

# logits = model.decision_function(X_test)

# # Calculate sigmoid probabilities manually (for demonstration)
# sigmoid_probabilities = sigmoid(logits)

# # Make predictions based on the maximum probability
# predictions = (sigmoid_probabilities > 0.55).astype(int)


# # Predict on the test set
# # predictions = model.predict(X_test_imputed)

# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# conf_matrix = confusion_matrix(y_test, predictions)

# # Print the results
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print("Confusion Matrix:\n", conf_matrix)

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.impute import SimpleImputer


# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))


# # Load dataset
# data = pd.read_csv("car_data.csv")

# # Convert 'Gender' column to numeric
# data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

# # Define features and target variable
# X = data[["UserID", "Gender", "Age", "AnnualSalary"]]
# y = data["Purchased"]

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=46
# )

# # Impute missing values in the test set using mean strategy
# imputer = SimpleImputer(strategy="mean")
# X_test_imputed = imputer.fit_transform(X_test)

# # Create a logistic regression model
# model = LogisticRegression(
#     random_state=46, C=0.6, multi_class="multinomial", solver="lbfgs", max_iter=300
# )

# # Train the model
# model.fit(X_train, y_train)

# # Predict on the test set
# logits = model.decision_function(X_test)

# # Calculate sigmoid probabilities manually (for demonstration)
# sigmoid_probabilities = sigmoid(logits)

# # Make predictions based on the maximum probability
# predictions = (sigmoid_probabilities > 0.5).astype(int)


# # Predict on the test set
# # predictions = model.predict(X_test_imputed)
# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# conf_matrix = confusion_matrix(y_test, predictions)

# # Print the results
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print("Confusion Matrix:\n", conf_matrix)
