# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     precision_score,
#     recall_score,
#     f1_score,
# )

# # Load Diabetes dataset from Kaggle
# dataset = pd.read_csv("diabetes.csv")

# # giving features (X) and target (y)
# X = dataset.drop("Outcome", axis=1)
# y = dataset["Outcome"]

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# # Create Decision Tree Classifier
# classifier = DecisionTreeClassifier(random_state=42)

# # Train the model
# classifier.fit(X_train, y_train)

# # Make predictions on test data
# y_pred = classifier.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)

# # Calculate confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cm)

# # Accuracy = (TP + TN) / (TP + TN + FP + FN) Total
# # precision = TP / (TP + FP)
# # RECALL = TP / (TP+ FN)
# # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
# # true positive instances that are correctly predicted by the model.

# # Calculate precision, recall, and F-measure
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F-measure:", f1)


# # Make predictions based on user input
# def make_prediction(input_data):
#     # Convert input data to DataFrame
#     input_df = pd.DataFrame(input_data, columns=X.columns)
#     print(input)
#     # Make prediction
#     prediction = classifier.predict(input_df)

#     return prediction


# # Example user input
# user_input = {
#     "Pregnancies": [5],
#     "Glucose": [166],
#     "BloodPressure": [72],
#     "SkinThickness": [19],
#     "Insulin": [175],
#     "BMI": [25.8],
#     "DiabetesPedigreeFunction": [0.587],
#     "Age": [51],
# }

# # Make prediction
# prediction = make_prediction(user_input)
# print("Prediction:", prediction)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Red Wine Quality dataset from Kaggle
dataset = pd.read_csv("WineQt.csv")

# Check column names
print(dataset.columns)

# Split dataset into features (X) and target (y)
X = dataset.drop(["quality", "Id"], axis=1)
y = dataset["quality"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Make predictions on test data
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# Make predictions based on user input
def make_prediction(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, columns=X.columns)

    # Make prediction
    prediction = classifier.predict(input_df)

    return prediction


# Example user input
user_input = {
    "fixed acidity": [7.4],
    "volatile acidity": [0.7],
    "citric acid": [0.0],
    "residual sugar": [1.9],
    "chlorides": [0.076],
    "free sulfur dioxide": [11],
    "total sulfur dioxide": [34],
    "density": [0.9978],
    "pH": [3.51],
    "sulphates": [0.56],
    "alcohol": [9.4],
}

# Make prediction
prediction = make_prediction(user_input)
print("Prediction:", prediction)
