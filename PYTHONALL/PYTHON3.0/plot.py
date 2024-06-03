import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


def decision_boundary(X, y, model, iris, two=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    if two:
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        for i, color in zip(np.unique(y), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, s=15)
        plt.show()

    else:
        set_ = {0, 1, 2}
        print(set_)
        for i, color in zip(range(3), plot_colors):
            idx = np.where(y == i)
            if np.any(idx):
                set_.remove(i)

                plt.scatter(
                    X[idx, 0],
                    X[idx, 1],
                    label=y,
                    cmap=plt.cm.RdYlBu,
                    edgecolor="black",
                    s=15,
                )

        for i in set_:
            idx = np.where(iris.target == i)
            plt.scatter(X[idx, 0], X[idx, 1], marker="x", color="black")

        plt.show()


def plot_probability_array(X, probability_array):
    plot_array = np.zeros((X.shape[0], 30))
    col_start = 0
    ones = np.ones((X.shape[0], 30))
    for class_, col_end in enumerate([10, 20, 30]):
        plot_array[:, col_start:col_end] = np.repeat(
            probability_array[:, class_].reshape(-1, 1), 10, axis=1
        )
        col_start = col_end
    plt.imshow(plot_array)
    plt.xticks([])
    plt.ylabel("samples")
    plt.xlabel("probability of 3 classes")
    plt.colorbar()
    plt.show()


# ADD CODEfrom sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the SVM classifier with linear kernel, gamma=0.5, and probability=True
svm_classifier = SVC(kernel="linear", gamma=0.5, probability=True)
svm_classifier.fit(X_train, y_train)

# Obtain predictions on the test set
predictions = svm_classifier.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# ADD CODE
