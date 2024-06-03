# ===================================================================================
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn import preprocessing
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# df = pd.read_csv("train.csv")
# X = df[["gender", "carowner", "travel", "il", "transportation"]].values

# d = {"male": 1, "female": 0}
# df["gender"] = df["gender"].map(d)
# d = {"cheap": 0, "standard": 1, "exp": 2}
# df["travel"] = df["travel"].map(d)
# print(df["il"].unique())

# # il = preprocessing.LabelEncoder()
# # il.fit(df["il"].unique())
# # df["il"] = il.transform(df["il"])

# d = {"low": 0, "med": 1, "high": 2}
# df["il"] = df["il"].map(d)

# X = df[["gender", "carowner", "travel", "il"]]

# d = {"bus": 0, "train": 1, "car": 2}
# df["transportation"] = df["transportation"].map(d)
# y = df["transportation"]


# xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=3)
# drugTree = DecisionTreeClassifier(criterion="gini", max_depth=6)
# drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
# drugTree.fit(xtrain, ytrain)
# # it shows the default parameters
# predTree = drugTree.predict(xtest)
# tree.plot_tree(drugTree)
# plt.show()
# print("DecisionTrees's Accuracy: ", accuracy_score(ytest, predTree))

# ============================================================================================

# from sklearn import metrics
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier

# df = pd.read_csv("teleCust1000t.csv")
# df.hist(
#     column="income", bins=100
# )  # m=df['income'] m.plot() for individual values visualization
# # print(df['custcat'].value_counts())
# # plt.show()
# # df.columns
# X = df[
#     [
#         "region",
#         "tenure",
#         "age",
#         "marital",
#         "address",
#         "income",
#         "ed",
#         "employ",
#         "retire",
#         "gender",
#         "reside",
#     ]
# ].values
# # print(X[0:5])
# y = df["custcat"].values
# # print(y[0:5])
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# print("Train set:", X_train.shape, y_train.shape)
# print("Test set:", X_test.shape, y_test.shape)
# scale = StandardScaler()
# X_train_norm = scale.fit(X_train).transform(X_train.astype(float))
# print(X_train_norm[0:5])
# k = 4
# # Train Model and Predict
# neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train_norm, y_train)
# print(neigh)
# X_test_norm = scale.fit(X_test).transform(X_test.astype(float))
# print(X_test_norm[0:5])
# yhat = neigh.predict(X_test_norm)
# print(yhat[0:5])
# print(
#     "Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train_norm))
# )
# print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# print(df['income'].value_counts())
# print(df['income'].mean())
# print(df['income'].mode())
# print(df['income'].median())
# print(df['income'].max())
# print(df['income'].min())
# count = df[(df['income'] >= 26) & (df['income'] <= 42)].shape[0]
# print(count)

# ===================================================================================================================


# import pandas as pd
# from sklearn import preprocessing
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split

# data = pd.read_csv("real_estate_data.csv")
# print(data.head())
# data.dropna(inplace=True)
# print(data.isna().sum())  # this will print empty / wrong values column.
# X = data.drop(columns=["MEDV"])
# print(pd.DataFrame(X))
# Y = data["MEDV"]
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.3, random_state=42
# )
# regression_tree = DecisionTreeRegressor(criterion="squared_error")
# x=regression_tree.fit(X_train, Y_train)
# y=x.score(X_test, Y_test)
# print("this is the value of y", y )
# prediction = regression_tree.predict(X_test)
# print(prediction)
# print("$", (prediction - Y_test).abs().mean() * 1000)


# =========================================================================================================


# import numpy
# from sklearn import metrics
# import matplotlib.pyplot as plt
# actual = numpy.random.binomial(1, 0.9, size=1000)
# predicted = numpy.random.binomial(1, 0.9, size=1000)
# con = metrics.confusion_matrix(actual, predicted)
# cm_display = metrics.ConfusionMatrixDisplay(
#     confusion_matrix=con, display_labels=[False, True]
# )
# cm_display.plot()
# plt.show()

# =============================================================================================================

# import numpy
# from sklearn import linear_model

# X = numpy.array(
#     [3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]
# ).reshape(-1, 1)
# y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
# logr = linear_model.LogisticRegression()
# logr.fit(X, y)
# predicted = logr.predict(numpy.array([5.46]).reshape(-1, 1))
# print(f"Predicted: {predicted[0]} Expected: 1")

# =====================================#
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# import numpy as np
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# df = pd.read_csv("logistic.csv")
# df = df[
#     [
#         "tenure",
#         "age",
#         "address",
#         "income",
#         "ed",
#         "employ",
#         "equip",
#         "callcard",
#         "wireless",
#         "churn",
#     ]
# ]
# df["churn"] = df["churn"].astype("int")
# print(df.head())
# print(df.shape)
# X = np.asarray(df[["tenure", "age", "address", "income", "ed", "employ", "equip"]])
# y = np.asarray(df["churn"])
# scale = preprocessing.StandardScaler()
# X = scale.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )
# print("Train set:", X_train.shape, y_train.shape)
# print("Test set:", X_test.shape, y_test.shape)
# LR = linear_model.LogisticRegression(C=0.01, solver="liblinear").fit(X_train, y_train)
# Regularization helps prevent overfitting by penalizing large coefficients.
# THE SOLVER PARAMETER IS USED IN OPTIMIZATION , LIBLINER IS SUITABLE FOR SMALL DATASETS.
# yhat = LR.predict(X_test)
# yhat_prob = LR.predict_proba(X_test)

# from sklearn import datasets
# from sklearn.linear_model import LogisticRegression

# iris = datasets.load_iris()

# X = iris["data"]
# y = iris["target"]

# logit = LogisticRegression(max_iter=10000)

# C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,2.25,2.50]

# scores = []

# for choice in C:
#     logit.set_params(C=choice)
#     logit.fit(X, y)
#     scores.append(logit.score(X, y))

# print(scores)

# ================================================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score, confusion_matrix
from sklearn import svm

# Read the CSV file
cell_df = pd.read_csv("cell_samples.csv")

# Plotting the first 50 malignant and benign samples
ax = cell_df[cell_df["Class"] == 4][0:50].plot(
    kind="scatter", x="Clump", y="UnifSize", color="DarkBlue", label="malignant"
)
cell_df[cell_df["Class"] == 2][0:50].plot(
    kind="scatter", x="Clump", y="UnifSize", color="Yellow", label="benign", ax=ax
)
plt.show()

# Select relevant columns
x = cell_df[
    [
        "ID",
        "Clump",
        "UnifSize",
        "UnifShape",
        "MargAdh",
        "SingEpiSize",
        "BareNuc",
        "BlandChrom",
        "NormNucl",
        "Mit",
        "Class",
    ]
]

# Convert "BareNuc" to numeric and drop rows with NaN values
cell_df["BareNuc"] = pd.to_numeric(cell_df["BareNuc"], errors="coerce")
cell_df = cell_df.dropna(subset=["BareNuc"])

# Convert "BareNuc" to integer
cell_df["BareNuc"] = cell_df["BareNuc"].astype("int")
print(cell_df.dtypes)

# Select features
feature_df = cell_df[
    [
        "Clump",
        "UnifSize",
        "UnifShape",
        "MargAdh",
        "SingEpiSize",
        "BareNuc",
        "BlandChrom",
        "NormNucl",
        "Mit",
    ]
]

# Convert features and target variable to NumPy arrays
X = np.asarray(feature_df)
y = np.asarray(cell_df["Class"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# Create and fit SVM classifier with RBF kernel
clf = svm.SVC(kernel="rbf")
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)

# Print evaluation metrics for the RBF kernel classifier
print("RBF Kernel:")
print("F1 Score:", f1_score(y_test, yhat, average="weighted"))
print("Jaccard Score:", jaccard_score(y_test, yhat, pos_label=2))

# Display confusion matrix for the RBF kernel classifier
cm_rbf = confusion_matrix(y_test, yhat)
print("Confusion Matrix (RBF Kernel):")
print(cm_rbf)

# Plot confusion matrix
plt.imshow(cm_rbf, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix (RBF Kernel)")
plt.colorbar()

classes = ["Benign", "Malignant"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# Create and fit SVM classifier with linear kernel
clf2 = svm.SVC(kernel="linear")
clf2.fit(X_train, y_train)
yhat2 = clf2.predict(X_test)

# Print evaluation metrics for the linear kernel classifier
print("\nLinear Kernel:")
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average="weighted"))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2, pos_label=2))

# ===============================================================================================

# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import pandas as pd
# from sklearn.datasets import load_digits
# import pandas as pd

# # Load the digits dataset
# digits = load_digits()

# # Convert to a Pandas DataFrame
# digits_df = pd.DataFrame(
#     data=digits.data, columns=[f"pixel_{i}" for i in range(digits.data.shape[1])]
# )
# digits_df["target"] = digits.target

# # Save the DataFrame to a CSV file
# digits_df.to_csv("mycsv.csv", index=False)


# X, y = digits.data, digits.target

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train a logistic regression model with OvR strategy
# logistic_reg_ovr = LogisticRegression(multi_class="ovr", solver="lbfgs")
# logistic_reg_ovr.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred_ovr = logistic_reg_ovr.predict(X_test)

# # Evaluate the accuracy
# accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
# print(f"Accuracy using OvR: {accuracy_ovr}")

# from sklearn.s import SVC
# svm_classifier = SVC(decision_function_shape="ovo")
# svm_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred_ovo = svm_classifier.predict(X_test)

# # Evaluate the accuracy
# accuracy_ovo = accuracy_score(y_test, y_pred_ovo)
# print(f"Accuracy using OvO: {accuracy_ovo}")

# classifiers = []
# for class_label in range(10):  # Assuming 10 classes
#     # Create a binary classifier for the current class
#     classifier = LogisticRegression()

#     # Assign binary labels: class_label vs. rest (including dummy class)
#     y_train_binary = np.where(y_train == class_label, 1, 0)

#     # Train the binary classifier
#     classifier.fit(X_train, y_train_binary)

#     classifiers.append(classifier)

# # Make predictions using all binary classifiers
# predictions = np.array([classifier.predict(X_test) for classifier in classifiers])

# # Use majority voting to determine the final predicted class
# final_predictions = np.argmax(np.sum(predictions, axis=0), axis=0)

# ============================================================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from plot import decision_boundary, plot_probability_array

# iris = datasets.load_iris()
# X = iris.data[:, [1, 3]]  # we only take the first two features .
# Y = iris.target
# # print(np.unique(Y))
# # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.RdYlBu)
# plt.xlabel(" sepal width (cm) ")
# plt.ylabel(" petal width ")
# # plt.show()
# # xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)
# lr = LogisticRegression(random_state=0)
# lr = lr.fit(X, Y)
# probability = lr.predict_proba(X)
# plot_probability_array(X, probability)
# # print(probability)
# print(probability[0, :])
# print(probability[0, :].sum())
# print(np.argmax(probability[0, :]))
# softmax_prediction = np.argmax(probability, axis=1)
# print(softmax_prediction)
# accuracy = accuracy_score(Y, softmax_prediction)
# print("Accuracy of logistic regression model is ", accuracy)

# from sklearn import svm

# support_classifier = svm(kernal="linear", gamma=0.5, probability=True)
# support_classifier.fit(X, Y)
# # classes_=set(np.unique(y)) find unique classes and make a set of it ,
# # K=len(classes_) find the no of classes , formula K * (K - 1) / 2

# ===================================================================================================
# #  One hot encoding
# import pandas
# from sklearn import linear_model
# cars = pandas.read_csv("DATA.csv")
# ohe_cars = pandas.get_dummies(cars[["Car"]])
# print(ohe_cars)
# num=ohe_cars.shape[1]
# print(num)
# X = pandas.concat([cars[["Volume", "Weight"]], ohe_cars], axis=1)
# print(X)
# y = cars["CO2"]
# regr = linear_model.LinearRegression()
# regr.fit(X, y)
# ##predict the CO2 emission of a Volvo where the weight is 2300kg, and the volume is 1300cm3:
# predictedCO2 = regr.predict(
#     [[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
# )
# print(predictedCO2)
# predictedCO2 = regr.predict(
#     [[2000, 1700, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
# )
# print(predictedCO2)

# ===========================================================================================

# from sklearn.cluster import KMeans
# import numpy as np
# import matplotlib.pyplot as plt

# x = np.array([4, 5, 10, 4, 3, 11, 14, 6, 10, 12])
# y = np.array([21, 19, 24, 17, 16, 25, 24, 22, 21, 21])
# plt.scatter(x, y)
# plt.show()
# data = list(zip(x, y))
# inertias = []

# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(data)
#     inertias.append(kmeans.inertia_)
# # The inertia_ attribute of the KMeans model represents the sum
# # of squared distances of samples to their closest cluster center.
# # This value is calculated for the current number of clusters (i)
# # and is added to the inertias list.
# print(inertias)
# plt.plot(range(1, 11), inertias, marker="o")
# plt.title("Elbow method")
# plt.xlabel("Number of clusters")
# plt.ylabel("Inertia")
# plt.show()
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(data)
# plt.scatter(x, y, c=kmeans.labels_)
# plt.show()

# ===========================================================================================================
""" # k means coursera
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs

# np.random.seed(0)
# X, y = make_blobs(
#     n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9
# )
# # plt.scatter(X[:, 0], X[:, 1], marker=".")
# # plt.show()
# k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
# k_means.fit(X)
# print(f"Predict labels: {k_means.predict(X)}")
# print(f"Labels: {y}")
# k_means_labels = k_means.labels_
# k_means_cluster_centers = k_means.cluster_centers_
# print(k_means_labels)
# print(k_means_cluster_centers)

# # Plot each point with color based on its cluster assignment
# # for i in range(len(X)):
# #     plt.scatter(X[i, 0], X[i, 1], marker=".", c=f"C{k_means_labels[i]}")

# # # Plot cluster centers with different colors
# # plt.scatter(
# #     k_means_cluster_centers[:, 0],
# #     k_means_cluster_centers[:, 1],
# #     marker="o",
# #     s=200,
# #     edgecolors="k",
# #     c="red",
# # )
# # plt.show()


# # # Initialize the plot with the specified dimensions.
# fig = plt.figure(figsize=(6, 4))
# # Colors uses a color map, which will produce an array of colors based on
# # the number of labels there are. We use set(k_means_labels) to get the
# # unique labels.
# colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# # Create a plot
# ax = fig.add_subplot(1, 1, 1)

# # For loop that plots the data points and centroids.
# # k will range from 0-3, which will match the possible clusters that each
# # data point is in.
# for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
#     # Create a list of all data points, where the data points that are
#     # in the cluster (ex. cluster 0) are labeled as true, else they are
#     # labeled as false.
#     my_members = k_means_labels == k

#     # Define the centroid, or cluster center.
#     cluster_center = k_means_cluster_centers[k]

#     # Plots the datapoints with color col.
#     ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")

#     # Plots the centroids with specified color, but with a darker outline
#     ax.plot(
#         cluster_center[0],
#         cluster_center[1],
#         "o",
#         markerfacecolor=col,
#         markeredgecolor="k",
#         markersize=6,
#     )

# # Title of the plot
# ax.set_title("KMeans")

# # Remove x-axis ticks
# # ax.set_xticks(())

# # # Remove y-axis ticks
# # ax.set_yticks(())

# # Show the plot
# plt.show()
 """

# ==========================================================================================================

"""import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

gm = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
gn = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(gm, gn))

# data={"column1":x,"column2":y}
# df = pd.DataFrame(data)
# df.plot(kind="scatter", x="column1", y="column2", color="blue", marker="o")

# Hierarchical clustering using scipy
linkage_data = linkage(data, method="ward", metric="euclidean")
dendrogram(linkage_data)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# Hierarchical clustering using sklearn
hierarchical_clusters = AgglomerativeClustering(
    n_clusters=2, metric="euclidean", linkage="ward"
)
labels = hierarchical_clusters.fit_predict(data)
plt.scatter(np.array(gm), np.array(gn), c=labels, cmap="viridis")
plt.title("Hierarchical Clustering Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Annotate each point with its (x, y) values
for i, txt in enumerate(data):
    plt.annotate(
        txt, (gm[i], gn[i]), textcoords="offset points", xytext=(0, 8), ha="center" ,va="center"
    )
#in enumerate i is index and txt is the value that is a tuple
#here txt represents the text to be displayed here it is a tuple
plt.show()
"""
# points will be plotted in scatter manner, model.fit_predict(data), label will be used in c in plt.
# scipy herarichal clustering
""" 

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic data with three clusters
np.random.seed(48)
data1 = np.random.randn(50, 2) + np.array([3, 3])
#this adds each random generated value to 3,3 value causing a random deviation
data2 = np.random.randn(50, 2) + np.array([-3, 3])
data3 = np.random.randn(50, 2) + np.array([0, -3])
data = np.concatenate((data1, data2, data3))
#this adds each 50 random generated values so the total values in data is 100
print(data)
# Hierarchical clustering using scipy
linkage_data = linkage(data, method="ward", metric="euclidean")

# Dendrogram
plt.figure(figsize=(12, 6))
dendrogram(
    linkage_data,
    leaf_rotation=90,
    leaf_font_size=8,
    labels=None,
    above_threshold_color="r",
)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Scatter plot with cluster labels
from scipy.cluster.hierarchy import fcluster

# Determine clusters using a distance threshold
threshold = 10
clusters = fcluster(linkage_data, threshold, criterion="distance")

# Scatter plot with different colors for each cluster
plt.scatter(data[:, 0], data[:, 1], c=clusters)
plt.title("Hierarchical Clustering Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show() 
"""

#
""" 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, to_tree, cut_tree
from sklearn.cluster import AgglomerativeClustering

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=1.0, random_state=42)

# Perform divisive hierarchical clustering
linkage_data = to_tree(
    AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(X).linkage
)
dendrogram(linkage_data)
plt.title("Divisive Hierarchical Clustering Dendrogram")
plt.show()

# Cut the dendrogram to obtain clusters
distance_threshold = 12  # Adjust this threshold based on dendrogram visualization
clusters = cut_tree(linkage_data, height=distance_threshold).flatten()

# Scatter plot with cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="viridis")
plt.title("Divisive Hierarchical Clustering Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show() 
"""

# bootstrap aggregation.

""" 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt

# wine_data = datasets.load_wine()
# columns = [f"data{i}" for i in range(wine_data.data.shape[1])]
# wine_df = pd.DataFrame(data=wine_data.data, columns=columns)
# wine_df["class"] = wine_data.target

# print(wine_df.head())
# wine_df.to_csv("mycaa.csv", index=False)


datam = datasets.load_wine(as_frame=True)  
# as_frame is used to import the columns names as well
# print(datam)
print(datam.data[:5])  # Print the first 5 rows

X = datam.data
y = datam.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=22
)
dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

print(
    "Train data accuracy:",
    accuracy_score(y_true=y_train, y_pred=dtree.predict(X_train)),
)
print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred))

estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]
models = []
scores = []

clf2 = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)
clf2.fit(X_train, y_train)
print(clf2.oob_score_)


for n_estimators in estimator_range:
    # Create bagging classifier
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)

    # Fit the model
    clf.fit(X_train, y_train)

    # Append the model and score to their respective list
    models.append(clf)
    scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))


# Generate the plot of scores against number of estimators
plt.figure(figsize=(9, 6))
plt.plot(estimator_range, scores)

# Adjust labels and font (to make visable)
plt.xlabel("n_estimators", fontsize=18)
plt.ylabel("score", fontsize=18)
plt.tick_params(labelsize=16)
plt.show()


plt.figure(figsize=(10,20 ))

plot_tree(clf2.estimators_[0], feature_names=X.columns)
plt.show()
 """

##########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##################
"""
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, LeaveOneOut

X, y = datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=42)

loo = LeaveOneOut()
k_folds = KFold(n_splits=5)
sk_folds = StratifiedKFold(n_splits=5)

scores = cross_val_score(clf, X, y, cv=sk_folds)  # cv=k_folds,cv=loo

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
"""
