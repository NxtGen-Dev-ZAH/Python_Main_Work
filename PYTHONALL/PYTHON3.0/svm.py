#without
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('diabetes.csv')
# split into train/test sets
X = data.drop(columns=['Outcome'])  # Using all features except the target column
y = data['Outcome'] 
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
model = SVC()
model = model.fit(trainX, trainy)
y_pred = model.predict(testX)
accuracy = accuracy_score(testy, y_pred) 
conf_matrix = confusion_matrix(testy, y_pred) 
print("OUTPUT:\n") 
print(f"Accuracy: {accuracy*100:.2f}%") 
print("Confusion Matrix: \n", conf_matrix)

print(classification_report(testy,y_pred,target_names=['Class 0','Class 1']))






# example of grid searching key hyperparametres for SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix 
import pandas as pd
from sklearn.svm import SVC
# define dataset
data = pd.read_csv('diabetes.csv')
# split into train/test sets

X = data.drop(columns=['Outcome'])  # Using all features except the target column
y = data['Outcome'] 
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# define model and parameters
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))