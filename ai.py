import numpy as np

import matplotlib.pyplot as plt

def compute_model_output(x, w, b):

# Computes the prediction of a linear modelArgs:

# x (ndarray (m,)): Data, m examples.

# w, b (scalar): model parameters

# Returns:

# f_wb (ndarray (m,)): model prediction

    m = x.shape[0]

    f_wb = np.zeros(m)
    for i in range(m):
         f_wb[i] = w * x[i] + b 
    return f_wb

# Example data (Replace x_train and y_train with your actual data)

x_train = np.array([1, 2, 3, 4, 5]) # Example input data

y_train = np.array([2, 4, 6, 8, 10]) # Example target values

w = 2 # Example model weight

b=0# Example model bias

#Compute the model output

tmp_f_wb = compute_model_output(x_train, w, b)

# Plot our model prediction

plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

# Plot the data points

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# Set the title and Labels

plt.title("Housing Prices")

plt.ylabel('Price (in 1000s of dollars)')

plt.xlabel('Size (1000 sqft)')

plt.legend()

plt.show()