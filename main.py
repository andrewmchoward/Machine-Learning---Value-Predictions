## Import packages
import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt

## Import data
paid_data = pd.ExcelFile('paid_data.xlsx').parse('Sheet0')
paid_data.head(6)

## Define input and output variables
x_train = np.array(paid_data['Number of Employees'])
y_train = np.array(paid_data['Claim Value Received'])
y_train = y_train / 10000
print(f"x consists of the number of employees in the business: {x_train[0:6]}")
print(f"y consists of the claim value the business received in $10,000's of dollars: {y_train[0:6]}")

## Define cost function
def compute_cost(x, y, w, b):
    
    m = x.shape[0]
    total_cost = 0
    cost = 0
    
    for i in range(m):
        f_wb = x[i] * w + b
        cost = cost + (f_wb - y[i])**2
        
    total_cost = cost * (1/(2 * m))
    return total_cost

## Test cost function with w = 0 and b = 0
test_cost = compute_cost(x_train, y_train, 0, 0)
print(test_cost)

## Define gradient function
def compute_gradient(x, y, w, b):
    
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = x[i] * w + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

## Test gradient function
test_gradient = compute_gradient(x_train, y_train, 0, 0)
print(test_gradient)

## Define gradient descent functions
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):

    m = len(x)
    
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range (num_iters):
        
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i<100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
        
        if i% math.ceil(num_iters/10) == 0 or i == num_iters - 1:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
            
    return w, b, J_history, w_history

## Define starting parameters
initial_w = 0
initial_b = 0

iterations = 12000
alpha = 0.0008

## Run gradient descent
w, b, J_hist, w_hist = gradient_descent(x_train, y_train, initial_w, initial_b, alpha, iterations, compute_cost, compute_gradient)

print(f"Resulting w and b: {w},{b}")

## Create line of best fit
m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

## Create scatter plot with linear fit
plt.plot(x_train, predicted, c = "b")

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Claim Value Received vs Number of Employees")
plt.ylabel("Claim Value in $10,000's")
plt.xlabel("Number of Employees")

## Test predictions
prediction1 = 20 * w + b
print(f"The predicted claim value for a company with 20 employees is ${(prediction1 * 10000):.2f}")

prediction2 = 10 * w + b
print(f"The predicted claim value for a company with 10 employees is ${(prediction2 * 10000):.2f}")