import os, json, argparse, sys
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def linear_regression(input_data):
    # Slicing the data into the first M-1 columns and the final column
    y_dat = np.array(input_data[:,-1]).T
    x_dat = np.array(input_data[:,0:-1])

    # Making the first column of the design matrix 1s
    ones = [1] * (len(input_data))
    x_dat = np.insert(x_dat, 0, ones, axis=1)

    # Going step-by-step through finding w
    # First: transpose the design matrix
    transpose = x_dat.T
    # Second: multiply the transpose with the original
    multiply_1 = np.matmul(transpose, x_dat)
    # Third: take the inverse of that multiplied matrix result
    inverse = np.linalg.inv(multiply_1)
    # Fourth: multiply the inverted matrix and the transpose
    multiply_2 = np.matmul(inverse, transpose)
    # Fifth: multiply by the T matrix
    multiply_3 = np.matmul(multiply_2, y_dat)

    # Output the results (Need to output to .out file)
    return multiply_3

def grad_descent(input_data, learning_rate, num_iter):
    print("hello world")


n = "1"

# Getting the absolute file path from the relative
my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

# Reading in the data and running the linear regression (Need to change to getting from command line input)
data_in = np.loadtxt("{0}/data/{1}.in".format(my_absolute_dirpath, n))
print(linear_regression(data_in))


json_input_file = "{0}/data/{1}.json".format(my_absolute_dirpath, n)
with open(json_input_file, 'r') as file:
    inputs = json.load(file)

    learning_rate = inputs['learning rate']
    num_iter = inputs['num iter']

print(grad_descent(data_in, learning_rate, num_iter))
