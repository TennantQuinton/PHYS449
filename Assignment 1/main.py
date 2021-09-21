import os
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# Write your assignment here

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

data_in = np.loadtxt("{0}/data/1.in".format(my_absolute_dirpath))
y_dat = np.array(data_in[:,-1]).T
print(y_dat)
print(data_in)

data_in_transpose = data_in.T
print(data_in_transpose)

multiply_1 = np.matmul(data_in_transpose, data_in)
print(multiply_1)

inverse = np.linalg.inv(np.matmul(data_in_transpose, data_in))
print(inverse)

multiply_2 = np.matmul(inverse, data_in_transpose)
print(multiply_2)

multiply_3 = np.matmul(multiply_2, y_dat)
print(multiply_3)


