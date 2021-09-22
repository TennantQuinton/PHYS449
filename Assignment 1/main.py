import os
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# Write your assignment here

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

data_in = np.loadtxt("{0}/data/2.in".format(my_absolute_dirpath))
y_dat = np.array(data_in[:,-1]).T
x_dat = np.array(data_in[:,0:-1])

row_count = (len(data_in))
ones = [1] * row_count
x_dat = np.insert(x_dat, 0, ones, axis=1)

#print(x_dat)
#print(y_dat)
#print(data_in)

transpose = x_dat.T
#print(transpose)

multiply_1 = np.matmul(transpose, x_dat)
#print(multiply_1)

inverse = np.linalg.inv(np.matmul(transpose, x_dat))
#print(inverse)

multiply_2 = np.matmul(inverse, transpose)
#print(multiply_2)

multiply_3 = np.matmul(multiply_2, y_dat)
print(multiply_3)


