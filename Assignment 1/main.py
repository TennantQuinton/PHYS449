import os
import numpy as np
import matplotlib.pyplot as plt
# Write your assignment here

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

data_in = np.loadtxt("{0}/data/1.in".format(my_absolute_dirpath))

col_count = (len(list(data_in[0])))

count=0
while count <= (col_count-2):
    x_dat = list(data_in[:,count])
    y_dat = list(data_in[:,-1])

    x_mean = np.mean(x_dat)
    y_mean = np.mean(y_dat)

    plt.scatter(x_dat, y_dat)

    x_minus_mean_sqrd_list = []
    xy_minus_mean_list = []

    index = 0
    while index <= len(x_dat)-1:
        x_minus_mean = x_dat[index] - x_mean
        x_minus_mean_sqrd = (x_minus_mean)*(x_minus_mean)
        x_minus_mean_sqrd_list.append(x_minus_mean_sqrd)

        y_minus_mean = y_dat[index] - y_mean

        xy_minus_mean = (x_minus_mean)*(y_minus_mean)
        xy_minus_mean_list.append(xy_minus_mean)

        index+=1

    x_minus_sqrd_sum = sum(x_minus_mean_sqrd_list)
    xy_minus_sum = sum(xy_minus_mean_list)

    slope = (xy_minus_sum/x_minus_sqrd_sum)
    intercept = y_mean - (slope*x_mean)
    print(slope)
    print(intercept)

    line_x_dat = np.arange(min(x_dat), max(x_dat), 0.1)
    line_y_dat = (slope*line_x_dat) + intercept

    plt.plot(line_x_dat, line_y_dat)
    #plt.show()

    y_diff = 0

    index = 0
    while index <= len(x_dat)-1:
        x_val = x_dat[index]
        y_val = y_dat[index]

        y_diff = (y_val - ((slope*x_val) + intercept))**2
        y_diff+=y_diff

        index+=1
    print(y_diff)
    count+=1
plt.show()

