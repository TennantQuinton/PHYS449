'''
PHYS 449 -- Fall 2021
Assignment 3
    Name: Quinton Tennant
    ID: 20717788
'''

import os, json, argparse, numpy as np, matplotlib.pyplot as plt, pandas as pd
import torch as T, torch.nn as nn, torch.optim as optim
import random as rd

def ode_sol(lb, ub, ntests):
    for i in np.arange(0, ntests, 1):
        rand_x, rand_y = rd.randrange(lb*1000, ub*1000)/1000, rd.randrange(lb*1000, ub*1000)/1000
        print(rand_x, rand_y)

        x, y = np.meshgrid(np.arange(lb, ub+(abs(lb-ub)/10), (abs(lb-ub)/10)), np.arange(lb, ub+(abs(lb-ub)/10), (abs(lb-ub)/10)))
    
        u = -y/np.sqrt(x**2 + y**2) #lambda x, y: -y/np.sqrt(x**2 + y**2)
        v = x/np.sqrt(x**2 + y**2) #lambda x, y: x/np.sqrt(x**2 + y**2)

        plt.quiver(x, y, u, v)
        plt.scatter(rand_x, rand_y)
        plt.show()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 3: Tennant, Quinton (20717788)')
    parser.add_argument('--param', default='param/param.json', help='relative file path for json attributes')
    parser.add_argument('-v', default=1, help='verbosity (default = 1)', type=float)
    parser.add_argument('--res_path', default='plots/', help='relative path to save the test plots at')
    parser.add_argument('--x_field', default='x**2', help='expression of the x-component of the vector field')
    parser.add_argument('--y_field', default='y**2', help='expression of the y-component of the vector field')
    parser.add_argument('--lb', default=-1, help='lower bound for initial conditions', type=int)
    parser.add_argument('--ub', default=+1, help='upper bound for initial conditions', type=int)
    parser.add_argument('--n_tests', default=3, help='number of test trajectories to plot', type=int)

    # Receiving the command line arguments
    args = parser.parse_args()

    # Getting the absolute file path from the relative
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    lower = args.lb
    upper = args.ub
    n_tests = args.n_tests

    ode_sol(lower, upper, n_tests)
