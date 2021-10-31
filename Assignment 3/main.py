'''
PHYS 449 -- Fall 2021
Assignment 3
    Name: Quinton Tennant
    ID: 20717788
'''

import os, json, argparse, numpy as np, matplotlib.pyplot as plt, pandas as pd
from numpy.core.fromnumeric import shape
import torch as T, torch.nn as nn, torch.optim as optim
import random as rd

def ode_solv(lb, ub, ntests):
    model = nn.Sequential(
        nn.Linear(1, 50), 
        nn.Sigmoid(),
        nn.Linear(50, 1, bias=False)
    )

    # For plotting vector field
    x_grid, y_grid = np.meshgrid(np.arange(lb, ub+(abs(lb-ub)/10), (abs(lb-ub)/10)), np.arange(lb, ub+(abs(lb-ub)/10), (abs(lb-ub)/10)))

    # For finding the gradient at each point
    u_lam = lambda x, y: -(y)/np.sqrt((x)**2 + (y)**2)
    v_lam = lambda x, y: (x)/np.sqrt((x)**2 + (y)**2)

    u_grid = u_lam(x_grid, y_grid)
    v_grid = v_lam(x_grid, y_grid)
    print(u_grid)

    # random starting position
    rand_x, rand_y = rd.randrange(lb*1000, ub*1000)/1000, rd.randrange(lb*1000, ub*1000)/1000

    # tensor of t-points
    X = T.tensor(np.linspace(lb, ub, 1000)).reshape((-1, 1))
    xx_list = []
    yy_list = []
    
    plt.scatter(rand_x, rand_y, color = 'red', zorder = 1)
    plt.quiver(x_grid, y_grid, u_lam(x_grid, y_grid), v_lam(x_grid, y_grid), zorder = 0)
    plt.show()


    


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 3: Tennant, Quinton (20717788)')
    parser.add_argument('--param', default='param/param.json', help='relative file path for json attributes (default = \'param/param.json\')')
    parser.add_argument('-v', default=1, help='verbosity (default = 1)', type=float)
    parser.add_argument('--res_path', default='plots/', help='relative path to save the test plots at (default = \'plots/\')')
    parser.add_argument('--x_field', default='x**2', help='expression of the x-component of the vector field (default = x**2)')
    parser.add_argument('--y_field', default='y**2', help='expression of the y-component of the vector field (default = y**2)')
    parser.add_argument('--lb', default=-10, help='lower bound for initial conditions (default = -1)', type=int)
    parser.add_argument('--ub', default=+10, help='upper bound for initial conditions (default = +1)', type=int)
    parser.add_argument('--n_tests', default=3, help='number of test trajectories to plot (default = 3)', type=int)

    # Receiving the command line arguments
    args = parser.parse_args()

    # Getting the absolute file path from the relative
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    lower = args.lb
    upper = args.ub
    n_tests = args.n_tests

    x_field = args.x_field
    y_field = args.y_field
    verbosity = args.v

    results_out = args.res_path
    param_in = args.param

    ode_solv(lower, upper, n_tests)
