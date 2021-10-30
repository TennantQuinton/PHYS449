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

def train_iter(x):
    print()

def slope_at(x_val, y_val):
    u_lam = lambda x, y: -y/np.sqrt(x**2 + y**2)
    v_lam = lambda x, y: x/np.sqrt(x**2 + y**2)

    u_point = u_lam(x_val, y_val)
    v_point = v_lam(x_val, y_val)

    return T.tensor((np.array[u_lam, v_lam]).reshape((-1,1)))


def ode_sol(lb, ub, ntests):
    model = nn.Sequential(
        nn.Linear(1, 50), 
        nn.Sigmoid(),
        nn.Linear(50, 2, bias=False)
    )

    optimizer = optim.Adam(model.parameters())
    nn_loss = nn.NLLLoss()

    for i in np.arange(0, ntests, 1):
        optimizer.zero_grad()
        x_grid, y_grid = np.meshgrid(np.arange(lb, ub+(abs(lb-ub)/10), (abs(lb-ub)/10)), np.arange(lb, ub+(abs(lb-ub)/10), (abs(lb-ub)/10)))
        u = -y_grid/np.sqrt(x_grid**2 + y_grid**2)
        v = x_grid/np.sqrt(x_grid**2 + y_grid**2)

        rand_x, rand_y = rd.randrange(lb*1000, ub*1000)/1000, rd.randrange(lb*1000, ub*1000)/1000

        X = (np.linspace(lb, ub, 100)).reshape((-1, 1))
        X = T.tensor(X)
        #u_point = slope_at(rand_x, rand_y)[0]
        #v_point = slope_at(rand_x, rand_y)[1]

        trial = lambda x: rand_x + x * model(x)

        X.requires_grad = True
        outputs = trial(X)
        trial_dt = T.autograd.grad(outputs, X, grad_outputs=T.ones_like(outputs), create_graph=True)[0]
        loss = T.mean((trial_dt - u(X, outputs))  ** 2)

        
        X = T.utils.data.TensorDataset(X, trial_dt)
        X = T.utils.data.DataLoader(X, batch_size = 10)
        loss = nn_loss()

        plt.scatter(rand_x, rand_y, color = 'red', zorder = 1)
    plt.quiver(x_grid, y_grid, u, v, zorder = 0)
    plt.show()


    


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 3: Tennant, Quinton (20717788)')
    parser.add_argument('--param', default='param/param.json', help='relative file path for json attributes (default = \'param/param.json\')')
    parser.add_argument('-v', default=1, help='verbosity (default = 1)', type=float)
    parser.add_argument('--res_path', default='plots/', help='relative path to save the test plots at (default = \'plots/\')')
    parser.add_argument('--x_field', default='x**2', help='expression of the x-component of the vector field (default = x**2)')
    parser.add_argument('--y_field', default='y**2', help='expression of the y-component of the vector field (default = y**2)')
    parser.add_argument('--lb', default=-1, help='lower bound for initial conditions (default = -1)', type=int)
    parser.add_argument('--ub', default=+1, help='upper bound for initial conditions (default = +1)', type=int)
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

    ode_sol(lower, upper, n_tests)
