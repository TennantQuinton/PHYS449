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
        nn.Linear(50, 2, bias=False)
    )
    model = model.float()

    x_grid, y_grid = np.meshgrid(np.arange(lb, ub+(abs(lb-ub)/10), (abs(lb-ub)/10)), np.arange(lb, ub+(abs(lb-ub)/10), (abs(lb-ub)/10)))
    u = -y_grid/np.sqrt(x_grid**2 + y_grid**2)
    v = x_grid/np.sqrt(x_grid**2 + y_grid**2)
    u_lam = lambda x, y: -(y)/np.sqrt((x)**2 + (y)**2) #-(y.detach().numpy())/np.sqrt((x.detach().numpy())**2 + (y.detach().numpy())**2)
    v_lam = lambda x, y: (x)/np.sqrt((x)**2 + (y)**2) #(x.detach().numpy())/np.sqrt((x.detach().numpy())**2 + (y.detach().numpy())**2)

    rand_x, rand_y = rd.randrange(lb*1000, ub*1000)/1000, rd.randrange(lb*1000, ub*1000)/1000

    X = T.tensor(np.linspace(lb, ub, 10)).reshape((-1, 1))
    
    trial_x = lambda t: rand_x + t * model(t.float())
    trial_y = lambda t: rand_y + t * model(t.float())

    for i in np.arange(100):
        X.requires_grad = True

        outputs_x = trial_x(X)
        outputs_y = trial_y(X)
        x_t = T.autograd.grad(outputs_x, X, grad_outputs=T.ones_like(outputs_x), create_graph=True)[0]
        y_t = T.autograd.grad(outputs_y, X, grad_outputs=T.ones_like(outputs_y), create_graph=True)[0]

        X_up = X.detach().numpy()
        outputs_x_up = trial_x(X).detach().numpy()
        outputs_y_up = trial_y(X).detach().numpy()
        
        loss_x = T.mean((x_t - T.tensor(u_lam(X_up, outputs_x_up)))**2)
        loss_y = T.mean((y_t - T.tensor(u_lam(X_up, outputs_y_up)))**2)

        optimizer_x = optim.Adam(model.parameters())
        optimizer_y = optim.Adam(model.parameters())
        
        loss_x.backward()
        loss_y.backward()

        loss_tot_x = 0
        loss_tot_y = 0

        optimizer_x.zero_grad()
        optimizer_y.zero_grad()

        optimizer_x.step()
        optimizer_y.step()

    tt = np.linspace(lb, ub, 100)[:, None]
    with T.no_grad():
        xx = trial_x(T.Tensor(tt)).numpy()
        yy = trial_y(T.Tensor(tt)).numpy()
    plt.plot(xx, yy)
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

    ode_solv(lower, upper, n_tests)
