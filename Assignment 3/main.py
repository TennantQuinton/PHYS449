'''
PHYS 449 -- Fall 2021
Assignment 3
    Name: Quinton Tennant
    ID: 20717788
'''

# Imports
import os, json, argparse, numpy as np, matplotlib.pyplot as plt
import torch as T, torch.nn as nn, torch.optim as optim
from torch.nn.modules.loss import MSELoss
import random as rd

def ode_solv(lb, ub, ntests, xfield, yfield, verb, results_out, param_in):
    # Getting the absolute file path from the relative
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    json_file = "{0}/{1}".format(my_absolute_dirpath, param_in)
    plot_file = "{0}/{1}".format(my_absolute_dirpath, results_out)

    # Checking if our json file exists
    if os.path.isfile(json_file):
        # Open the json file
        with open(json_file, 'r') as file:
            # Getting parameters from json
            paras = json.load(file)
            hidden_layer_size = paras['hidden layers']
            epoch_max = paras['epoch max']
            step_size = paras['solution step size']
    else:
        print('Filepath {0} does not exist'.format(json_file))

    # For plotting vector field
    x_grid, y_grid = T.tensor(np.meshgrid(np.arange(lb, ub+(abs(lb-ub)/25), (abs(lb-ub)/25)), np.arange(lb, ub+(abs(lb-ub)/25), (abs(lb-ub)/25))))

    # For finding the gradient at each point
    u_lam = lambda x, y: eval(xfield) #-(y)/np.sqrt((x)**2 + (y)**2) #np.sin(np.pi*x) + np.sin(np.pi*y) #-(y)/np.sqrt((x)**2 + (y)**2) #
    v_lam = lambda x, y: eval(yfield) #(x)/np.sqrt((x)**2 + (y)**2) #np.cos(np.pi*y) #(x)/np.sqrt((x)**2 + (y)**2) #

    # Establishing the network
    model = nn.Sequential(
        nn.Linear(2, hidden_layer_size), 
        nn.ReLU(),
        nn.Linear(hidden_layer_size, 2, bias=False)
    ).float()

    # Establishing the optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    nn_loss = MSELoss(reduction='sum')

    loss_total = 0
    # Status update
    print('Starting Training...')
    # Iterating over the number of training epochs
    for t in np.arange(epoch_max):
        # Iterating over the x positions in the grid
        for x_pos in x_grid[0]:
            # Iterating over the y positions in the grid
            for y_pos in y_grid:
                # Setting the position tensor to be fed into the network
                pos_tens = (T.tensor([x_pos, y_pos[0]])).float()

                # Setting what the true values of the grad are from the u and v functions
                true_x_grad = u_lam(x_pos, y_pos[0]).float()
                true_y_grad = v_lam(x_pos, y_pos[0]).float()

                # Running the model at that position
                probs = model(pos_tens)
                # Compute loss between the expected and true value
                loss = nn_loss(probs, T.tensor([true_x_grad, true_y_grad]))

                # Zero the gradients
                optimizer.zero_grad()

                # Backprop
                loss.backward()
                # Step
                optimizer.step()
                # Update
        if (verb>=1):
            print("Epoch:{0}/{1}, Loss:{2}".format(t+1, epoch_max, loss.item()))

    # Status update
    print('Plotting...')
    figure = plt.figure(figsize=(15,10))
    plt.gca()
    # Iterating over the number of starting points
    for i in np.arange(ntests):
        # Random starting position
        rand_x, rand_y = rd.randrange(lb*1000, ub*1000)/1000, rd.randrange(lb*1000, ub*1000)/1000

        # Small step size for iterating the gradient
        dt = step_size

        # Set the first positions
        x_pos = rand_x
        y_pos = rand_y
        # Initialize plot lists
        x_list = [x_pos]
        y_list = [y_pos]

        # Iterating over the small steps
        for t in np.arange(0, 10, dt):
            # Setting the position to be tested in the model
            pos = (T.tensor([x_pos, y_pos])).float()
            # Running the model
            with T.no_grad():
                probs = model(pos)

            # What does the model predict for dxdt and dydt
            pred_xgrad = probs[0].item()
            pred_ygrad = probs[1].item()

            # Update the position by the small step in the grad direction
            x_pos += dt * pred_xgrad
            y_pos += dt * pred_ygrad

            # Append to the plotting lists
            x_list.append(x_pos)
            y_list.append(y_pos)

        # COMMENTED OUT: Plotting the actual solution from u,v
        # x_pos = rand_x
        # y_pos = rand_y
        # x_list = [rand_x]
        # y_list = [rand_y]
        # for t in X:
        #     t = t.item()
        #     x_pos += t * u_lam(x_pos, y_pos)
        #     y_pos += t * v_lam(x_pos, y_pos)
        #     x_list.append(x_pos)
        #     y_list.append(y_pos)

        # Plotting the proposed solution
        plt.plot(x_list, y_list, zorder = 1, label='Starting from [{0},{1}]'.format(rand_x, rand_y), linewidth=3)

        # Plotting the starting points
        plt.scatter(rand_x, rand_y, color = 'red', zorder = 2, s=100)

    # Plotting the vector field
    plt.quiver(x_grid, y_grid, u_lam(x_grid, y_grid), v_lam(x_grid, y_grid), zorder = 0)

    # Sometimes the plots run off the page. This keeps the plot where we want it
    plt.xlim(lb, ub)
    plt.ylim(lb, ub)
    plt.title('Approximate Solution of u={0}, v={1}'.format(xfield, yfield))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=12)

    # Save plot
    print('Saving Plot to {0}plot_output.jpg'.format(results_out))
    plt.savefig('{0}/plot_output.jpg'.format(plot_file))


    


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 3: Tennant, Quinton (20717788)')
    parser.add_argument('--param', default='param/param.json', help='relative file path for json attributes (default = \'param/param.json\')')
    parser.add_argument('-v', default=1, help='verbosity (default = 1)', type=float)
    parser.add_argument('--res_path', default='plots/', help='relative path to save the test plots at (default = \'plots/\')')
    parser.add_argument('--x_field', default='y**2', help='expression of the x-component of the vector field (default = y**2)')
    parser.add_argument('--y_field', default='x**2', help='expression of the y-component of the vector field (default = x**2)')
    parser.add_argument('--lb', default=-1, help='lower bound for initial conditions (default = -1)', type=int)
    parser.add_argument('--ub', default=+1, help='upper bound for initial conditions (default = +1)', type=int)
    parser.add_argument('--n_tests', default=3, help='number of test trajectories to plot (default = 3)', type=int)

    # Receiving the command line arguments
    args = parser.parse_args()

    # Turning the arguments into variables
    lower = args.lb             # Lower bound
    upper = args.ub             # Upper bound
    n_tests = args.n_tests      # Number of tests

    x_field = args.x_field      # X vector field
    y_field = args.y_field      # Y vector field
    verbosity = args.v          # Verbosity

    results_out = args.res_path # Plotting location
    param_in = args.param       # Parameters location

    # Call the function
    ode_solv(lower, upper, n_tests, x_field, y_field, verbosity, results_out, param_in)
