'''
PHYS 449 -- Fall 2021
Assignment 4
    Name: Quinton Tennant
    ID: 20717788
'''

# Imports
import os, argparse, numpy as np, matplotlib.pyplot as plt

# Function for taking the +- in file and converting to usable integers (+-1)
def data_array_create(data):
    # Initialize spin list
    spin_list = []
    # Iterating over the lines of the data file
    for line in data:
        # Iterating over the characters in each line
        for i in line:
            # Append each converted value to the list
            spin_list.append(int(('{0}1'.format(i))))
    # Convert to array
    spin_array = np.array(spin_list)
    # Reshape array to be individual rows of N length
    spin_array = spin_array.reshape((-1, len(data[0])))
    
    # Output the array of spins
    return spin_array

# Function for finding the energy given a set J_list and spins
def energy(Js, spin_row):
    # Initialize the energy sum
    E_sum = 0
    # Iterate over the spins in each row
    for j, l in enumerate(spin_row):
        # For each iteration add to the Energy sum
        E_sum += -(Js[j] * spin_row[j] * spin_row[(j+1)%len(spin_row)])
        
    # Output the final sum
    return E_sum

# Function to create a random state of N length
def rand_state(N):
    rand = np.random.choice([-1, 1], N).tolist()
    
    # Output
    return rand

# Function to find the output of the Monte-Carlo Markov Chain using Metropolis-Hastings sampling
def MCMC(J_list, state_x, state_y):
    # calculate the energy with the current J and each fed in state
    E_y = energy(J_list, state_y)
    E_x = energy(J_list, state_x)
    
    # Calculating the Metropolis-Hastings normalized (0-1) probability
    prob = (np.exp(-E_x + E_y))/(3000)
    
    # Conditional tree for comparing the energies
    if (E_y <= E_x):
        return state_y, 1
    elif (E_x < E_y):
        # Take a random number
        isisnot = np.random.random(1)
        # Check where that random number lies on our probability and adjust accordingly
        if (isisnot < prob):
            return state_y, 1-prob
        else:
            return state_x, prob
    
# Function to take an array of spins and find the sum of the ij nearest neighbours in each line
def nearest_n_sum(array):
    # Initialize sum list for sum of each column
    sum_list = []
    # Iterating over the columns of the array
    for N in range(len(array[0])):
        # Initalize the column list of ij values
        col_list = []
        # Iterating over the spins in each row of the array
        for i, spins in enumerate(array):
            # Append these values to the col_list
            col_list.append(spins[N] * spins[(N+1) % len(spins)])
            
        # Append to the sum list the sum of each column of ij nearest neighbours
        sum = np.sum(np.array(col_list))
        sum_list.append(sum)
        
    # Output
    return sum_list
    
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 4: Tennant, Quinton (20717788)')
    parser.add_argument('-data_path', default='data/in.txt', help='relative file path for data input (default data/in.txt)')
    parser.add_argument('-output_path', default='outputs/', help='relative file path for output plots and weights (default output/)')
    parser.add_argument('-verb', default=2, help='verbosity of the code (from 0-2)')
    parser.add_argument('-n_epochs', default=5, help='number of epochs to update Jij over (default 5)')
    parser.add_argument('-n_plambda', default=1000, help='number of iterations running the MCMC to find a suitable p_lambda (default 1000)')

    # Receiving the command line arguments
    args = parser.parse_args()

    # Turning the arguments into variables
    data_path = args.data_path
    output_path = args.output_path
    verbosity = args.verb
    n_epochs = args.n_epochs
    n_plambda = args.n_plambda

    # Getting the absolute file path from the relative
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    
    # output path
    out_path = "{0}/{1}".format(my_absolute_dirpath, output_path)
    if (verbosity > 1):
        print('Making output directory: {0}'.format(output_path))
        
    if (os.path.isdir(out_path) == False):
        os.makedirs(out_path)
    
    # Checking if the in file exists
    if os.path.isfile("{0}/{1}".format(my_absolute_dirpath, data_path)):
        # Reading in the data and running the linear regression
        data_in = np.loadtxt("{0}/{1}".format(my_absolute_dirpath, data_path), dtype=str)
    else:
        # If the filepath does not exist then throw error
        print('Filepath {0}/{1} does not exist'.format(my_absolute_dirpath, data_path))
    
    # Arrange the data
    data = data_array_create(data_in)
    
    # Initialize the guesses for J_ij
    J_list = rand_state(len(data[0]))
    
    if (verbosity >= 0):
        # Update the user
        print('Random initalization of J: {0}'.format(J_list))
    
    # Run over the update of Jij N times
    N = n_epochs
    d_loss_list = []
    t_list = []
    loss_list = []
    
    for t in range(N):
        if (verbosity > 0):
            # Update user
            print("{0}/{1} of updating J_ij".format(t+1, N))
        
        # Initialize sum of the energy sums (NOT USED)
        E_sum_total = 0
        for i in data:
            E_sum_total += energy(J_list, i)
        
        # Initialize the neg_phase list and the p_lambda list
        neg_list = []
        plambda_list = []
        
        # Iterate over the dataset
        for iter in range(len(data)):
            if (verbosity > 1):
                # Print update every 200 steps in finding p_lambdas
                if ((iter % (len(data)/5)) == 0):
                    print("{0}/{1} completed p_lambda states".format(iter, len(data)))
                # Also print update at the last step
                elif (iter == (len(data) - 1)):
                    print("{0}/{1} completed p_lambda states".format(iter+1, len(data)))
                    
            elif (verbosity == 1):
                # Print update every 200 steps in finding p_lambdas
                if ((iter % (len(data)/2)) == 0):
                    print("{0}/{1} completed p_lambda states".format(iter, len(data)))
                # Also print update at the last step
                elif (iter == (len(data) - 1)):
                    print("{0}/{1} completed p_lambda states".format(iter+1, len(data)))
                
            # Get a random x and y state
            rand_state_x = rand_state(len(data[0]))
            rand_state_y = rand_state(len(data[0]))
            
            # Use the MCMC for that first state
            MCMC_init = MCMC(J_list, rand_state_x, rand_state_y)[0]
            # Iterate 100 times to get best p_lambda value
            for k in range(n_plambda):
                # Get another random state to compare to our initial
                rand_state_new = rand_state(len(data[0]))
                
                # Get the p_lambda and state output in each loop
                pl_res = (MCMC(J_list, MCMC_init, rand_state_new))[1]
                MCMC_res = (MCMC(J_list, MCMC_init, rand_state_new))[0]
                
                # Update the initial state with the best state for this step
                MCMC_init = MCMC_res
                
                # Append the p_lambda to the list
                plambda_list.append(pl_res)
            # Append the neg_phase to our list
            neg_list.append(MCMC_res)
                
        # Convert both to arrays
        neg_array = np.array(neg_list)
        pl_array = np.array(plambda_list)
        
        # Initialize the negative phase
        neg_phase_list = []
        # Length of the p_lambda data
        D_neg = len(neg_array)
        
        # Finding the negative phase
        sum = np.array(nearest_n_sum(neg_array))
        avg = [sum_val/D_neg for sum_val in sum]
        neg_phase_list.append(avg)
            
        # Initialize the positive phase
        pos_phase_list = []
        # Length of the dataset
        D_pos = len(data)
        
        # Finding the positive phase
        sum = np.array(nearest_n_sum(data))
        avg = avg = [sum_val/D_pos for sum_val in sum]
        pos_phase_list.append(avg)
            
        # Convert to arrays
        neg_phase_array = np.array(neg_phase_list)
        pos_phase_array = np.array(pos_phase_list)
        
        # Compute the update rule and loss
        d_loss = pos_phase_array - neg_phase_array
        d_loss_list.append(d_loss)
        t_list.append(t)
        loss = -(1/(len(pl_array)))*(np.sum(np.log(pl_array)))
        loss_list.append(loss)
        
        # Update our Jij values by the update rule
        J_list = (np.array(J_list) + d_loss)[0]
        
        if (verbosity > 0):
            # Update the user
            print('Loss update rule: {0}'.format(d_loss))
            print('Updated coupler values: {0}'.format(J_list))
            print()
        
    if (verbosity > 0):
        # Now normalizing the edge-weights since we are restricted to +-1
        print('Normalizing coupler values')
    J_out = ([round(val/(abs(val))) for val in J_list])
    
    # Convert to dictionary as set in the assignment
    J_dict = {}
    for index, value in enumerate(J_out):
        # J_dict['({0}, {1})'.format(index, (index+1)%len(J_out))] = value
        J_dict[(index, (index+1)%len(J_out))] = value
        
    # Plotting if verbosity is > 0
    if (verbosity > 0):
        print('Plotting to {0}output_plot.jpg'.format(output_path))
        plt.figure(figsize=(10,10))
        plt.gca()
        for i in range(len(J_list)):
            plot_list = []
            for line in d_loss_list:
                plot_list.append(line[0][i])
            
            plt.plot(t_list, plot_list, label=f'$J_{{{i}, {(i+1)%(len(J_list))}}}$')
            
        plt.title(r'$\frac{d}{d\lambda}L$ over each iteration loop')
        plt.xlabel('iteration')
        plt.ylabel(r'$\frac{d}{d\lambda}L$')
        plt.legend(fontsize=12)
        plt.savefig('{0}/output_plot.jpg'.format(out_path))
        
        if (verbosity > 1):
            plt.show()
    
    # Final update
    print('Final coupler values found: {0}'.format(J_dict))
    print('Saving coupler dictionary to {0}couplers_out.txt'.format(output_path))
    with open('{0}/couplers_out.txt'.format(out_path), 'w') as out_txt:
        out_txt.write('{0}'.format(J_dict))