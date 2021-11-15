'''
PHYS 449 -- Fall 2021
Assignment 4
    Name: Quinton Tennant
    ID: 20717788
'''

# Imports
import os, json, argparse, numpy as np, matplotlib.pyplot as plt


def data_array_create(data):
    spin_list = []
    for line in data:
        for i in line:
            spin_list.append(int(('{0}1'.format(i))))
    spin_array = np.array(spin_list)
    spin_array = spin_array.reshape((-1, 4))
    
    return spin_array

def energy(Js, spin_row):
    E_sum = 0
    for j, l in enumerate(spin_row):
        E_sum += -(Js[j] * spin_row[j] * spin_row[(j+1)%len(spin_row)])
    return E_sum

def rand_state(N):
    rand = np.random.choice([-1, 1], N).tolist()
    return rand

def MCMC(J_list, state_x, state_y):
    E_y = energy(J_list, state_y)
    E_x = energy(J_list, state_x)
    
    prob = (np.exp(-E_x + E_y))/(3000)
    if (E_y <= E_x):
        return state_y, 1
    elif (E_x < E_y):
        isisnot = np.random.random(1)
        if (isisnot < prob):
            return state_y, 1-prob
        else:
            return state_x, prob
    
def nearest_n_sum(state):
    sum = 0
    for i, spin in enumerate(state):
        # print("{0}*{1} = {2}".format(state[i], state[(i+1) % len(state)], state[i] * state[(i+1) % len(state)]))
        sum += state[i] * state[(i+1) % len(state)]
    return sum

def loss(d_loss, p_lambda, data):
    sum = 0
    D_data = len(data)
    
    
        

    
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 4: Tennant, Quinton (20717788)')
    parser.add_argument('-data_path', default='data/in.txt', help='relative file path for data input')

    # Receiving the command line arguments
    args = parser.parse_args()

    # Turning the arguments into variables
    data_path = args.data_path

    # Getting the absolute file path from the relative
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    
    # Checking if the in file exists
    if os.path.isfile("{0}/{1}".format(my_absolute_dirpath, data_path)):
        # Reading in the data and running the linear regression
        data_in = np.loadtxt("{0}/{1}".format(my_absolute_dirpath, data_path), dtype=str)
    else:
        print('Filepath {0}/{1} does not exist'.format(my_absolute_dirpath, data_path))
    
    # Arrange the data
    data = data_array_create(data_in)
    
    # Initialize the guesses for J_ij
    J_list = rand_state(4)
    print('Random initalization of J: {0}'.format(J_list))
    
    E_sum_total = 0
    for i in data:
        E_sum_total += energy(J_list, i)
    
    neg_list = []
    plambda_list = []
    
    for iter in range(len(data)):
        if ((iter % 200) == 0):
            print("{0}/{1} iteration of finding p_lambda states".format(iter, len(data)))
        rand_state_x = rand_state(4)
        rand_state_y = rand_state(4)
        
        MCMC_init = MCMC(J_list, rand_state_x, rand_state_y)[0]
        for k in range(1000):
            rand_state_new = rand_state(4)
            MCMC_res = (MCMC(J_list, MCMC_init, rand_state_new))[0]
            MCMC_init = MCMC_res
            
        neg_list.append(MCMC_res)
            
    neg_array = np.array(neg_list)
    
    neg_phase_list = []
    D_neg = len(neg_array)
    
    for state in neg_array:
        sum = nearest_n_sum(state)
        avg = sum/D_neg
        neg_phase_list.append(avg)
        
    pos_phase_list = []
    D_pos = len(data)
    
    for state in data:
        sum = nearest_n_sum(state)
        avg = sum/D_pos
        pos_phase_list.append(avg)
        
    neg_phase_array = np.array(neg_phase_list)
    pos_phase_array = np.array(pos_phase_list)
    
    d_loss = pos_phase_array - neg_phase_array
    print(np.log(p_lambda_array))
    loss = (np.sum(np.log(p_lambda_array)))
    print(loss)