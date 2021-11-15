'''
PHYS 449 -- Fall 2021
Assignment 4
    Name: Quinton Tennant
    ID: 20717788
'''

# Imports
import os, json, argparse, numpy as np, matplotlib.pyplot as plt
import torch as T, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.nn.modules.loss import MSELoss
from torch.autograd import Variable
import random


def data_array_create(data):
    spin_list = []
    for line in data:
        for i in line:
            spin_list.append(int(('{0}1'.format(i))))
    spin_array = np.array(spin_list)
    spin_array = spin_array.reshape((-1, 1, 4))
    spin_tensor = (T.tensor(spin_array))
    
    return spin_array

def energy(Js, spins):
    E_sum = 0
    for k, i in enumerate(spins):
        for j, l in enumerate(i[0]):
            print('here:')
            print(i[0])
            print(l)
            print(len(i[0]))
            print(j)
            E_sum += -(Js[k] * spins[k])# * spins[(k+1)%len(spins)])
    return E_sum

def rand_state(N):
    return np.random.choice([-1, 1], N)

def MCMC(J_list, state_y, state_x):
    E_y = energy(J_list, state_y)
    E_x = energy(J_list, state_x)
    if (E_y < E_x):
        return state_y
    elif (E_x < E_y):
        return state_x

    
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
    print('Associated cost of first coupling: {0}'.format(energy(J_list, data)))
    
    rand_state_x = rand_state(4)
    rand_state_y = rand_state(4)
    neg_phase_list = []
    
    print(rand_state_x)
    print(rand_state_y)
    print(MCMC(J_list, rand_state_x, rand_state_y))
            
    # cost_total = 0
    # for spins in data:
    #      cost = energy(J_list, spins)
    #      cost_total+=cost
    #      #print(cost)
    # print(cost_total)

    
