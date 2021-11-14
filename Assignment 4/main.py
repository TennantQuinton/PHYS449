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


def data_tensor_create(data):
    spin_list = []
    for line in data:
        for i in line:
            spin_list.append(int(('{0}1'.format(i))))
    spin_array = np.array(spin_list)
    spin_array = spin_array.reshape((-1, 1, 4))
    spin_tensor = (T.tensor(spin_array))
    
    return spin_tensor

def energy(Js, spins):
    E_sum = 0
    for k, i in enumerate(spins[0]):
        E_sum += -(Js[k] * spins[0][k].item() * spins[0][(k+1)%len(spins[0])].item())
    return E_sum   

def closure():
    optim.zero_grad()
    loss = energy(J_list, data[0])
    loss.backward()
    return loss
    
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
    data = data_tensor_create(data_in)
    
    # Initialize the guesses for J_ij
    J_list = np.random.choice([-1, 1], 4)
    J_tens = T.Tensor(J_list)
        
    print('Random initalization of J: {0}'.format(J_list))
    print('Associated cost of first coupling: {0}'.format(energy(J_list, data[0])))
    
    optim = T.optim.SGD(J_tens, lr=0.1)
    
    J_pt = [J_tens]
    cost_pt = [energy(J_tens, data[0])]
    x = [0]
    
    for i in range(100):
        optim.step(closure)
        if (i + 1) % 5 == 0:
            x.append(i)
            J_up = optim.param_groups[0]['params']
            costn = energy(J_tens, data[i])
            J_pt.append(J_up)
            cost_pt.append(costn)
            
            # for clarity, the angles are printed as numpy arrays
            print("Energy after step {:5d}: {: .7f} | Angles: {}".format(
                i+1, costn, J_up.detach().numpy()),"\n"
            )
            
    # cost_total = 0
    # for spins in data:
    #      cost = energy(J_list, spins)
    #      cost_total+=cost
    #      #print(cost)
    # print(cost_total)

    
