'''
PHYS 449 -- Fall 2021
Assignment 2
    Name: Quinton Tennant
    ID: 20717788
'''

import os, json, argparse, numpy as np, matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def digit_id(input_data):
    stop_iterating_at = (len(input_data)) - 3000

    index = 0
    for row in input_data:
        data_array = (np.array(np.split((row[0:-1]), 14)))
        plt.imshow(data_array, cmap='Greys')
        plt.show()
        if (index == stop_iterating_at):
            break
        index+=1
    

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 1: Tennant, Quinton (20717788)')
    parser.add_argument('-in_file', default='data/1.in', help='The relative path of a file containing the input data. Defaults to \'data/1.in\'')
    parser.add_argument('-json_file', default='data/1.json', help='The relative path of a file containing the json parameters. Defaults to \'data/1.json\'')

    # Receiving the command line arguments
    args = parser.parse_args()
    in_file = 'data/even_mnist.csv' #args.in_file
    json_file = 'param/parameters.json' #args.json_file

    # Getting the absolute file path from the relative
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    data_in = np.loadtxt("{0}/{1}".format(my_absolute_dirpath, in_file))
    digit_id(data_in)

