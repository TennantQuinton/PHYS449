'''
PHYS 449 -- Fall 2021
Assignment 2
    Name: Quinton Tennant
    ID: 20717788
'''

import os, json, argparse, numpy as np, matplotlib.pyplot as plt, pandas as pd
import torch as T, torch.nn as nn, torchvision as TV, torchvision.transforms as tf
from torchvision.utils import make_grid

def digit_id(input_data):
    # Training data split (take the first 26492 digits)
    training = (input_data[0:-2999])
    training_data = training[training.columns[:-1]]
    training_labels = training[training.columns[-1]]

    # Testing data split (take the final 3000 digits)
    testing = (input_data[26491:])
    testing_data = testing[testing.columns[:-1]]
    testing_labels = testing[testing.columns[-1]]   

    # Convert to array
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    testing_data = np.array(testing_data)
    testing_labels = np.array(testing_labels)

    # Reshape to the correct dimensions for analysis
    training_data = training_data.reshape((-1, 1, 14, 14))
    training_labels = training_labels.reshape(training_labels.shape[0], 1)
    testing_data = testing_data.reshape((-1, 1, 14, 14))
    testing_labels = testing_labels.reshape(testing_labels.shape[0], 1)

    # Convert to PyTorch tensor and normalize
    training_data_tens = T.tensor(training_data)/255.0
    training_labels_tens = T.tensor(training_labels)
    testing_data_tens = T.tensor(testing_data)/255.0
    testing_labels_tens = T.tensor(testing_labels)

    training_tensor = T.utils.data.TensorDataset(training_data_tens, training_labels_tens)
    testing_tensor = T.utils.data.TensorDataset(testing_data_tens, testing_labels_tens)
    load_training = T.utils.data.DataLoader(training_tensor, batch_size = 64, num_workers = 2, shuffle = True)
    load_testing = T.utils.data.DataLoader(testing_tensor, batch_size = 64, num_workers = 2, shuffle = True)

    #print(training_data_tens.shape)
    #print(training_labels_tens.shape)

    #for index in range(1, 61):
    #    plt.subplot(6, 10, index)
    #    plt.axis('off')
    #    plt.imshow(training_data_tens[index].numpy().squeeze(), cmap='gray_r')
    #plt.show()

    epoch_tot = 10
    learning_rate = 0.001

    input_size = 196
    classes = 5
    hidden_layers = 100

    model = nn.Sequential(nn.Linear(input_size, hidden_layers),
                        nn.ReLU(),
                        nn.Linear(hidden_layers, hidden_layers),
                        nn.ReLU(),
                        nn.Linear(hidden_layers, classes),
                        nn.LogSoftmax(dim=1))
    
    optimizer = T.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    samples=len(load_training)

    for epoch in np.arange(0, epoch_tot+1, 1):
        for images, labels in enumerate(load_training):
            


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

    data_in = pd.read_csv("{0}/{1}".format(my_absolute_dirpath, in_file), delimiter=' ')
    digit_id(data_in)

