'''
PHYS 449 -- Fall 2021
Assignment 2
    Name: Quinton Tennant
    ID: 20717788
'''

from operator import truediv
import os, json, argparse, numpy as np, matplotlib.pyplot as plt, pandas as pd
import torch as T, torch.nn as nn, torchvision as TV, torchvision.transforms as tf, torch.optim as optim
from torchvision.utils import make_grid

def conversion(input_data):
    # Training data split (take the first 26492 digits) in Pandas df format
    training = (input_data[0:-2999])
    training_data = training[training.columns[:-1]]
    training_labels = training[training.columns[-1]]

    # Testing data split (take the final 3000 digits) in Pandas df format
    testing = (input_data[26491:])
    testing_data = testing[testing.columns[:-1]]
    testing_labels = testing[testing.columns[-1]]   

    # Convert to numpy array
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    testing_data = np.array(testing_data)
    testing_labels = np.array(testing_labels)

    # Reshape to the correct dimensions for analysis
    training_data = training_data.reshape((-1, 1, 14, 14))
    training_labels = training_labels.reshape(training_labels.shape[0])
    testing_data = testing_data.reshape((-1, 1, 14, 14))
    testing_labels = testing_labels.reshape(testing_labels.shape[0])

    # Convert to PyTorch tensor and normalize values
    training_data_tens = T.tensor(training_data)/255.0
    training_labels_tens = T.tensor(training_labels)
    testing_data_tens = T.tensor(testing_data)/255.0
    testing_labels_tens = T.tensor(testing_labels)

    batch_size = 100

    # Convert to tensor with both data and labels. Then load the data
    training_tensor = T.utils.data.TensorDataset(training_data_tens, training_labels_tens)
    testing_tensor = T.utils.data.TensorDataset(testing_data_tens, testing_labels_tens)
    load_training = T.utils.data.DataLoader(training_tensor, batch_size = batch_size, shuffle = True)
    load_testing = T.utils.data.DataLoader(testing_tensor, batch_size = batch_size, shuffle = True)

    return (load_training, load_testing)

def training_testing(load_training, load_testing):

    dataiter = iter(load_training)
    images, labels = dataiter.next()

    # get inputs from json file
    input_size = 14*14 # =196
    hidden_layer = 100
    output_size = 9
    learning_rate = 0.001
    epoch_max = 30

    # Initialize the model
    model = nn.Sequential(nn.Linear(input_size, hidden_layer),
                            nn.ReLU(),
                            nn.Linear(hidden_layer, output_size),
                            nn.LogSoftmax(dim=1))

    # Optimizer/loss functions used for learning
    optimizer = optim.Adam(model.parameters(), learning_rate)
    nn_loss = nn.NLLLoss()

    # Set the data used for the images and labels within the learning
    images, labels = next(iter(load_training))
    images = images.view(images.shape[0], -1)

    # Get the probabilities (log from model) and the loss initials
    log_probs = model(images)
    loss = nn_loss(log_probs, labels)

    # Loop over the number of epochs specified
    for epoch in np.arange(0, epoch_max+1, 1):
        # Intialize the total loss to be updated over loops
        loss_total = 0

        # Loop over the images and labels we found earlier
        for images, labels in load_training:
            images = images.view(images.shape[0], -1)       # Re-flatten the images
            optimizer.zero_grad()                           # Run the training optimizer with the set model
            output = model(images)                          # Find the output
            loss = nn_loss(output, labels)                  #   and the loss
            loss.backward()                                 # backpropogate
            optimizer.step()                                # Continue
            loss_total += loss.item()                       # Update the total loss
        
        # Print an update
        print('Epoch [{}/{}]'.format(epoch, num_epochs)+\
            '\tTraining Loss: {:.4f}'.format(train_val)+\
            '\tTest Loss: {:.4f}'.format(test_val))
        print('Epoch: {0}, Training Loss: {1}'.format(epoch, loss_total/(len(load_training))))



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
    converted_data = conversion(data_in)
    training_testing(converted_data[0], converted_data[1])

