'''
PHYS 449 -- Fall 2021
Assignment 5
    Name: Quinton Tennant
    ID: 20717788
'''
import os, json, argparse, numpy as np, matplotlib.pyplot as plt, pandas as pd
import torch, torch.nn as nn, torch.optim as optim, torchvision, torchvision.transforms as transforms, torch.nn.functional as F
import random

class var_aenc(nn.Module):
    def __init__(self):
        super(var_aenc, self).__init__()
        
        # ENCODER
        
        # FULLY-CONNECTED LAYERS
        
        # DECODER
        
    def forward(self, x):
        # Run through encoding
        
        # Learn what we can from the encoded data
        
        # Run through decoding
        pass

def conversion(input_data, test_size, batch_size):
    test_size = test_size
    batch_size = batch_size
    # Training data split (take the first 26492 digits) in Pandas df format
    training = (input_data[0:-test_size-1])
    training_data = training[training.columns[:-1]]
    training_labels = training[training.columns[-1]]

    # Testing data split (take the final 3000 digits) in Pandas df format
    testing = (input_data[29492-test_size-1:])
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
    # for i in range(0,5):
    #     rand_index = random.choice(range(0,2000))
    #     print(training_labels[rand_index])
    #     plt.imshow(training_data[rand_index][0], cmap='gray_r')
    #     plt.show()

    # Convert to PyTorch tensor and normalize values
    training_data_tens = torch.tensor(training_data)/255.0
    training_labels_tens = torch.tensor(training_labels)
    testing_data_tens = torch.tensor(testing_data)/255.0
    testing_labels_tens = torch.tensor(testing_labels)

    # Convert to tensor with both data and labels. Then load the data
    training_tensor = torch.utils.data.TensorDataset(training_data_tens, training_labels_tens)
    testing_tensor = torch.utils.data.TensorDataset(testing_data_tens, testing_labels_tens)
    load_training = torch.utils.data.DataLoader(training_tensor, batch_size = batch_size, shuffle = True)
    load_testing = torch.utils.data.DataLoader(testing_tensor, batch_size = batch_size, shuffle = True)
    # print(training_data_tens)
    # print(type(training_data_tens[0][0][0][0].item()))

    return (load_training, load_testing)

# Function used for calculating the KL Divergence from input
def KL_Div(logvar, mu):
    return -0.5 * torch.sum(1 + logvar - mu**2 - np.exp(logvar))

# Training loop function
def training(model, optimizer, loss_f, dataloader):    
    model.train()
    loss_total = 0
    count = 0
    
    for data in dataloader:
        data = data[0]
        optimizer.zero_grad()
        rec, mu, logvar = model(data)
        BCE_loss = loss_f(rec, data)
        loss = BCE_loss + KL_Div(logvar, mu)
        loss.backward()
        loss_total += loss.item()
        optimizer.step()
        count+=1
        
    train_loss = loss_total/count
    return train_loss

def testing(model, optimizer, loss_f, dataloader):
    model.eval()
    loss_total = 0
    count = 0
    
    with torch.no_grad():
        for data in dataloader:
            data = data[0]
            rec, mu, logvar = model(data)
            BCE_loss = loss_f(rec, data)
            loss = BCE_loss + KL_Div(logvar, mu)
            loss_total += loss.item()
            
            image = rec
            count += 1
            
    test_loss = loss_total/count
    return test_loss, image
        
    
    
    
    
    

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 5: Tennant, Quinton (20717788)')
    parser.add_argument('-json_file', default='param/parameters.json', help='The relative path of a file containing the json parameters')

    # Receiving the command line arguments
    args = parser.parse_args()
    in_file = 'data/even_mnist.csv' #args.in_file
    json_file = 'param/parameters.json' #args.json_file

    # Getting the absolute file path from the relative
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    # The direct path to the json file
    json_file = "{0}/param/parameters.json".format(my_absolute_dirpath)
    
    # Checking if the csv data file exists
    if os.path.isfile("{0}/data/even_mnist.csv".format(my_absolute_dirpath)):
        # Reading the data from the csv file
        data_in = pd.read_csv("{0}/data/even_mnist.csv".format(my_absolute_dirpath), delimiter=' ')

        # Getting path and filename for output file usage
        filename = 'report'
        pathname = (((in_file).split('.')[0]).split('/'))[-2]
        out_file = "{0}/{1}/{2}.out".format(my_absolute_dirpath, pathname, filename)

        # Checking if our json file exists
        if os.path.isfile(json_file):
            # Open the json file
            with open(json_file, 'r') as file:
                # Getting parameters from json
                paras = json.load(file)
                testing_data_size = paras['testing data size']
                batch_size = paras['batch size']
                input_size = paras['input length']
                hidden_layer_size = paras['hidden layer size']
                output_size = paras['output size']
                learning_rate = paras['learning rate']
                num_epochs = paras['number of epochs']
            
        else:
            print('Filepath {0} does not exist'.format(json_file))
    else:
        print('Filepath {0}/data/even_mnist.csv does not exist'.format(my_absolute_dirpath))
