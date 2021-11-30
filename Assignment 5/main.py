'''
PHYS 449 -- Fall 2021
Assignment 5
    Name: Quinton Tennant
    ID: 20717788
'''

# Imports
import os, json, argparse, numpy as np, matplotlib.pyplot as plt, pandas as pd
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image

# Class used for defining our model
class var_aenc(nn.Module):
    def __init__(self, in_size, h_size1, h_size2, mv_size):
        super(var_aenc, self).__init__()
        
        # ENCODER
        self.encode1 = nn.Linear(in_size, h_size1)
        self.encode2 = nn.Linear(h_size1, h_size2)
        
        # MEAN/VAR LAYERS
        self.fc_mu = nn.Linear(h_size2, mv_size)
        self.fc_var = nn.Linear(h_size2, mv_size)
        
        # DECODER
        self.decode1 = nn.Linear(mv_size, h_size2)
        self.decode2 = nn.Linear(h_size2, h_size1)
        self.decode3 = nn.Linear(h_size1, in_size)
    
    # Running through encoding layers
    def encoding(self, x):
        z = F.relu(self.encode1(x))
        z = F.relu(self.encode2(z))
        return self.fc_mu(z), self.fc_var(z)
    
    # Running through decoding layers
    def decoding(self, x):
        z = F.relu(self.decode1(x))
        z = F.relu(self.decode2(z))
        return torch.sigmoid(self.decode3(z))
    
    # Forward prop
    def forward(self, x):
        mu, lvar = self.encoding(x.view(-1, 196))
        std = torch.exp((1/2)*lvar)
        eps = torch.randn_like(std)
        z = mu + (std * eps)    # Sampling
        return self.decoding(z), mu, lvar

def conversion(input_data, test_size, batch_size):
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
    training_data = training_data.reshape((-1, 14, 14))
    training_labels = training_labels.reshape(training_labels.shape[0])
    testing_data = testing_data.reshape((-1, 14, 14))
    testing_labels = testing_labels.reshape(testing_labels.shape[0])

    # Convert to PyTorch tensor and normalize values
    training_data_tens = torch.tensor(training_data)/255.0
    training_labels_tens = torch.tensor(training_labels)
    testing_data_tens = torch.tensor(testing_data)/255.0
    testing_labels_tens = torch.tensor(testing_labels)

    # Convert to tensor with both data and labels. Then load the data
    training_tensor = TensorDataset(training_data_tens, training_labels_tens)
    testing_tensor = TensorDataset(testing_data_tens, testing_labels_tens)
    load_training = DataLoader(training_tensor, batch_size = batch_size, shuffle = True)
    load_testing = DataLoader(testing_tensor, batch_size = batch_size, shuffle = False)

    return (load_training, load_testing)

# Loss function to combine KL Divergence and BCE into total loss
def loss_f(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 196), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# Training loop function
def training(dataloader, epoch):  
    model.train()
    loss_total = 0          # Initialize the total losee
    count = 0               # Initialize the count
    
    # Loop over the converted data
    for data in dataloader:
        # Give the data it's proper shape
        data = data[0]#.reshape((-1, 1, 14, 14))
        
        # Zero the gradient before each
        optimz.zero_grad()
        
        # Retrieve the reconstructed image, mu, and log of variance from the model on this data
        rec, mu, logvar = model(data)
        # Propogate the loss comparing the reconstructed to the actual
        loss = loss_f(rec, data, mu, logvar)
        loss.backward()
        
        # Update the total loss
        loss_total += loss.item()
        
        # Step forward
        optimz.step()
        count+=1
        
        # Every 5000th data point in the epoch print out an update containing:
        # Epoch number
        # What data point we are out of the total
        # The loss found at this point
        if ((count % 50 == 0) and (verbosity >= 2)):
            print('\tImage {1}/{2}, Loss: {4}'.format(epoch, (count * len(data)), (len(dataloader.dataset)), int(round((100. * count / len(dataloader)), 0)), round((loss.item() / len(data)), 3)))
        
    # The training loss is found by dividing our loss sum by the total data length
    train_loss = loss_total / len(dataloader.dataset)
    return train_loss

# Testing loop function
def testing(dataloader):
    model.eval()
    # Initialize the total loss and count
    loss_total = 0
    count = 0
    
    # Use without gradient
    with torch.no_grad():
        # Loop through converted data
        for data in dataloader:
            # Get data from converted loader
            data = data[0]
            
            # Find the reconstructed images, mu, and log variance
            rec, mu, logvar = model(data)
            
            # Update the total loss
            loss = loss_f(rec, data, mu, logvar)
            loss_total += loss.item()
            
            image = rec
            count += 1
    
    # Output the final loss and image
    test_loss = loss_total/len(dataloader.dataset)
    return test_loss, image
        
    
    
    
    
    

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 5: Tennant, Quinton (20717788)')
    parser.add_argument('-in_file', default='data/even_mnist.csv', help='The relative path of a file containing the flattened input images. Default: data/even_mnist.csv', type=str)
    parser.add_argument('-json_file', default='param/parameters.json', help='The relative path of a file containing the json parameters. Default: param/parameters.json', type=str)
    parser.add_argument('-o', default='outputs/', help='The relative path of a directory for the final outputs. Default: outputs/', type=str)
    parser.add_argument('-epoch_dir', default='epoch_outputs/', help='The relative path of a directory for the grid of images created after each epoch within the output folder. Used to see how the reconstruction improves through training. Default: epoch_outputs/', type=str)
    parser.add_argument('-n', default=100, help='The relative path of a file containing the json parameters. Default: 100', type=int)
    parser.add_argument('-verbosity', default=2, help='The verbosity of the python program. Default: 2', type=int)
    
    # Receiving the command line arguments
    args = parser.parse_args()
    in_file = args.in_file
    json_file = args.json_file
    result_dir = args.o
    epoch_dir = args.epoch_dir
    n_outputs = args.n
    verbosity = args.verbosity

    # Getting the absolute file path from the relative
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    # Getting arguments from the command line
    in_path = '{0}/{1}'.format(my_absolute_dirpath, in_file)
    json_path = '{0}/{1}'.format(my_absolute_dirpath, json_file)
    result_path = '{0}/{1}'.format(my_absolute_dirpath, result_dir)
    epoch_path = '{0}/{1}/{2}'.format(my_absolute_dirpath, result_dir, epoch_dir)
    model_path = '{0}/trained_model.pth'.format(my_absolute_dirpath)
    
    # Make both output directories if they don't exist
    if (os.path.exists(result_path) == False):
        os.makedirs(result_path)
    
    if (os.path.exists(epoch_path) == False):
        os.makedirs(epoch_path)
        
    if (os.path.exists('{0}/loss/'.format(result_path)) == False):
        os.makedirs('{0}/loss/'.format(result_path))
    
    # Checking if the csv data file exists
    if os.path.isfile(in_path):
        # Reading the data from the csv file
        data_in = pd.read_csv(in_path, delimiter=' ')

        # Getting path and filename for output file usage
        filename = 'report'
        pathname = (((in_file).split('.')[0]).split('/'))[-2]
        out_file = '{0}/{1}/{2}.out'.format(my_absolute_dirpath, pathname, filename)

        # Checking if our json file exists
        if os.path.isfile(json_file):
            # Open the json file
            with open(json_file, 'r') as file:
                # Getting parameters from json
                paras = json.load(file)
                testing_data_size = paras['testing data size']                
                learning_rate = paras['learning rate']
                num_epochs = paras['number of epochs']
                batch_size = paras['batch size']
                from_scratch = paras['from scratch']
                
                if (from_scratch == 0):
                    from_scratch = False
                elif (from_scratch == 1):
                    from_scratch = True
                
                # Convert the in_data
                converted_data = conversion(data_in, testing_data_size, batch_size)
                
                # Set the model and optimizer
                model = var_aenc(196, 128, 64, 2)
                optimz = optim.Adam(model.parameters(), learning_rate)
                
                # Initialize the lists for plotting
                train_loss_list = []
                test_loss_list = []
                e_list = []
                # and a grid for epoch outputs to see how the model improves (for interest)
                z_grid = torch.randn(64, 2)
                
                if (os.path.isfile(model_path) == False):
                    if (verbosity > 0):
                        print('Model does not exist. Starting Training')
                    # Loop over the epochs
                    for e in range(1, num_epochs+1):
                        if (verbosity >= 0):
                            # Update
                            print('Epoch: {0}/{1}'.format(e, num_epochs))
                        
                        # Get the training and test loss
                        train_loss = training(converted_data[0], e)
                        test_loss = testing(converted_data[1])
                        
                        if (verbosity >= 1):
                            # Update
                            print('\tAverage Training Loss: {1}, Test Loss: {2}\n'.format(e, round(train_loss, 3), round(test_loss[0], 3)))
                        
                        # Append to plotting lists
                        train_loss_list.append(train_loss)
                        test_loss_list.append(test_loss[0])
                        e_list.append(e)
                        
                        # Every 10th epoch save grid of numbers to epoch_outputs folder (for interest)
                        if ((e % 10 == 0) or (e == 1)):
                            if (verbosity > 0):
                                print('Creating a sample grid of reconstructed digits for Epoch {0}'.format(e))
                            with torch.no_grad():
                                output = model.decoding(z_grid)
                                save_image(output.view(64, 1, 14, 14), '{0}/sample{1}.jpg'.format(epoch_path, e))
                    
                    if (verbosity > 0):
                        print('Plotting Loss')
                    # Plot the loss over epochs
                    plt.figure(figsize=(10,10))
                    plt.gca()
                    plt.plot(e_list, train_loss_list, label='Training Loss')
                    plt.plot(e_list, test_loss_list, label='Testing Loss')
                    plt.title('Loss over the Epochs')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig('{0}/loss/loss.pdf'.format(result_path))
                    
                    # Save the trained model
                    torch.save(model, model_path)
                    
                elif ((os.path.isfile(model_path) == True) and (from_scratch == True)):
                    if (verbosity > 0):
                        print('Model does exist. Starting Training from scratch anyways')
                    # Loop over the epochs
                    for e in range(1, num_epochs+1):
                        if (verbosity >= 0):
                            # Update
                            print('Epoch: {0}/{1}'.format(e, num_epochs))
                        
                        # Get the training and test loss
                        train_loss = training(converted_data[0], e)
                        test_loss = testing(converted_data[1])
                        
                        if (verbosity >= 1):
                            # Update
                            print('\tAverage Training Loss: {1}, Test Loss: {2}\n'.format(e, round(train_loss, 3), round(test_loss[0], 3)))
                        
                        # Append to plotting lists
                        train_loss_list.append(train_loss)
                        test_loss_list.append(test_loss[0])
                        e_list.append(e)
                        
                        # Every 10th epoch save grid of numbers to epoch_outputs folder (for interest)
                        if ((e % 10 == 0) or (e == 1)):
                            if (verbosity > 0):
                                print('Creating a sample grid of reconstructed digits for Epoch {0}'.format(e))
                            with torch.no_grad():
                                output = model.decoding(z_grid)
                                save_image(output.view(64, 1, 14, 14), '{0}/sample{1}.jpg'.format(epoch_path, e))
                    
                    if (verbosity > 0):
                        print('Plotting Loss')
                    # Plot the loss over epochs
                    plt.figure(figsize=(10,10))
                    plt.gca()
                    plt.plot(e_list, train_loss_list, label='Training Loss')
                    plt.plot(e_list, test_loss_list, label='Testing Loss')
                    plt.title('Loss over the Epochs')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig('{0}/loss/loss.pdf'.format(result_path))
                    
                    # Save the trained model
                    torch.save(model, model_path)
                
                model = torch.load(model_path)
                if (verbosity > 0):
                    print('Creating {0} reconstructed images'.format(n_outputs))
                # Now creating n images AFTER training (as set by the problem statement)
                for i in range(1, n_outputs + 1):
                    with torch.no_grad():
                        z_grid = torch.randn(1, 2)
                        sample = model.decoding(z_grid)
                        save_image(sample.view(1, 1, 14, 14), '{0}/{1}.pdf'.format(result_path, i))
                print('Finished!')
        else:
            print('Filepath {0} does not exist'.format(json_file))
    else:
        print('Filepath {0}/data/even_mnist.csv does not exist'.format(my_absolute_dirpath))
        
    # TODO: Maybe set up a way to run with an already trained model instead of retraining every time
