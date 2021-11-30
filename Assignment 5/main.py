'''
PHYS 449 -- Fall 2021
Assignment 5
    Name: Quinton Tennant
    ID: 20717788
'''
import os, json, argparse, numpy as np, matplotlib.pyplot as plt, pandas as pd
import torch, torch.nn as nn, torch.optim as optim, torchvision, torchvision.transforms as transforms, torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import random

kernel_size = 2 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling

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
    
    def encoding(self, x):
        z = F.relu(self.encode1(x))
        z = F.relu(self.encode2(z))
        return self.fc_mu(z), self.fc_var(z)
    
    def decoding(self, x):
        z = F.relu(self.decode1(x))
        z = F.relu(self.decode2(z))
        return F.sigmoid(self.decode3(z))
    
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
    training_tensor = TensorDataset(training_data_tens, training_labels_tens)
    testing_tensor = TensorDataset(testing_data_tens, testing_labels_tens)
    # print(training_tensor.tensors[0].shape)
    # quit()
    load_training = DataLoader(training_tensor, batch_size = batch_size, shuffle = True)
    load_testing = DataLoader(testing_tensor, batch_size = batch_size, shuffle = True)

    return (load_training, load_testing)

# Function used for calculating the KL Divergence from input
def KL_Div(logvar, mu):
    return -0.5 * torch.sum(1 + logvar - mu**2 - np.exp(logvar))

# Training loop function
def training(loss_f, dataloader, epoch):    
    model.train()
    loss_total = 0
    count = 0
    
    for data in dataloader:
        data = data[0]
        optimz.zero_grad()
        
        rec, mu, logvar = model(data)
        BCE_loss = loss_f(rec, data)
        loss = BCE_loss + KL_Div(logvar, mu)
        loss.backward()
        
        loss_total += loss.item()
        optimz.step()
        count+=1
        
        if count % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, count * len(data), len(dataloader.dataset),
                100. * count / len(dataloader), loss.item() / len(data)))
            
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_total / count))
    return loss_total/count

def testing(loss_f, dataloader):
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
    print('====> Test set loss: {:.4f}'.format(test_loss))
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
                input_size = paras['input length']
                hidden_layer_size = paras['hidden layer size']
                output_size = paras['output size']
                
                learning_rate = paras['learning rate']
                num_epochs = paras['number of epochs']
                batch_size = paras['batch size']
                
                converted_data = conversion(data_in, testing_data_size, batch_size)
                
                model = var_aenc(196, 256, 128, 2)
                optimz = optim.Adam(model.parameters(), learning_rate)
                loss_f = nn.BCELoss(reduction='sum')
                
                train_loss = []
                test_loss = []
                for e in range(num_epochs):
                    training(loss_f, converted_data[0], e)
                    testing(loss_f, converted_data[1])
                    # training_loss = training(model, optimz, loss_f, converted_data[0])
                    # testing_loss = testing(model, loss_f, converted_data[1])
                    
                    # train_loss.append(training_loss)
                    # test_loss.append(testing_loss)
                    # print('Epoch {0}/{1}'.format(e, num_epochs))
                    # print('Training Loss: {0}, Testing Loss: {1}'.format(training_loss, testing_loss))
        else:
            print('Filepath {0} does not exist'.format(json_file))
    else:
        print('Filepath {0}/data/even_mnist.csv does not exist'.format(my_absolute_dirpath))
