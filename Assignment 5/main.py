'''
PHYS 449 -- Fall 2021
Assignment 5
    Name: Quinton Tennant
    ID: 20717788
'''
import os, json, argparse, numpy as np, matplotlib.pyplot as plt, pandas as pd
import torch as T, torch.nn as nn, torch.optim as optim, torchvision, torchvision.transforms as tf

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

    # Convert to PyTorch tensor and normalize values
    training_data_tens = T.tensor(training_data)/255.0
    training_labels_tens = T.tensor(training_labels)
    testing_data_tens = T.tensor(testing_data)/255.0
    testing_labels_tens = T.tensor(testing_labels)

    # Convert to tensor with both data and labels. Then load the data
    training_tensor = T.utils.data.TensorDataset(training_data_tens, training_labels_tens)
    testing_tensor = T.utils.data.TensorDataset(testing_data_tens, testing_labels_tens)
    load_training = T.utils.data.DataLoader(training_tensor, batch_size = batch_size, shuffle = True)
    load_testing = T.utils.data.DataLoader(testing_tensor, batch_size = batch_size, shuffle = True)
    # print(training_data_tens)
    # print(type(training_data_tens[0][0][0][0].item()))

    return (load_training, load_testing)

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
            # Calling the functions above
            converted_data = conversion(data_in, testing_data_size, batch_size)
            # digit_rec = training_testing(converted_data[0], converted_data[1], input_size, hidden_layer_size, output_size, learning_rate, num_epochs)
            
            # Print update
            print('\nAfter running through the {0} Epochs we found:\n\tTraining Loss: {1}\n\tTesting Loss: {3}\n\tAccuracy: {2}%'.format(digit_rec[0], digit_rec[1], digit_rec[2], digit_rec[3]))
            
            # Write the results to the .out file
            with open(out_file, 'w') as out_file:
                out_file.write('After running through the {0} Epochs we found:\n\tTraining Loss: {1}\n\tTesting Loss: {3}\n\tAccuracy: {2}%'.format(digit_rec[0], round(digit_rec[1], 4), round(digit_rec[2], 2), round(digit_rec[3], 4)))
            
            plt.title('Losses of Digit Recognition Program in Training and Testing')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig("{0}/{1}/Losses.jpg".format(my_absolute_dirpath, pathname))
            print('Plotting to {0}/{1}/Losses.jpg'.format(my_absolute_dirpath, pathname))
        else:
            print('Filepath {0} does not exist'.format(json_file))
    else:
        print('Filepath {0}/data/even_mnist.csv does not exist'.format(my_absolute_dirpath))
