'''
PHYS 449 -- Fall 2021
Assignment 2
    Name: Quinton Tennant
    ID: 20717788
'''
import os, json, argparse, numpy as np, matplotlib.pyplot as plt, pandas as pd
import torch as T
import torch.nn as nn
import torch.optim as optim

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

    return (load_training, load_testing)

def training_testing(load_training, load_testing, input_size, hidden_layer, output_size, learning_rate, epoch_max):

    dataiter = iter(load_training)
    images, labels = dataiter.next()

    # get inputs from json file
    input_size = input_size*input_size #14*14 # =196
    hidden_layer = hidden_layer #100
    output_size = output_size #9
    learning_rate = learning_rate #0.001
    epoch_max = epoch_max #30

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

    # Initalizing lists for plotting
    obj_vals = []
    cross_vals = []

    # Initialize the plot
    figure = plt.figure(figsize=(15,10))
    plt.gca()

    # Loop over the number of epochs specified
    for epoch in np.arange(1, epoch_max+1, 1):
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
            training_loss = loss_total/(len(load_training)) # Training loss
        
        # Print an update
        print('\nEpoch: {0}/{1},\nTraining Loss: {2}'.format(epoch, epoch_max, training_loss))
        obj_vals.append(training_loss)

    # Plotting the loss over iterations
    plt.plot(np.arange(1, epoch_max+1, 1), obj_vals, label='Training Loss')

    # Initializing the variables used for finding the accuracy/loss
    num_correct, num_wrong, num_total, testing_loss = 0, 0, 0, 0

    # Loop through both the images and labels in the 3000 testing dataset
    for images, labels in load_testing:
        # Loop through the length of each
        for index in np.arange(0, len(labels), 1):
            # Update the denominator
            num_total+=1

            # Find the images being used and reflatten them
            using = images[index].view(1, 196)
            images = images.view(images.shape[0], -1)

            # Finding the log of probabilities without gradient use
            with T.no_grad():
                log_probs = model(using)

            # Removing the log and creating a list
            probs = T.exp(log_probs)
            probability = list(probs.numpy()[0])

            # Using our model find what it thinks the number is
            prediction = probability.index(max(probability))
            # What is the number actually from the labels:
            actually = labels.numpy()[index]

            # Conditional: Find if the prediction is correct
            if (actually == prediction):
                # Update for accuracy usage
                num_correct+=1
            else:
                ## Plot the image of what was found incorrectly (for interest)
                #plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
                #plt.show()
                num_wrong += 1
                print('Does not match')

            # Find how far off the prediction was
            loss = abs(prediction-actually)/(num_total)
            testing_loss += loss

            # Print update
            print('\nWritten Digit: {0}, Recognized as: {1},\nTesting Loss: {2}'.format(actually, prediction, testing_loss))
            cross_vals.append(testing_loss)
    # The testing accuracy is the percentage of correct identifications
    testing_accuracy = round(((num_correct/num_total)*100),2)
    # Print update
    print("Model Accuracy = {0}%".format(testing_accuracy))

    # Plot the losses over time
    plt.plot(np.arange(1, (len(cross_vals))+1, 1)/(len(cross_vals)/epoch_max), cross_vals, label='Testing Loss')

    return (epoch_max, obj_vals[-1], testing_accuracy, cross_vals[-1], figure)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 2: Tennant, Quinton (20717788)')
    parser.add_argument('json_file', default='param/parameters.json', help='The relative path of a file containing the json parameters')

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
            digit_rec = training_testing(converted_data[0], converted_data[1], input_size, hidden_layer_size, output_size, learning_rate, num_epochs)
            
            # Print update
            print('\nAfter running through the {0} Epochs we found:\n\tTraining Loss: {1}\n\tTesting Loss: {3}\n\tAccuracy: {2}%'.format(digit_rec[0], digit_rec[1], digit_rec[2], digit_rec[3]))
            
            # Write the results to the .out file
            with open(out_file, 'w') as out_file:
                out_file.write('After running through the {0} Epochs we found:\n\tTraining Loss: {1}\n\tTesting Loss: {3}\n\tAccuracy: {2}%'.format(digit_rec[0], digit_rec[1], digit_rec[2], digit_rec[3]))
            
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