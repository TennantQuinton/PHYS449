'''
PHYS 449 -- Fall 2021
Assignment 1
    Name: Quinton Tennant
    ID: 20717788
'''

import os, json, argparse, numpy as np

def linearRegression(input_data):
    # Slicing the data into the first M-1 columns and the final column
    y_dat = np.array(input_data[:,-1]).T
    x_dat = np.array(input_data[:,0:-1])

    # Making the first column of the design matrix 1s
    ones = np.ones(len(input_data))
    x_dat = np.insert(x_dat, 0, ones, axis=1)

    # Going step-by-step through finding w
    # First: transpose the design matrix
    transpose = x_dat.T
    # Second: multiply the transpose with the original
    multiply_1 = np.matmul(transpose, x_dat)
    # Third: take the inverse of that multiplied matrix result
    inverse = np.linalg.inv(multiply_1)
    # Fourth: multiply the inverted matrix and the transpose
    multiply_2 = np.matmul(inverse, transpose)
    # Fifth: multiply by the T matrix
    multiply_3 = np.matmul(multiply_2, y_dat)

    # Output the results (Need to output to .out file)
    return multiply_3

def gradDescent(input_data, learning_rate, num_iter):
    # Our first "guess" of w_0
    w_0 = np.zeros(len(input_data[0]))
    update = np.zeros(len(input_data))

    # Slicing the data into the first M-1 columns and the final column
    y_dat = np.array(input_data[:,-1]).T
    x_dat = np.array(input_data[:,0:-1])

    # Making the first column of the design matrix ones
    ones = np.ones(len(input_data))
    x_dat = np.insert(x_dat, 0, ones, axis=1)

    # transposing x outside of our loop for efficiency
    transpose_x = x_dat.T

    # Initializing iteration count for loop
    iteration = 0
    while iteration <= num_iter:
        # Creating a matrix that is updated with each iteration
        update = np.matmul(x_dat, w_0)
        # Calculating the loss we incur by our decision
        loss = update - y_dat
        # Calculating the normalized gradient
        grad = np.matmul(transpose_x, loss)/(len(input_data))
        # Updating w_0 through each iteration
        w_0 = w_0 - (learning_rate*grad)

        # Conditional that if our gradient reaches a sufficiently small value then exit loop early
        if (np.isclose(grad.all(), 0) and iteration > 1):
            print("Breaking at iteration {0} with gradient sufficiently small".format(iteration, grad))
            break
        
        # incrementing our iteration
        iteration+=1
    return(w_0)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Assignment 1: Tennant, Quinton (20717788)')
    parser.add_argument('in_file', default='data/1.in', help='The relative path of a file containing the input data. Defaults to \'data/1.in\'')
    parser.add_argument('json_file', default='data/1.json', help='The relative path of a file containing the json parameters. Defaults to \'data/1.json\'')

    # Receiving the command line arguments
    args = parser.parse_args()
    in_file = args.in_file
    json_file = args.json_file

    # Getting the absolute file path from the relative
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    # Checking if the in file exists
    if os.path.isfile("{0}/{1}".format(my_absolute_dirpath, in_file)):
        # Reading in the data and running the linear regression
        data_in = np.loadtxt("{0}/{1}".format(my_absolute_dirpath, in_file))

        # Getting path and filename for output file usage
        filename = (((in_file).split('.')[0]).split('/'))[-1]
        pathname = (((in_file).split('.')[0]).split('/'))[-2]
        out_file = "{0}/{1}/{2}.out".format(my_absolute_dirpath, pathname, filename)

        # Checking if the json file exists
        if os.path.isfile("{0}/{1}".format(my_absolute_dirpath, json_file)):
            # Reading the json file inputs
            json_input_file = "{0}/{1}".format(my_absolute_dirpath, json_file)

            # Reading the json parameters
            with open(json_input_file, 'r') as file:
                paras = json.load(file)
                learning_rate = paras['learning rate']
                num_iter = paras['num iter']

            # The outputs from running our two functions
            # LINEAR REGRESSION
            lin_reg_output = linearRegression(data_in)
            print("w_analytic: {0}".format(lin_reg_output))
            
            # GRADIENT DESCENT
            grad_des_output = gradDescent(data_in, learning_rate, num_iter)
            print("w_GD: {0}".format(grad_des_output))

            # Write the results to the .out file
            with open(out_file, 'w') as out_file:
                for i in lin_reg_output:
                    out_file.write("{0}\n".format(i))
                out_file.writelines("\n")
                for j in grad_des_output:
                    out_file.write("{0}\n".format(j))

        else:
            print('Filepath {0}/{1} does not exist'.format(my_absolute_dirpath, json_file))

    else:
        print('Filepath {0}/{1} does not exist'.format(my_absolute_dirpath, in_file))