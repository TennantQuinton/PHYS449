# PHYS449

## Dependencies

- os
- json
- argparse
- numpy
- matplotlib
- PyTorch
- random

## Attributes

- "hidden layers": The number of nodes in the hidden layer
- "epoch max": The number of epochs to run through in training
- "solution step size": How large of increments when plotting the NN solution
- "cut off plot": Indicates whether the plot should cut off at the upper and lower bounds (1=yes, 0=no)

- Optimizer: Adam
- Model: Linear, ReLU, Linear
- Loss Function: Mean Squared Error

## Running `main.py`

To run `main.py`, use

```sh
python main.py --param param/param.json -v 2 --res_path plots --x_field "-y/np.sqrt(x**2 + y**2)" --y_field "x/np.sqrt(x**2 + y**2)" --lb -1.0 --ub 1.0 --n_tests 3
```

Important: When runnning main.py the required hyphens in parameter names is instead underscores (e.g. --res-path should be --res_path)
