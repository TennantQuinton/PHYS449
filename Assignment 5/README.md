# PHYS449

## Dependencies

- os
- json
- argparse
- numpy
- matplotlib
- pandas
- Pytorch (torch, torchvision)

## Arguments
- '-in_file': The relative path of a file containing the flattened input images. Default: data/even_mnist.csv
- '-json_file': The relative path of a file containing the json parameters. Default: param/parameters.json
- '-o': The relative path of a directory for the final outputs. Default: outputs/
- '-epoch_dir': The relative path of a directory for the grid of images created after each epoch within the output folder. \n\tUsed to see how the reconstruction improves through training. Default: epoch_outputs/
- '-n': The relative path of a file containing the json parameters. Default: 100
- '-verbosity': The verbosity of the python program. Default: 2

## Parameters
- 'testing data size': The number of images to set aside from the total dataset for testing
- 'batch_size': The batch size used in each training run
- 'learning rate': The learning rate for the model
- 'number of epochs': The number of epochs run through by training

## Running `main.py`

To run `main.py`, use

```sh
python main.py -in_file data/even_mnist.csv -json_file param/parameters.json -o outputs/ -epoch_dir epoch_outputs/ -n 100 -verbosity 2
```
