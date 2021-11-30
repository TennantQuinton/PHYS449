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
- 'run training': Experimental implementation. Can be set to 1 (yes) or 0 (no) to set if we want to run through training. If set to no and the saved PyTorch model exists in the proper location then it will just create 100 pdf digit images from the pretrained model. If set to yes then the program will run through training whether the model exists or not.

## Running `main.py`

To run `main.py`, use

```sh
python main.py -in_file data/even_mnist.csv -json_file param/parameters.json -o outputs/ -epoch_dir epoch_outputs/ -n 100 -verbosity 2
```
