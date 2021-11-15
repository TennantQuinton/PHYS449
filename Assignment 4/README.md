# PHYS449

## Dependencies

- numpy
- os
- argparse
- matplotlib

## Arguments
 - '-data_path': The relative file to the in.txt data (default data/in.txt)
 - '-output_path': The relative directory to the output (default output/)
 - '-verb': The verbosity of the program. Can be set from low-0 to high-2 (default 2)
 - '-n_epochs': The number of epochs in which we adjust our Jij values (default 5)
 - '-n_plambda': Stating the number of times to compare two states in the MCMC before taking the final one (default 1000)


## Running `main.py`

To run `main.py`, use

```sh
python main.py -data_path data/in.txt -output_path output/ -verb 2 -n_epochs 5 -n_plambda 1000
```
