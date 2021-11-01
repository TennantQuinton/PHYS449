# PHYS449

## Dependencies

- os
- json
- argparse
- numpy
- matplotlib
- PyTorch
- random

## Running `main.py`

To run `main.py`, use

```sh
python main.py --param param/param.json -v 2 --res_path plots --x_field "-y/np.sqrt(x**2 + y**2)" --y_field "x/np.sqrt(x**2 + y**2)" --lb -1.0 --ub 1.0 --n_tests 3
```

Important: When runnning main.py the required hyphens in parameter names is instead underscores (e.g. --res-path should be --res_path)
