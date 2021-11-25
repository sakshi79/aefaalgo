# AEFA

AEFA is a Python library for AEFA: Artificial electric field algorithm, a novel algorithm for solving non-linear optimization problems.
The details of the algorithm can be found here at: https://medium.com/artifical-mind/artificial-electric-field-algorithm-for-optimization-fb6f57f413b4
You may access the paper for the same at http://www.sciencedirect.com/science/article/pii/S2210650218305030

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install aefaalgo.

```bash
pip install aefaalgo
```

## Usage

```python
from aefaalgo.aefa_optimize import aefa

# returns optimum fitness value and space coordinates
aefa().optimize(N, max_iter, func_num)
Keyword arguments:
N: number of particles in search space

max_iter: number of iterations

func_num: Specifies the function to be optimized

Optional Keyword Arguments: 
tag: specifies whether we want maxima or minima.
0 by default for maximization. Specify tag=1 for minimization.

Rpower: exponent for the normalized distance between the particles.
Default value 1

FCheck: This factor ensures that only 2-6% charges apply force to others in the last iterations.
Set to True by default. 

show_plot: True if you want to visualize convergence to the optimum, False otherwise and default.

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)