# Harapanahalli_CDC2025

The code accompanying the CDC 20225 submission entitled "Parametric Reachable Sets Via Controlled Dynamical Embeddings".

# Installing immrax

`immrax` is a tool for interval analysis and mixed monotone reachability analysis in JAX.

Inclusion function transformations are composable with existing JAX transformations, allowing the use of Automatic Differentiation to learn relationships between inputs and outputs, as well as parallelization and GPU capabilities for quick, accurate reachable set estimation. For more information, please see the current [repository](https://github.com/gtfactslab/immrax).

You will need to install the version of `immrax` in this repository provided in the `immrax` directory, which will run the examples from the paper.

## Setting up a `conda` environment

We recommend installing JAX and `immrax` into a `conda` environment ([miniconda](https://docs.conda.io/projects/miniconda/en/latest/)).
```shell
conda create -n immrax python=3.12
conda activate immrax
```

## Installing `immrax`

Change into the `immrax` directory and `pip install` it.

```shell
cd immrax
pip install .
```
You may need to install the following packages if the build fails.
```shell
sudo apt-get install libcdd-dev libgmp-dev
```

# Running the examples

## Van der Pol oscillator
The Van der Pol oscillator can be run in the `VanderPol.ipynb` Jupyter notebook.

## Robot arm
For the robot arm example in `RobotArmAdjoint.ipynb`, you will need to install `cvxpy`.
```shell
pip install cvxpy
```

We recommend using [MOSEK](https://www.mosek.com/) as the solver for `cvxpy`. You will need to follow the instructions on their installation page to obtain a license and then install it.
```shell
pip install mosek
``` 
If you don't want to go through the hassle of installing MOSEK, you can use a default installed solver, but it will be slower. To do this, simply change `CVXPY_SOLVER` to `cp.SCS` in the `RobotArmAdjoint.ipynb` notebook.
