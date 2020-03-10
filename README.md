# ACC 2020 Convexified contextual optimization for on-the-fly control of smooth systems

- Figure 1
- Figure 2
- Empirical comparison of the proposed approach, `C2Opt`, with `SINDYc` and `CGP-UCB`
    - Figure 3 and 4 are given by C2Opt.ipynb (Try it on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abyvinod/ACC2020_C2Opt/master?filepath=C2Opt.ipynb))

## Reproducibility instructions

1. Create conda environment via `conda create -n
   congolexamples python=3.7 scipy matplotlib jupyter`
1. Install `gurobi` via `conda install -c gurobi gurobi`
1. Install `congol` via `pip install -e .` in the `/congol`
   directory.
1. Run `jupyter notebook` and then run the notebook `C2Opt.ipynb`
