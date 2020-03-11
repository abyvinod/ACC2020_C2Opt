| Title      | Convexified contextual optimization for on-the-fly control of smooth systems                 |
|------------|----------------------------------------------------------------------------------------------|
| Authors    | Abraham P. Vinod, Arie Israel, and Ufuk Topcu                                                |
| Conference | American Control Conference, 2020                                                            |

---
**NOTE**

This repository uses a Python module **congol** for constrained global optimization for Lipschitz smooth functions. Please contact the author at [aby.vinod@gmail.com](mailto:aby.vinod@gmail.com) for access to this module (included in this repository as a submodule). This Python module will be released to the public shortly.

---

This repository provides codes to reproduce the figures of the paper.
- Figure 1
- Figure 2
- Empirical comparison of the proposed approach, `C2Opt`, with `SINDYc` and `CGP-UCB`
    - Figure 3 and 4 are given by C2Opt.ipynb (Try it on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abyvinod/ACC2020_C2Opt/master?filepath=C2Opt.ipynb))

## Reproducibility instructions

1. Create conda environment via `conda create -n
   congolexamples python=3.7 scipy matplotlib jupyter`
1. Install `gurobi` via `conda install -c gurobi gurobi`
    1. Availability of `gurobi` provides for a faster computation. However, 
       its use requires a license (see https://www.gurobi.com/ for more details). 
    1. For a GPLv3-based implementation, we have also formulated the problem
       using `cvxpy` (https://www.cvxpy.org/).
1. Install `congol` via `cd congol && pip install -e . && cd ..`.
1. Install `gpyopt` via `pip install gpyopt==1.2.5`
1. Run `jupyter notebook` and then open the desired notebook.
