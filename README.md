| Title      | Convexified contextual optimization for on-the-fly control of smooth systems                 |
|------------|----------------------------------------------------------------------------------------------|
| Authors    | Abraham P. Vinod, Arie Israel, and Ufuk Topcu                                                |
| Conference | American Control Conference, 2020                                                            |

This repository provides codes to reproduce the figures of the paper.
- Correct-by-construction approximation bounds under
  smoothness, convexity, and monotonicity
    - Figures 1 and 2 are given by `Approximation bounds via side information.ipynb` 
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abyvinod/ACC2020_C2Opt/master?filepath=Approximation%20bounds%20via%20side%20information.ipynb)
- Empirical comparison of the proposed approach, `C2Opt`,
  with `SINDYc`, `CGP-LCB`, and the true one-step optimal
  trajectory
    - Figures 3 and 4 are given by `One-step control of
      unicycle.ipynb` [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abyvinod/ACC2020_C2Opt/master?filepath=One-step%20control%20of%20unicycle.ipynb)
    - You have to change `figure_3 = True` to `figure_3=False` in the 
    first cell to obtain Figure 4.

## Reproducibility instructions

Use `$ conda env create -f conda_env_for_acc2020.yml` to get the 
exact environment used for developing and testing this code.

### Step-by-step guide

1. Create conda environment
    - `$ conda create -n acc2020_c2opt python=3.7 scipy matplotlib 
    jupyter tqdm pandas`
1. Install `gurobi`
    - `$ conda install -c gurobi gurobi`
    - Availability of `gurobi` provides for a faster
      computation. However, its use requires a license (see
      https://www.gurobi.com/ for more details). 
1. Install `cvxpy` and `gpyopt`
    - `$ pip install cvxpy gpyopt==1.2.5`
    - Free software alternative to `gurobi` (See www.cvxpy.org 
      for more details)
1. Install `coveropt` 
    - `$ cd coveropt && pip install -e . && cd ..`
1. Run `$ jupyter notebook` and then open the desired
   notebook.

## Scripts

### Notebooks and the associated script files

1. `Approximation bounds via side
   information.ipynb` or `Approximation bounds via side
   information.py` generates Figures 1 and 2
    - `BoundsWithSideInfo.py` provides the class that uses
      `coveropt/coveropt/classes/LipSmoothFun.py` to implement
      the correct-by-construction bounds
1. `One-step control of unicycle.ipynb` or `One-step control of
   unicycle.py` generates Figures 3 and 4
    - `MyopicDataDrivenControl.py` provides the base class
      for one-step control of unknown dynamics
    - `MyopicDataDrivenControlSINDYc.py` implements the
      SINDYc method via LASSO 
      - See *Kaiser, Eurika, J.  Nathan Kutz, and Steven L.
        Brunton. "Sparse identification of nonlinear
        dynamics for model predictive control in the
        low-data limit." Proceedings of the Royal Society A
        474.2219 (2018): 20180335* for more details
    - `MyopicDataDrivenControlContextGP.py` implements the
      approach utilizing Contextual Gaussian Process
      optimized via lower confidence bound. This method
      utilizes `gpyopt`.
      - See *Krause, Andreas, and Cheng S. Ong. "Contextual
        gaussian process bandit optimization." Advances in
        neural information processing systems. 2011* and
        https://github.com/SheffieldML/GPyOpt for more
        details
    - `MyopicDataDrivenControlTrue.py` provides a benchmark
      solution --- the optimal one-step trajectory under the
      knowledge of the dynamics. It uses `scipy.optimize` to
      solve the one-step receding horizon control problem.
    - `coveropt/coveropt/classes/applications/C2Opt/` implements
      the proposed `C2Opt` method.

# Contact 

Please contact Abraham Vinod (aby.vinod@gmail.com) for bugs
or questions regarding this code.
