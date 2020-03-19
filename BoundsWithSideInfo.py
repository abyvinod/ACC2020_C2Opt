import cvxpy as cp
import congol as cg
import numpy as np
import multiprocessing as mp
import tqdm as tq
import time as time


class BoundsWithSideInfo(cg.LipSmoothFun):
    """
    Generate bounds on a function
    """
    def __init__(self, *args, **kwargs):
        # Default arguments
        self.is_convex = kwargs.pop('is_convex', False)
        self.is_concave = kwargs.pop('is_concave', False)
        self.is_monotone_inc = kwargs.pop('is_monotone_inc', False)
        self.is_monotone_dec = kwargs.pop('is_monotone_dec', False)
        self.fun_ub = kwargs.pop('fun_ub', None)
        self.fun_lb = kwargs.pop('fun_lb', None)
        cg.LipSmoothFun.__init__(self, *args, **kwargs)
        if self.arg_dim != 1:
            raise ValueError('Expected 1-dimensional function')
        # Overwrite the CVXPY problems
        self.cvxpy_arg = {'solver':'ECOS'}
        self._cvxpy_probs_setup()


    def get_bounds(self, arg_test_vec):
        """
        Get the upper and lower bounds of the function
        """
        lb_pcon = self.get_lower_bound_pcon(arg_test_vec)
        ub_pcvx = self.get_upper_bound_pcvx(arg_test_vec)
        lb_pcvx, ub_pcon = self.get_bounds_no_grad(arg_test_vec)
        if self.fun_lb is None:
            lb = np.max(np.vstack((lb_pcon, lb_pcvx)).T, axis=1)
        else:
            lb = np.max(np.vstack((lb_pcon,
                                   lb_pcvx,
                                   self.fun_lb(arg_test_vec))).T, axis=1)

        if self.fun_ub is None:
            ub = np.min(np.vstack((ub_pcon,
                                   ub_pcvx)).T, axis=1)
        else:
            ub = np.min(np.vstack((ub_pcon,
                                   ub_pcvx,
                                   self.fun_ub(arg_test_vec))).T, axis=1)

        return lb, ub

    def get_lower_bound_pcon(self, arg_test_vec):
        """
        Use cg.LipSmoothFun.get_pwq_lower_normalized to obtain the matrices
        """
        return self.evaluate_minorant(arg_test_vec, is_convex=self.is_convex)

    def get_upper_bound_pcvx(self, arg_test_vec):
        """
        Use cg.LipSmoothFun.get_pwq_upper_normalized to obtain the matrices
        """
        return self.evaluate_majorant(arg_test_vec, is_concave=self.is_concave)

    def get_bounds_no_grad(self, arg_test_vec, chunksize=4, verbose=True,
                           compute_ub=True, compute_lb=True):
        """
        Computes upper and lower bounds at the given collection of arg_test
        (a matrix with self.arg_dim columns)

        Use multiprocessing for parallel computation
        """

        n_arg_test, arg_dim = arg_test_vec.shape

        if arg_dim != self.arg_dim:
            raise ValueError('Invalid dimensional argument')

        bounds_ub = [np.inf] * n_arg_test
        bounds_lb = [-np.inf] * n_arg_test
        if (not compute_lb) and (not compute_ub):
            return bounds_lb, bounds_ub

        pool = mp.Pool()
        if verbose:
            # Using multiprocessing/map_async which is non-blocking for pretty
            # progress bar
            if compute_ub:
                # Compute the upper bounds
                bounds_ub_r_iter = pool.imap(self.get_single_upper_bound_pcon,
                                             arg_test_vec, chunksize)
                # Progress bar for the upper bound
                pbar_ub = tq.tqdm(total=n_arg_test,
                                  desc='Upper bound computation', smoothing=0)
                # Iterate over the imap iterator, update the pbar, append list
                bounds_ub = []
                for bound_ub_single_arg in bounds_ub_r_iter:
                    bounds_ub.append(bound_ub_single_arg)
                    pbar_ub.update()
                # Close the tqdm progress bar
                pbar_ub.close()
            if compute_lb:
                # Compute the lower bounds
                bounds_lb_r_iter = pool.imap(self.get_single_lower_bound_pcvx,
                                             arg_test_vec, chunksize)

                # Progress bar for the upper bound
                pbar_lb = tq.tqdm(total=n_arg_test,
                                  desc='Lower bound computation', smoothing=0)
                # Iterate over the imap iterator, update the pbar, append list
                bounds_lb = []
                for bound_lb_single_arg in bounds_lb_r_iter:
                    bounds_lb.append(bound_lb_single_arg)
                    pbar_lb.update()
                # Close the tqdm progress bar
                pbar_lb.close()
        else:
            # Using multiprocessing/map which is blocking
            if compute_ub:
                # Compute the upper bounds
                bounds_ub = pool.map(self.get_single_upper_bound_pcon,
                                     arg_test_vec, chunksize)
            if compute_lb:
                # Compute the lower bounds
                bounds_lb = pool.map(self.get_single_lower_bound_pcvx,
                                     arg_test_vec, chunksize)

        # Housekeeping for multiprocessing
        pool.close()
        pool.join()

        return np.array(bounds_lb), np.array(bounds_ub)

    def _cvxpy_probs_setup(self):
        """
        Formulate the upper and lower bound computations as CVXPY optimization
        problems.
        """
        arg_test = cp.Parameter((1, self.arg_dim), name='arg_test')

        (const_lb, const_ub, fun_test) = self._cvxpy_objective_and_constraints(
            self.arg, arg_test, self.fun, self.grad, self.grad_lips_constant,
            is_monotone_inc=self.is_monotone_inc,
            is_monotone_dec=self.is_monotone_dec, is_convex=self.is_convex,
            is_concave=self.is_concave)

        self.ub_prob = cp.Problem(cp.Maximize(fun_test), const_ub)
        self.lb_prob = cp.Problem(cp.Minimize(fun_test), const_lb)

    @staticmethod
    def _cvxpy_objective_and_constraints(arg, arg_test, fun, grad,
                                         grad_lips_constant,
                                         is_monotone_inc=False,
                                         is_monotone_dec=False, is_convex=False,
                                         is_concave=False):
        """
        Create the collection of linear constraints for the lower and upper
        bounds

        Sets up the optimization problem

        LB:  minimize     y_0
             subject to   y_i - g_0^\top (x_i - x_0)
                                           + L/2|x_i-x_0|^2 <= y_0
                                                |g_i - g_0| <= L |x_i-x_0|
                                    for each i\\in[1,..,N+1]

        UB:  maximize     y_0
             subject to   y_i - g_0^\top (x_i - x_0)
                                           - L/2|x_i-x_0|^2 <= y_0
                                                |g_i - g_0| <= L |x_i-x_0|
                                    for each i\\in[1,..,N+1]
        with decision variables y_0 and g_0
        """
        n_args = arg.shape[0]

        # Variables for the data points
        # Fun_test is the cvx variable denoting the possible values of the
        # interpolant at arg_test
        fun_test = cp.Variable((1,))
        grad_test = cp.Variable((1, 1), nonneg=is_monotone_inc,
                                nonpos=is_monotone_dec)

        # Constraints for existence
        rep_mat = np.ones((n_args, 1))
        # kron to repeat the elements and np.newaxis to retain the dimension
        # \nabla f(x_j) - \nabla f(x_0) for each j
        delta_grad = grad - cp.kron(rep_mat, grad_test)
        # x_j - x_0 for each j
        delta_arg = arg - cp.kron(rep_mat, arg_test)
        # L/2 ||x_j - x_0||^2 for each i and j
        delta_arg_norm = cp.Pnorm(delta_arg, p=2, axis=1)
        L_times_half_delta_arg_sqr = grad_lips_constant \
                                     * (delta_arg_norm ** 2)/2

        const_ub = [cp.abs(delta_grad) <= grad_lips_constant *cp.abs(delta_arg)]
        const_lb = [cp.abs(delta_grad) <= grad_lips_constant *cp.abs(delta_arg)]
        if is_convex:
            const_ub.append(fun >= fun_test + (delta_arg @ grad_test)[:, 0])
            const_lb.append(fun <= fun_test + (delta_arg @ grad_test)[:, 0]
                            + L_times_half_delta_arg_sqr)
        elif is_concave:
            const_ub.append(fun >= fun_test + (delta_arg @ grad_test)[:, 0]
                            - L_times_half_delta_arg_sqr)
            const_lb.append(fun <= fun_test + (delta_arg @ grad_test)[:, 0])
        else:
            const_ub.append(fun >= fun_test + (delta_arg@grad_test)[:, 0]
                            - L_times_half_delta_arg_sqr)
            const_lb.append(fun <= fun_test + (delta_arg @ grad_test)[:, 0]
                            + L_times_half_delta_arg_sqr)

        return const_lb, const_ub, fun_test

    def get_single_lower_bound_pcvx(self, arg_test_val):
        # Get the parameter
        list_of_params = self.lb_prob.parameters()
        arg_test_lb = list_of_params[0]

        try:
            # Parameter must be a row vector expressed as a numpy 2-D array
            arg_test_lb.value = np.array([arg_test_val])
        except:
            raise ValueError('This cryptic error typically arises when the '
                             'argument is weird. Expected a numpy array of '
                             'shape %s (got %s).\nThe argument provided is %s'
                             % (arg_test_lb.shape,
                                np.shape(np.array([arg_test_val])),
                                str(arg_test_val)))
        # Solve the optimization problem
        try:
            self.lb_prob.solve(**self.cvxpy_arg)
        except cp.SolverError as e:
            print(e)
            return np.inf
        return self.lb_prob.value

    def get_single_upper_bound_pcon(self, arg_test_val):
        # Get the parameter
        list_of_params = self.ub_prob.parameters()
        arg_test_ub = list_of_params[0]

        try:
            # Parameter must be a row vector expressed as a numpy 2-D array
            arg_test_ub.value = np.array([arg_test_val])
        except:
            raise ValueError('This cryptic error typically arises when the '
                             'argument is weird. Expected a numpy array of '
                             'shape %s (got %s).\nThe argument provided is %s'
                             % (arg_test_ub.shape,
                                np.shape(np.array([arg_test_val])),
                                str(arg_test_val)))
        # Solve the optimization problem
        try:
            self.ub_prob.solve(**self.cvxpy_arg)
        except cp.SolverError as e:
            print(e)
            return -np.inf
        return self.ub_prob.value
