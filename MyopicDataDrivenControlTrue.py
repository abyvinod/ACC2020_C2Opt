import numpy as np
import scipy.optimize as spo
import congol as cg
import time

class MyopicDataDrivenControlTrue(cg.SmoothMyopicDataDrivenControl):
    '''
    Use scipy.optimize for one-step optimal control
    '''

    def __init__(self, training_data, input_lb, input_ub, objective,  
            one_step_dyn=None, exit_condition=None, solver_style='L-BFGS-B'):
        """
        Based on 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html, 
        we use L-BFGS-B, the recommended approach for bound minimization

        self.trajectory excludes the current state
        """

        # Unpack training data
        self.trajectory = training_data['trajectory'][:-1, :]
        self.input_seq = training_data['input_seq']
        self.cost_val_vec = training_data['cost_val']

        # Bounds
        self.bounds = spo.Bounds(input_lb, input_ub)

        # Current state
        self.current_state = self.trajectory[-1:, :]

        # Constants
        self.context_arg_dim = 0          # Context is exactly the state, so not
                                          # returning it
        self.solver_style = solver_style
        self.marker_color = 'deepskyblue'
        self.marker_label = r'$\mathrm{Opt.\ traj.}$'
        self.marker_type = '^'
        self.zorder = 10
        self.marker_default_size = 20

        # Functions
        self.objective = objective
        self.exit_condition = exit_condition
        self.one_step_dyn = one_step_dyn

    def compute_decision_for_current_context(self, verbose=False):
        """
        1. Get the decision for the current context (current state)
        2. Use scipy.optimize to compute the best control action
        3. Return the decision
        """

        initial_action = self.bounds.lb
            # np.mean(np.vstack((self.bounds.lb, self.bounds.ub)), axis=0)
        if verbose:
            print('Current state: {:s} | Initial action guess: {:s}'.format(
                np.array_str(self.current_state, precision=2), 
                np.array_str(initial_action, precision=2)))

        # Compute the decision
        query_timer_start = time.time()
        res = spo.minimize(
                lambda u: self.objective(self.current_state[0, :], u), 
                initial_action, bounds=self.bounds, method=self.solver_style)
        current_decision = np.array([res.x])
        query_timer_end = time.time()
        query_time = query_timer_end - query_timer_start

        if verbose:
            # Update user
            print('Solver method: {:s}'.format(self.solver_style))
            print('Best action = {:s} | Attained cost = {:1.4f} | Time = '
                  '{:1.4f} s '.format(np.array_str(current_decision, 
                                                   precision=2),
                      res.fun[0], query_time))

        # Get the next state
        self.current_state = self.one_step_dyn(self.current_state, 
                                               current_decision)

        # Update the data matrices
        self.trajectory = np.vstack((self.trajectory, self.current_state))
        self.input_seq = np.vstack((self.input_seq, res.x))
        self.cost_val_vec = np.hstack((self.cost_val_vec, res.fun))

        # Return decision
        solution_dict = {'next_query': current_decision,
                         'lb_opt_val': res.fun,
                         'query_time': query_time}
        return solution_dict
