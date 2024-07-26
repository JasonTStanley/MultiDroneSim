from __future__ import division
import numpy as np
from scipy import signal
from control import base_controller
from cbf import CBF

from cvxopt import matrix, solvers
from utils.model_conversions import obs_to_lin_model
# solvers.options['solver'] = 'mosek'
solvers.options['show_progress'] = False
# solvers.options['feastol'] = 1e-9 # default:1e-7

class DroneQPTracker(object):
    def __init__(self, cbf, order=2, num_robots=1):
        self.cbf = cbf
        self.order = order
        self.num_robots = num_robots
        self.qp_tracker = QPTracker(cbf=cbf, order=order)

    def compute_control(self, obs, xdes, u_nominal, ignore_zmin=False, x_obs=None, obs_r_list=None):
        # Compute control using QPTracker
        x = np.zeros((self.num_robots, 9))
        for i in range(self.num_robots):
            x[i] = obs_to_lin_model(obs[i], dim=9)

        self.qp_tracker.cbf.set_xdes(xdes)
        success, u = self.qp_tracker.solve_fun(x, u_nominal, ignore_zmin=ignore_zmin, x_obs=x_obs, obs_r_list=obs_r_list)
        if success:
            return u
        else:
            print("QPTracker cannot find cbf-qp controller. Using nominal control.")
            return u_nominal

class QPTracker(object):
    """
    Modified QPTracker from erl_trackers to use only CBFs / LQR or custom control. no CLF support for now.

    Wrapper around CBF, LQR and cvxopt QP solver to accomplish QP-based low-level control for integrator systems.
        cbf=cbf, lqr=None Only ensure collision avoidance by minimally changing the nominal control provided by user
            ---> solve_fun(self, x, u, x_obs=None, obs_r_list=None)
        (not supported) cbf=cbf, lqr=lqr: Modify nominal control based on LQR solution for fixed boundary and fixed time min-energy problem.
            ---> solve_fun(self, x, xf, t, tf, x_obs=None, obs_r_list=None)

    TODO: make naming consistent, e.g., what is LgV?
    """

    def __init__(self, cbf, weighted_penalty=False, dim=3, order=3, clf_slack_weight=1.0):

        # TODO: modify the cost function to penalize deviation from ustar

        assert order>=1 and order <=4, "Currently just support 1-4th order integrator"
        # Initialize the system model
        self.dim = dim
        self.order = order

        # Other params
        self.clf_slack_weight = clf_slack_weight

        # Set appropriate method to compute control
        self.cbf = cbf
        self.weighted_penalty = weighted_penalty # whether turn CLF requirement into a weighted penalty for deviating from lqr control
        self.solve_fun = self._get_solve_fun(cbf, weighted_penalty=weighted_penalty)


    def _get_solve_fun(self, cbf, lqr=None, weighted_penalty=False):
        # Return the appropriate method to use given input arguments
        if (cbf is not None) and (lqr is None):
            return self._rectify  # ECBF-QP given user-provied nominal control

        elif (cbf is not None) and (lqr is not None): # and (clf is None):
            assert self.order==cbf.order==lqr.order, "Require that all qptracker, cbf, lqr to have the same order."
            assert self.dim==cbf.dim==lqr.dim==3, "Require that all qptracker, cbf, lqr to have same dimension of 3."

            if weighted_penalty:
                return self._get_weighted_safe_lqr_control  # Mission rate deviation is penalized via weighted norm

            return self._get_safe_lqr_control  # No mission rate deviation

        else:
            # TODO: do so cleanly
            assert False, "No solver can be found given cbf={}, lqr={}, clf={}".format(cbf, lqr)


    def _rectify(self, x, uhat, ignore_zmin=False, x_obs=None, obs_r_list=None, weights=None):
        # Objective (no slack in this case)
        N = len(uhat)

        if weights is None:
            P = np.eye(4*N)
            q = -np.hstack(uhat).reshape(4*N,)
        else:
            P = np.eye(4*N)
            for i, W in enumerate(weights):
                P[i*4:(i+1)*4, i*4:(i+1)*4] = W

            uhat_weighted = [-uh.dot(W) for uh, W in zip(uhat, weights)]
            q = np.hstack(uhat_weighted).reshape((-1,))

        # CBF constraint (inter-agent collision avoidance, umax, box constraints, and optional obstacle avoidance)
        G, h = self.cbf._build_ineq_const(x, ignore_zmin, x_obs, obs_r_list)
        u = None
        success = False
        try:
            sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))  # without equality constraint
            u = np.asarray(sol['x'])
            u = np.asarray([u[i*4:(i+1)*4] for i in range(N)]).squeeze()
            success = True

        except Exception as e:
            print("qptracker cannot find cbf-qp controller. Error: {}".format(str(e)))

        return success, u

    def _get_safe_lqr_control(self, x, xf, t, tf, ignore_zmin=False, x_obs=None, obs_r_list=None):
        # Compute nominal lqr control (with replanning) for each agent
        uhat = [self.lqr.get_control(xi, xfi, t, tf) for xi, xfi in zip(x, xf)]
        # invoke _rectify
        return self._rectify(x, uhat, ignore_zmin, x_obs, obs_r_list)

    def _get_weighted_safe_lqr_control(self, x, xf, t, tf, ignore_zmin=False, x_obs=None, obs_r_list=None):
        # Compute nominal lqr control (with replanning) for each agent
        uhat = [self.lqr.get_control(xi, xfi, t, tf) for xi, xfi in zip(x, xf)]

        # Change the penalty weight to reflect CLF constraint
        LgV_list = [-uh.reshape((1, self.dim)) for uh in uhat] # LgV happens to be -u_lqr
        weights = [ np.eye(self.dim) +
                    self.clf_slack_weight * (LgV.T.dot(LgV)) / (LgV.dot(LgV.T).item())
                    for LgV in LgV_list] # (I+lgV.T@LgV)

        # invoke _rectify
        return self._rectify(x, uhat, ignore_zmin, x_obs, obs_r_list, weights=weights)


    def rescale_Gh(self, G, h):
        # scale = G[0, :-1].dot(G[0, :-1]) # || dV/dx * B ||^2
        scale = np.sqrt(G[0, :-1].dot(G[0, :-1])) # || dV/dx * B ||
        new_G = G.copy()
        new_h = h.copy()
        new_G[0, :-1] = new_G[0, :-1]/(scale+1e-6) # Do not rescale weight for slack
        new_h /= (scale+1e-6)
        return new_G ,new_h