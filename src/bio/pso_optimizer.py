"""
FuzzyPSOOptimizer — Membership-Center Tuning with PSO
======================================================

Uses `mealpy.OriginalPSO` to minimize the Mean Squared Error (MSE) of a
simplified 100-step closed-loop simulation driven by the fuzzy controller.

Decision vector (10 dimensions):

    [err_NB, err_NS, err_ZE, err_PS, err_PB,
     der_NB, der_NS, der_ZE, der_PS, der_PB]

The first 5 values are the centers of the `error` membership functions
(universe [-5, 5]); the last 5 correspond to `delta_error`
(universe [-2, 2]).
"""

import numpy as np
from mealpy import FloatVar, PSO

from src.fuzzy.fuzzy_controller import FuzzyController


class FuzzyPSOOptimizer:
    """
    PSO-based optimizer for the FuzzyController membership parameters.
    """

    def __init__(
        self,
        num_particles: int = 30,
        max_iter: int = 50,
        lb_error: float = -4.0,
        ub_error: float = 4.0,
        lb_delta: float = -1.6,
        ub_delta: float = 1.6,
        seed: int = 42,
    ) -> None:
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.seed = seed

        # Concatenate bounds: 5 error dims + 5 delta dims
        self.lb = np.array([lb_error] * 5 + [lb_delta] * 5)
        self.ub = np.array([ub_error] * 5 + [ub_delta] * 5)

        self.best_params: np.ndarray | None = None
        self.best_mse: float = float("inf")
        self.model = None

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------
    def objective_function(self, solution: np.ndarray) -> float:
        """
        Simulate a 100-step closed-loop fuzzy controller and return the MSE.

        Simulation
        ----------
        reference   = 5.0 (constant)
        state(0)    = 0.0
        error(t)    = reference - state(t)
        delta_err   = error(t) - error(t-1)
        action(t)   = fuzzy.compute(error, delta_err)
        state(t+1)  = state(t) + action(t) * dt   (simple integrator plant)

        Parameters
        ----------
        solution : np.ndarray
            Length-10 vector of membership centers.

        Returns
        -------
        float
            Mean squared error over the 100-step horizon.
        """
        fc = FuzzyController()
        fc.set_membership_params(np.array(solution))

        dt = 0.05
        reference = 5.0
        state = 0.0
        previous_error = reference - state
        squared_errors = []

        for _ in range(100):
            error = reference - state
            delta_error = error - previous_error
            previous_error = error

            action = fc.compute(error, delta_error)

            # Simple integrator plant
            state = state + action * dt

            squared_errors.append(error ** 2)

        return float(np.mean(squared_errors))

    # ------------------------------------------------------------------
    # Optimization entry point
    # ------------------------------------------------------------------
    def optimize(self) -> np.ndarray:
        """
        Run PSO and return the best membership-center vector found.

        Returns
        -------
        np.ndarray
            Length-10 vector of optimized centers.
        """
        problem = {
            "obj_func": self.objective_function,
            "bounds": FloatVar(lb=self.lb, ub=self.ub),
            "minmax": "min",
        }

        self.model = PSO.OriginalPSO(
            epoch=self.max_iter,
            pop_size=self.num_particles,
            seed=self.seed,
        )
        self.model.solve(problem=problem)

        self.best_params = np.array(self.model.g_best.solution, dtype=float)
        self.best_mse = float(self.model.g_best.target.fitness)

        return self.best_params