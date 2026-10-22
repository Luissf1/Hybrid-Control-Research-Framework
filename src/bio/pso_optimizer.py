"""
Optimización del controlador difuso mediante PSO (mealpy).

Vector de decisión (10 dimensiones):
    [err_NB, err_NS, err_ZE, err_PS, err_PB,
     der_NB, der_NS, der_ZE, der_PS, der_PB]
"""

import numpy as np
from mealpy import FloatVar, PSO

from src.fuzzy.fuzzy_controller import FuzzyController


class FuzzyPSOOptimizer:
    def __init__(
        self,
        num_particles: int = 30,
        max_iter: int = 50,
        lb_error: float = -4.0,
        ub_error: float = 4.0,
        lb_delta: float = -1.6,
        ub_delta: float = 1.6,
        seed: int = 42,
    ):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.lb_error = lb_error
        self.ub_error = ub_error
        self.lb_delta = lb_delta
        self.ub_delta = ub_delta
        self.seed = seed

        self.lb = np.array([lb_error] * 5 + [lb_delta] * 5)
        self.ub = np.array([ub_error] * 5 + [ub_delta] * 5)

        self.best_params = None
        self.best_mse = np.inf
        self.model = None

    # ------------------------------------------------------------------
    def objective_function(self, solution: np.ndarray) -> float:
        """
        Simula el lazo de control difuso y retorna el MSE.
        """
        fc = FuzzyController()
        fc.set_membership_params(np.array(solution))

        dt = 0.05
        referencia = 5.0
        estado = 0.0
        error_anterior = referencia - estado
        errores = []

        for _ in range(100):
            error = referencia - estado
            delta_error = error - error_anterior
            error_anterior = error

            accion = fc.compute(error, delta_error)
            estado = estado + accion * dt

            errores.append(error ** 2)

        return float(np.mean(errores))

    # ------------------------------------------------------------------
    def optimize(self) -> np.ndarray:
        """Ejecuta PSO y retorna los mejores parámetros."""
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

        self.best_params = np.array(self.model.solution, dtype=float)
        self.best_mse = float(self.model.fitness)

        return self.best_params