"""
Controlador Difuso Mamdani para estabilización del quadrotor.

Entradas:
    error       ∈ [-5, 5]
    delta_error ∈ [-2, 2]
Salida:
    control_action ∈ [-2, 2]

5 conjuntos triangulares (NB, NS, ZE, PS, PB) y 5 reglas diagonales.
set_membership_params permite al PSO modificar los centros de las funciones.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyController:
    LABELS = ["NB", "NS", "ZE", "PS", "PB"]

    def __init__(self, resolution: int = 101):
        self.resolution = resolution

        self.universe_error = np.linspace(-5.0, 5.0, resolution)
        self.universe_delta = np.linspace(-2.0, 2.0, resolution)
        self.universe_control = np.linspace(-2.0, 2.0, resolution)

        self.error = ctrl.Antecedent(self.universe_error, "error")
        self.delta_error = ctrl.Antecedent(self.universe_delta, "delta_error")
        self.control_action = ctrl.Consequent(self.universe_control, "control_action")

        # Parámetros por defecto [a, b, c]
        self.default_params_error = np.array([
            [-5.0, -4.0, -2.5],
            [-3.5, -2.0, -0.5],
            [-1.0,  0.0,  1.0],
            [ 0.5,  2.0,  3.5],
            [ 2.5,  4.0,  5.0],
        ])
        self.default_params_delta = np.array([
            [-2.0, -1.6, -1.0],
            [-1.4, -0.8, -0.2],
            [-0.4,  0.0,  0.4],
            [ 0.2,  0.8,  1.4],
            [ 1.0,  1.6,  2.0],
        ])
        self.default_params_control = np.array([
            [-2.0, -1.6, -1.0],
            [-1.4, -0.8, -0.2],
            [-0.4,  0.0,  0.4],
            [ 0.2,  0.8,  1.4],
            [ 1.0,  1.6,  2.0],
        ])

        self._build_membership_functions(
            self.default_params_error,
            self.default_params_delta,
        )
        self._build_rules()
        self._build_simulation()

    # ------------------------------------------------------------------
    def _build_membership_functions(self, params_error, params_delta):
        for i, label in enumerate(self.LABELS):
            self.error[label] = fuzz.trimf(self.universe_error, params_error[i])
            self.delta_error[label] = fuzz.trimf(self.universe_delta, params_delta[i])
            self.control_action[label] = fuzz.trimf(
                self.universe_control, self.default_params_control[i]
            )

    def _build_rules(self):
        self.rules = [
            ctrl.Rule(self.error["NB"] & self.delta_error["NB"], self.control_action["NB"]),
            ctrl.Rule(self.error["NS"] & self.delta_error["NS"], self.control_action["NS"]),
            ctrl.Rule(self.error["ZE"] & self.delta_error["ZE"], self.control_action["ZE"]),
            ctrl.Rule(self.error["PS"] & self.delta_error["PS"], self.control_action["PS"]),
            ctrl.Rule(self.error["PB"] & self.delta_error["PB"], self.control_action["PB"]),
        ]

    def _build_simulation(self):
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    # ------------------------------------------------------------------
    def set_membership_params(self, params_array: np.ndarray) -> None:
        """
        Actualiza centros de las funciones de membresía triangulares.

        Args:
            params_array: Vector de 10 elementos
                [err_NB, err_NS, err_ZE, err_PS, err_PB,
                 der_NB, der_NS, der_ZE, der_PS, der_PB]
        """
        centers_error = params_array[:5]
        centers_delta = params_array[5:10]

        params_error = np.zeros((5, 3))
        params_delta = np.zeros((5, 3))

        for i in range(5):
            c_err = centers_error[i]
            a_err = max(-5.0, c_err - 1.5)
            b_err = min(5.0, c_err + 1.5)
            params_error[i] = [a_err, c_err, b_err]

            c_del = centers_delta[i]
            a_del = max(-2.0, c_del - 0.6)
            b_del = min(2.0, c_del + 0.6)
            params_delta[i] = [a_del, c_del, b_del]

        self._build_membership_functions(params_error, params_delta)
        self._build_rules()
        self._build_simulation()

    # ------------------------------------------------------------------
    def compute(self, error: float, delta_error: float) -> float:
        """Evalúa el sistema difuso y retorna la acción defusificada."""
        error_clipped = float(np.clip(error, -5.0, 5.0))
        delta_clipped = float(np.clip(delta_error, -2.0, 2.0))

        self.simulation.input["error"] = error_clipped
        self.simulation.input["delta_error"] = delta_clipped

        try:
            self.simulation.compute()
            return float(self.simulation.output["control_action"])
        except Exception:
            return 0.0