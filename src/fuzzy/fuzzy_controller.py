"""
Mamdani Fuzzy Controller
=========================

Two-input, one-output fuzzy inference system:

    error          ∈ [-5, 5]   (altitude error)
    delta_error    ∈ [-2, 2]   (change of error)
    control_action ∈ [-2, 2]   (corrective signal)

Each variable is described by 5 triangular membership functions:

    NB = Negative Big
    NS = Negative Small
    ZE = Zero
    PS = Positive Small
    PB = Positive Big

Rule base (5 diagonal rules):

    IF error IS NB AND delta IS NB THEN action IS NB
    IF error IS NS AND delta IS NS THEN action IS NS
    IF error IS ZE AND delta IS ZE THEN action IS ZE
    IF error IS PS AND delta IS PS THEN action IS PS
    IF error IS PB AND delta IS PB THEN action IS PB

This configuration is functionally equivalent to a fuzzy PD controller
and can be optimized by PSO via `set_membership_params()`.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyController:
    """
    Mamdani fuzzy controller with tunable membership parameters
    (designed to be optimized by PSO).
    """

    LABELS = ["NB", "NS", "ZE", "PS", "PB"]

    def __init__(self, resolution: int = 101) -> None:
        """
        Parameters
        ----------
        resolution : int
            Number of points in each universe of discourse (higher =
            more precise defuzzification, at a higher computational cost).
        """
        self.resolution = resolution

        # --- Universes of discourse ---
        self.universe_error = np.linspace(-5.0, 5.0, resolution)
        self.universe_delta = np.linspace(-2.0, 2.0, resolution)
        self.universe_control = np.linspace(-2.0, 2.0, resolution)

        # --- Fuzzy variables ---
        self.error = ctrl.Antecedent(self.universe_error, "error")
        self.delta_error = ctrl.Antecedent(self.universe_delta, "delta_error")
        self.control_action = ctrl.Consequent(self.universe_control, "control_action")

        # --- Default triangular parameters [a, b, c] for each label ---
        self.default_params_error = np.array([
            [-5.0, -4.0, -2.5],   # NB
            [-3.5, -2.0, -0.5],   # NS
            [-1.0,  0.0,  1.0],   # ZE
            [ 0.5,  2.0,  3.5],   # PS
            [ 2.5,  4.0,  5.0],   # PB
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

        # Build the initial inference system
        self._build_membership_functions(
            self.default_params_error,
            self.default_params_delta,
        )
        self._build_rules()
        self._build_simulation()

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _build_membership_functions(self, params_error, params_delta) -> None:
        """Assign triangular membership functions to each linguistic label."""
        for i, label in enumerate(self.LABELS):
            self.error[label] = fuzz.trimf(self.universe_error, params_error[i])
            self.delta_error[label] = fuzz.trimf(self.universe_delta, params_delta[i])
            self.control_action[label] = fuzz.trimf(
                self.universe_control, self.default_params_control[i]
            )

    def _build_rules(self) -> None:
        """Generate the 5 diagonal rules described in the module docstring."""
        self.rules = [
            ctrl.Rule(self.error["NB"] & self.delta_error["NB"], self.control_action["NB"]),
            ctrl.Rule(self.error["NS"] & self.delta_error["NS"], self.control_action["NS"]),
            ctrl.Rule(self.error["ZE"] & self.delta_error["ZE"], self.control_action["ZE"]),
            ctrl.Rule(self.error["PS"] & self.delta_error["PS"], self.control_action["PS"]),
            ctrl.Rule(self.error["PB"] & self.delta_error["PB"], self.control_action["PB"]),
        ]

    def _build_simulation(self) -> None:
        """Create the fuzzy control system and its simulator."""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_membership_params(self, params_array: np.ndarray) -> None:
        """
        Update the centers of the triangular membership functions
        (invoked by the PSO optimizer).

        Parameters
        ----------
        params_array : np.ndarray
            Length-10 vector:
            [err_NB, err_NS, err_ZE, err_PS, err_PB,
             der_NB, der_NS, der_ZE, der_PS, der_PB]
        """
        centers_error = params_array[:5]
        centers_delta = params_array[5:10]

        params_error = np.zeros((5, 3))
        params_delta = np.zeros((5, 3))

        for i in range(5):
            # Error centers with a half-base of ±1.5
            c_err = centers_error[i]
            a_err = max(-5.0, c_err - 1.5)
            b_err = min(5.0, c_err + 1.5)
            params_error[i] = [a_err, c_err, b_err]

            # Delta centers with a half-base of ±0.6
            c_del = centers_delta[i]
            a_del = max(-2.0, c_del - 0.6)
            b_del = min(2.0, c_del + 0.6)
            params_delta[i] = [a_del, c_del, b_del]

        # Rebuild the whole inference system with the new parameters
        self._build_membership_functions(params_error, params_delta)
        self._build_rules()
        self._build_simulation()

    def compute(self, error: float, delta_error: float) -> float:
        """
        Evaluate the fuzzy inference system and return the defuzzified
        control action (centroid method).

        Parameters
        ----------
        error : float
            Current error (clipped to [-5, 5]).
        delta_error : float
            Change of error (clipped to [-2, 2]).

        Returns
        -------
        float
            Defuzzified control action.
        """
        # Clip inputs to their universes of discourse
        error_clipped = float(np.clip(error, -5.0, 5.0))
        delta_clipped = float(np.clip(delta_error, -2.0, 2.0))

        # Feed the inputs into the simulator
        self.simulation.input["error"] = error_clipped
        self.simulation.input["delta_error"] = delta_clipped

        try:
            self.simulation.compute()
            return float(self.simulation.output["control_action"])
        except Exception:
            # Safe fallback: if no rule fires (e.g., error and delta in
            # opposite quadrants with only diagonal rules), return neutral 0.0
            return 0.0