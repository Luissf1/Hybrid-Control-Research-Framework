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

Design notes
------------
1. Full 5x5 rule grid (25 rules), not just the diagonal. The diagonal
   alone leaves large regions of the (error, delta_error) plane with
   no active rule, causing the controller to fall back to 0.0 — which
   is exactly what we do NOT want when the system is far from the
   set-point.

2. "Shoulder" membership functions at the universe boundaries: the
   NB and PB triangles reach full membership at x = ±universe_limit
   rather than tapering to zero there. This guarantees that, for ANY
   input in the universe, at least one membership function has a
   non-zero degree, so skfuzzy can always activate a rule.

3. The PSO tunes only the ANTECEDENT centers (error and delta_error).
   The consequent (control_action) is kept fixed with sensible
   defaults. This matches the decision vector of 10 dimensions.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyController:
    """
    Mamdani fuzzy controller with tunable membership parameters.

    Designed to be optimized by the PSO module: `set_membership_params`
    updates the centers of the triangular membership functions, and the
    inference system is rebuilt automatically.
    """

    LABELS = ["NB", "NS", "ZE", "PS", "PB"]

    # Rule table (5x5): rows = error, cols = delta_error.
    #                delta:  NB    NS    ZE    PS    PB
    RULE_TABLE = [
        ["NB", "NB", "NB", "NS", "ZE"],   # error NB
        ["NB", "NB", "NS", "ZE", "PS"],   # error NS
        ["NB", "NS", "ZE", "PS", "PB"],   # error ZE
        ["NS", "ZE", "PS", "PB", "PB"],   # error PS
        ["ZE", "PS", "PB", "PB", "PB"],   # error PB
    ]

    def __init__(self, resolution: int = 101) -> None:
        """
        Parameters
        ----------
        resolution : int
            Number of points in each universe of discourse.
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
        # Shoulder functions (NB and PB) reach full membership at the
        # universe edges to guarantee coverage for every input.
        # For 'error' (universe [-5, 5]):
        self.default_params_error = np.array([
            [-5.0, -5.0, -2.5],   # NB: shoulder at -5
            [-3.5, -2.0, -0.5],   # NS
            [-1.0,  0.0,  1.0],   # ZE
            [ 0.5,  2.0,  3.5],   # PS
            [ 2.5,  5.0,  5.0],   # PB: shoulder at +5
        ])
        # For 'delta_error' (universe [-2, 2]):
        self.default_params_delta = np.array([
            [-2.0, -2.0, -1.0],   # NB: shoulder at -2
            [-1.4, -0.8, -0.2],   # NS
            [-0.4,  0.0,  0.4],   # ZE
            [ 0.2,  0.8,  1.4],   # PS
            [ 1.0,  2.0,  2.0],   # PB: shoulder at +2
        ])
        # For 'control_action' (universe [-2, 2]):
        self.default_params_control = np.array([
            [-2.0, -2.0, -1.0],   # NB: shoulder at -2
            [-1.4, -0.8, -0.2],   # NS
            [-0.4,  0.0,  0.4],   # ZE
            [ 0.2,  0.8,  1.4],   # PS
            [ 1.0,  2.0,  2.0],   # PB: shoulder at +2
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
        """
        Build the full 5x5 rule grid (25 rules).

        The full grid guarantees that every (error, delta_error) pair
        activates at least one rule, which is essential for the hybrid
        controller operating at large errors.
        """
        self.rules = []
        for i, err_label in enumerate(self.LABELS):
            for j, der_label in enumerate(self.LABELS):
                out_label = self.RULE_TABLE[i][j]
                self.rules.append(
                    ctrl.Rule(
                        self.error[err_label] & self.delta_error[der_label],
                        self.control_action[out_label],
                    )
                )

    def _build_simulation(self) -> None:
        """Create the fuzzy control system and its simulator."""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_membership_params(self, params_array: np.ndarray) -> None:
        """
        Update the centers of the triangular membership functions.

        The outermost triangles (NB and PB) are built as shoulder
        functions that saturate at the universe boundary, so the edges
        of the input space remain covered after PSO tuning.
        """
        centers_error = params_array[:5]
        centers_delta = params_array[5:10]

        params_error = np.zeros((5, 3))
        params_delta = np.zeros((5, 3))

        for i in range(5):
            # --- error membership functions ---
            c_err = centers_error[i]
            if i == 0:
                # NB: shoulder at -5
                params_error[i] = [-5.0, -5.0, min(5.0, c_err + 1.5)]
            elif i == 4:
                # PB: shoulder at +5
                params_error[i] = [max(-5.0, c_err - 1.5), 5.0, 5.0]
            else:
                params_error[i] = [
                    max(-5.0, c_err - 1.5),
                    c_err,
                    min(5.0, c_err + 1.5),
                ]

            # --- delta_error membership functions ---
            c_del = centers_delta[i]
            if i == 0:
                # NB: shoulder at -2
                params_delta[i] = [-2.0, -2.0, min(2.0, c_del + 0.6)]
            elif i == 4:
                # PB: shoulder at +2
                params_delta[i] = [max(-2.0, c_del - 0.6), 2.0, 2.0]
            else:
                params_delta[i] = [
                    max(-2.0, c_del - 0.6),
                    c_del,
                    min(2.0, c_del + 0.6),
                ]

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
            Defuzzified control action in [-2, 2].
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
            # With full rule coverage and shoulder MFs this should never
            # trigger, but we keep it as a defensive measure.
            return 0.0