"""
QuadrotorEnv — Custom Gymnasium Environment
============================================

Simplified physics of a quadrotor constrained to vertical motion (1D),
with a dynamic tilt coupling used as a perturbation.

Discrete-time Euler integration:

    v_{t+1}     = v_t + (a_cmd - perturbation) * dt
    z_{t+1}     = z_t + v_t * dt
    tilt_{t+1}  = tilt_t + k * v_t * dt

where:
    a_cmd        is the vertical acceleration commanded by the discrete action
    perturbation is a time-varying effective gravity (sinusoidal disturbance)
    k            is the velocity-to-tilt coupling coefficient

Spaces
------
Observation : MultiDiscrete([10, 10, 10]) -> (altitude, velocity, tilt) bins
Action      : Discrete(5) -> 5 vertical acceleration commands

Reward
------
    r = -|z - z_target|
    additional -100 penalty if |tilt| > tilt_limit (terminates the episode)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class QuadrotorEnv(gym.Env):
    """
    Simplified 1D quadrotor with Euler-integrated dynamics.

    Parameters
    ----------
    max_steps : int
        Maximum steps per episode (truncation).
    target_altitude : float
        Reference altitude (m).
    tilt_limit : float
        Tilt angle (degrees) that terminates the episode.
    dt : float
        Integration step (s).
    seed : int, optional
        Seed for the underlying NumPy RNG.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        max_steps: int = 200,
        target_altitude: float = 5.0,
        tilt_limit: float = 30.0,
        dt: float = 0.05,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        # --- Simulation parameters ---
        self.max_steps = max_steps
        self.target_altitude = target_altitude
        self.tilt_limit = tilt_limit
        self.dt = dt

        # --- Physical bounds of the continuous state ---
        self.alt_min, self.alt_max = 0.0, 10.0
        self.vel_min, self.vel_max = -3.0, 3.0
        self.tilt_min, self.tilt_max = -45.0, 45.0

        # --- Discretization: 10 bins per variable ---
        self.n_bins = 10
        self.alt_bins = np.linspace(self.alt_min, self.alt_max, self.n_bins + 1)
        self.vel_bins = np.linspace(self.vel_min, self.vel_max, self.n_bins + 1)
        self.tilt_bins = np.linspace(self.tilt_min, self.tilt_max, self.n_bins + 1)

        # --- Gymnasium spaces ---
        self.observation_space = spaces.MultiDiscrete(
            [self.n_bins, self.n_bins, self.n_bins]
        )
        self.action_space = spaces.Discrete(5)

        # --- Action -> commanded vertical acceleration (m/s²) ---
        self.action_to_accel = {
            0: +1.0,   # climb fast
            1: +0.5,   # climb slowly
            2:  0.0,   # hold altitude
            3: -0.5,   # descend slowly
            4: -1.0,   # descend fast
        }

        # --- Internal continuous state ---
        self.altitude = 0.0
        self.velocity = 0.0
        self.tilt = 0.0
        self.current_step = 0

        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def _get_discrete_state(self) -> tuple[int, int, int]:
        """
        Convert the continuous state into discrete bin indices.

        np.digitize returns indices in [1, len(bins)-1]; we subtract 1 to
        obtain zero-based indices and clip to the valid range.
        """
        alt_idx = int(np.digitize(self.altitude, self.alt_bins) - 1)
        vel_idx = int(np.digitize(self.velocity, self.vel_bins) - 1)
        tilt_idx = int(np.digitize(self.tilt, self.tilt_bins) - 1)

        alt_idx = int(np.clip(alt_idx, 0, self.n_bins - 1))
        vel_idx = int(np.clip(vel_idx, 0, self.n_bins - 1))
        tilt_idx = int(np.clip(tilt_idx, 0, self.n_bins - 1))

        return (alt_idx, vel_idx, tilt_idx)

    def _get_continuous_state(self) -> np.ndarray:
        """Return the raw continuous state (useful for logging/plotting)."""
        return np.array(
            [self.altitude, self.velocity, self.tilt], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment to a random initial state near the ground.

        Returns
        -------
        (obs, info) : tuple
            Discrete observation and an (empty) info dictionary.
        """
        super().reset(seed=seed)

        self.altitude = float(np.random.uniform(0.0, 1.0))
        self.velocity = float(np.random.uniform(-0.2, 0.2))
        self.tilt = float(np.random.uniform(-2.0, 2.0))
        self.current_step = 0

        return self._get_discrete_state(), {}

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Advance the simulation by one Euler step.

        Returns
        -------
        next_state : tuple
            Discrete observation after the action.
        reward : float
            Negative absolute altitude error, plus a strong penalty if
            the tilt limit is exceeded.
        terminated : bool
            True if the episode ended due to excessive tilt.
        truncated : bool
            True if max_steps was reached.
        info : dict
            Auxiliary diagnostics (continuous state, instantaneous error).
        """
        self.current_step += 1

        # Acceleration commanded by the discrete action
        a_cmd = self.action_to_accel[action]

        # Time-varying disturbance: effective gravity oscillates sinusoidally
        perturbation = 0.1 * np.sin(0.3 * self.current_step)

        # --- Euler integration ---
        self.velocity += (a_cmd - perturbation) * self.dt
        self.altitude += self.velocity * self.dt

        # Dynamic coupling: tilt responds to vertical velocity
        self.tilt += 2.0 * self.velocity * self.dt

        # Clipping to physical bounds
        self.altitude = float(np.clip(self.altitude, self.alt_min, self.alt_max))
        self.velocity = float(np.clip(self.velocity, self.vel_min, self.vel_max))
        self.tilt = float(np.clip(self.tilt, self.tilt_min, self.tilt_max))

        # --- Reward ---
        error = abs(self.altitude - self.target_altitude)
        reward = -error

        # Strong penalty and termination if tilt exceeds the safety limit
        terminated = False
        if abs(self.tilt) > self.tilt_limit:
            reward -= 100.0
            terminated = True

        truncated = self.current_step >= self.max_steps

        info = {
            "altitude": self.altitude,
            "velocity": self.velocity,
            "tilt": self.tilt,
            "error": error,
        }

        return self._get_discrete_state(), float(reward), terminated, truncated, info