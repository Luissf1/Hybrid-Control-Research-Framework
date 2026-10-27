"""
QuadrotorEnv — Custom Gymnasium Environment
============================================

Simplified physics of a quadrotor constrained to vertical motion (1D),
with a dynamic tilt coupling used as a perturbation.

Reward shaping
--------------
The reward gives a strong positive bonus for staying inside a narrow
tolerance band around the set-point, a mild bonus for being close, and
a distance penalty for being far. Without this shaping, a pure -|error|
reward gives the agent no incentive to *hold* at the set-point: at
z=0 and z=10 the error is the same (5 m), so the agent is indifferent
between the two extremes and never learns to track the center.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class QuadrotorEnv(gym.Env):
    """Simplified 1D quadrotor with Euler-integrated dynamics."""

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

        self.max_steps = max_steps
        self.target_altitude = target_altitude
        self.tilt_limit = tilt_limit
        self.dt = dt

        self.alt_min, self.alt_max = 0.0, 10.0
        self.vel_min, self.vel_max = -3.0, 3.0
        self.tilt_min, self.tilt_max = -45.0, 45.0

        self.n_bins = 10
        self.alt_bins = np.linspace(self.alt_min, self.alt_max, self.n_bins + 1)
        self.vel_bins = np.linspace(self.vel_min, self.vel_max, self.n_bins + 1)
        self.tilt_bins = np.linspace(self.tilt_min, self.tilt_max, self.n_bins + 1)

        self.observation_space = spaces.MultiDiscrete(
            [self.n_bins, self.n_bins, self.n_bins]
        )
        self.action_space = spaces.Discrete(5)

        self.action_to_accel = {
            0: +1.0,   # climb fast
            1: +0.5,   # climb slowly
            2:  0.0,   # hold altitude
            3: -0.5,   # descend slowly
            4: -1.0,   # descend fast
        }

        self.altitude = 0.0
        self.velocity = 0.0
        self.tilt = 0.0
        self.current_step = 0

        if seed is not None:
            np.random.seed(seed)

    def _get_discrete_state(self) -> tuple[int, int, int]:
        alt_idx = int(np.digitize(self.altitude, self.alt_bins) - 1)
        vel_idx = int(np.digitize(self.velocity, self.vel_bins) - 1)
        tilt_idx = int(np.digitize(self.tilt, self.tilt_bins) - 1)

        alt_idx = int(np.clip(alt_idx, 0, self.n_bins - 1))
        vel_idx = int(np.clip(vel_idx, 0, self.n_bins - 1))
        tilt_idx = int(np.clip(tilt_idx, 0, self.n_bins - 1))

        return (alt_idx, vel_idx, tilt_idx)

    def _get_continuous_state(self) -> np.ndarray:
        return np.array(
            [self.altitude, self.velocity, self.tilt], dtype=np.float32
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.altitude = float(np.random.uniform(0.0, 1.0))
        self.velocity = float(np.random.uniform(-0.2, 0.2))
        self.tilt = float(np.random.uniform(-2.0, 2.0))
        self.current_step = 0

        return self._get_discrete_state(), {}

    def step(self, action: int):
        self.current_step += 1

        a_cmd = self.action_to_accel[action]
        perturbation = 0.1 * np.sin(0.3 * self.current_step)

        # Euler integration
        self.velocity += (a_cmd - perturbation) * self.dt
        self.altitude += self.velocity * self.dt
        self.tilt += 2.0 * self.velocity * self.dt

        # Clipping
        self.altitude = float(np.clip(self.altitude, self.alt_min, self.alt_max))
        self.velocity = float(np.clip(self.velocity, self.vel_min, self.vel_max))
        self.tilt = float(np.clip(self.tilt, self.tilt_min, self.tilt_max))

        # --- REWARD SHAPING ---
        error = abs(self.altitude - self.target_altitude)

        if error < 0.5:
            reward = 10.0
        elif error < 1.0:
            reward = 2.0 - error
        else:
            reward = -error

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