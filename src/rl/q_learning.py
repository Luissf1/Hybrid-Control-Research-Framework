"""
Classical Tabular Q-Learning Agent
===================================

Implements the off-policy Q-Learning update rule over a dictionary-based
Q-table of the form:

    q_table[state][action] = Q(state, action)

Bellman update equation:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

where:
    alpha  is the learning rate      (how strongly new information
                                      overwrites the old estimate)
    gamma  is the discount factor    (how much future rewards matter)
    max_a' Q(s',a') is omitted when the episode terminates (`done=True`),
    because there is no future to bootstrap from.

No function approximation, no neural networks: this is purely tabular
Q-Learning, which keeps the representation interpretable and suitable for
small discrete state spaces (as required by the doctoral thesis).
"""

from collections import defaultdict
import random

import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning agent for discrete state and action spaces.

    Parameters
    ----------
    alpha : float
        Learning rate in (0, 1].
    gamma : float
        Discount factor in [0, 1].
    epsilon : float
        Initial exploration probability.
    epsilon_decay : float
        Multiplicative decay applied to epsilon after each episode.
    epsilon_min : float
        Lower bound for epsilon.
    seed : int, optional
        Seed for reproducibility (affects random and numpy.random).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # `defaultdict(dict)` automatically creates an empty dict when a new
        # state is accessed. Individual action values are initialized later
        # according to the size of the action space.
        self.q_table: dict = defaultdict(dict)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def choose_action(
        self,
        state,
        action_space_size: int,
        explore: bool = True,
    ) -> int:
        """
        Epsilon-greedy action selection.

        With probability `epsilon`, pick a uniformly random action
        (exploration). Otherwise, pick the action with the highest Q-value
        for the given state (exploitation), breaking ties randomly.

        Parameters
        ----------
        state : hashable
            Discrete state (integer, tuple, etc.).
        action_space_size : int
            Total number of available actions.
        explore : bool, default True
            If False, exploration is disabled (useful for evaluation).

        Returns
        -------
        int
            Selected action index.
        """
        # Initialize Q-values for a previously unseen state
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(action_space_size)}

        # --- Exploration ---
        if explore and random.random() < self.epsilon:
            return random.randint(0, action_space_size - 1)

        # --- Exploitation ---
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    # ------------------------------------------------------------------
    # Q-table update
    # ------------------------------------------------------------------
    def update_q_table(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        action_space_size: int | None = None,
    ) -> None:
        """
        Apply the Bellman update to Q(state, action).

        Parameters
        ----------
        state, action : current state and action taken
        reward : scalar reward received
        next_state : resulting state
        done : True if the episode terminated (no bootstrap)
        action_space_size : used to initialize unseen states; if None,
            defaults to 5 (matches QuadrotorEnv's action space)
        """
        # Ensure both states exist in the Q-table
        if state not in self.q_table:
            size = action_space_size if action_space_size else 5
            self.q_table[state] = {a: 0.0 for a in range(size)}
        if next_state not in self.q_table:
            size = action_space_size if action_space_size else 5
            self.q_table[next_state] = {a: 0.0 for a in range(size)}

        current_q = self.q_table[state][action]

        # --- Compute the Bellman target ---
        if done:
            # Terminal state: no future reward to estimate
            target = reward
        else:
            max_next_q = max(self.q_table[next_state].values())
            target = reward + self.gamma * max_next_q

        # --- Incremental update ---
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """Reduce epsilon after each episode, bounded below by epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)