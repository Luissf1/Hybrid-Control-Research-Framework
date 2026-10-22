"""
Agente Q-Learning Tabular Clásico.

Implementación off-policy sin aproximación de funciones. La tabla Q se
almacena en un defaultdict anidado: q_table[estado][accion] = valor_Q.

Ecuación de Bellman (actualización Q-Learning):
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
"""

from collections import defaultdict
import random
import numpy as np


class QLearningAgent:
    """
    Agente Q-Learning para espacios de estados y acciones discretos.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Tabla Q: estado -> {accion: valor}
        self.q_table = defaultdict(dict)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Selección de acción: epsilon-greedy
    # ------------------------------------------------------------------
    def choose_action(self, state, action_space_size: int, explore: bool = True) -> int:
        """
        Selecciona una acción usando la política epsilon-greedy.

        Args:
            state: Estado discreto (tuple o int).
            action_space_size: Número de acciones disponibles.
            explore: Si es False, desactiva la exploración (evaluación).

        Returns:
            Índice de la acción seleccionada.
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(action_space_size)}

        # Exploración
        if explore and random.random() < self.epsilon:
            return random.randint(0, action_space_size - 1)

        # Explotación
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    # ------------------------------------------------------------------
    # Actualización de la tabla Q (Bellman)
    # ------------------------------------------------------------------
    def update_q_table(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        action_space_size: int = None,
    ) -> None:
        """
        Actualiza Q(s,a) usando la regla de Q-Learning tabular.
        """
        if state not in self.q_table:
            size = action_space_size if action_space_size else 5
            self.q_table[state] = {a: 0.0 for a in range(size)}
        if next_state not in self.q_table:
            size = action_space_size if action_space_size else 5
            self.q_table[next_state] = {a: 0.0 for a in range(size)}

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            max_next_q = max(self.q_table[next_state].values())
            target = reward + self.gamma * max_next_q

        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    # Decaimiento de epsilon
    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """Reduce epsilon después de cada episodio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)