"""
Script principal de la tesis: orquesta las tres fases del sistema híbrido.

Fase A: PSO optimiza el FuzzyController.
Fase B: Q-Learning entrena en QuadrotorEnv.
Fase C: Sistema híbrido RL + Difuso.
"""

import os
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.rl.q_learning import QLearningAgent
from src.environments.quadrotor_env import QuadrotorEnv
from src.fuzzy.fuzzy_controller import FuzzyController
from src.bio.pso_optimizer import FuzzyPSOOptimizer


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("SISTEMA HÍBRIDO DE CONTROL ADAPTATIVO (Tesis Doctoral)")
    print("=" * 60)

    # ==================================================================
    # FASE A: PSO
    # ==================================================================
    print("\n[FASE A] Optimización del FuzzyController con PSO...")
    pso_cfg = cfg["pso"]

    optimizer = FuzzyPSOOptimizer(
        num_particles=pso_cfg["num_particles"],
        max_iter=pso_cfg["max_iter"],
        lb_error=pso_cfg["lb_error"],
        ub_error=pso_cfg["ub_error"],
        lb_delta=pso_cfg["lb_delta"],
        ub_delta=pso_cfg["ub_delta"],
        seed=pso_cfg["seed"],
    )

    best_params = optimizer.optimize()
    print(f"  Mejor MSE: {optimizer.best_mse:.6f}")
    print(f"  Parámetros: {np.round(best_params, 4)}")

    fuzzy_params_file = cfg["paths"]["fuzzy_params_file"]
    with open(fuzzy_params_file, "wb") as f:
        pickle.dump(best_params, f)
    print(f"  Guardado en: {fuzzy_params_file}")

    # ==================================================================
    # FASE B: Q-Learning
    # ==================================================================
    print("\n[FASE B] Entrenamiento del agente Q-Learning...")
    rl_cfg = cfg["rl"]
    env_cfg = cfg["environment"]

    env = QuadrotorEnv(
        max_steps=env_cfg["max_steps"],
        target_altitude=env_cfg["target_altitude"],
        tilt_limit=env_cfg["tilt_limit"],
        dt=env_cfg["dt"],
    )

    agent = QLearningAgent(
        alpha=rl_cfg["alpha"],
        gamma=rl_cfg["gamma"],
        epsilon=rl_cfg["epsilon"],
        epsilon_decay=rl_cfg["epsilon_decay"],
        epsilon_min=rl_cfg["epsilon_min"],
        seed=rl_cfg.get("seed", None),
    )

    action_space_size = env.action_space.n
    episodes = rl_cfg["episodes"]
    episode_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.choose_action(state, action_space_size)
            next_state, reward, done, truncated, info = env.step(action)
            agent.update_q_table(
                state, action, reward, next_state, done, action_space_size
            )
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            print(
                f"  Ep {ep+1}/{episodes} | R: {total_reward:.2f} | "
                f"eps: {agent.epsilon:.4f}"
            )

    window = 20
    moving_avg = np.convolve(
        episode_rewards, np.ones(window) / window, mode="valid"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Recompensa por episodio")
    plt.plot(
        range(window - 1, len(episode_rewards)),
        moving_avg,
        color="red",
        linewidth=2,
        label=f"Media móvil (ventana={window})",
    )
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa acumulada")
    plt.title("Curva de aprendizaje - Q-Learning tabular")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg["paths"]["rl_curve_file"], dpi=150)
    plt.close()
    print(f"  Curva guardada en: {cfg['paths']['rl_curve_file']}")

    # ==================================================================
    # FASE C: Sistema híbrido
    # ==================================================================
    print("\n[FASE C] Evaluación del sistema híbrido (RL + Difuso)...")

    fuzzy_controller = FuzzyController()
    with open(fuzzy_params_file, "rb") as f:
        loaded_params = pickle.load(f)
    fuzzy_controller.set_membership_params(loaded_params)

    error_threshold = 2.0

    # --- Solo RL ---
    env_rl = QuadrotorEnv(
        max_steps=env_cfg["max_steps"],
        target_altitude=env_cfg["target_altitude"],
        tilt_limit=env_cfg["tilt_limit"],
        dt=env_cfg["dt"],
    )
    state, _ = env_rl.reset()
    altitudes_rl = [env_rl.altitude]
    done = False
    truncated = False

    while not (done or truncated):
        action = agent.choose_action(state, action_space_size, explore=False)
        state, _, done, truncated, info = env_rl.step(action)
        altitudes_rl.append(info["altitude"])

    # --- Híbrido ---
    env_hybrid = QuadrotorEnv(
        max_steps=env_cfg["max_steps"],
        target_altitude=env_cfg["target_altitude"],
        tilt_limit=env_cfg["tilt_limit"],
        dt=env_cfg["dt"],
    )
    state, _ = env_hybrid.reset()
    altitudes_hybrid = [env_hybrid.altitude]
    prev_error = env_cfg["target_altitude"] - env_hybrid.altitude
    done = False
    truncated = False

    while not (done or truncated):
        error = env_cfg["target_altitude"] - env_hybrid.altitude
        delta_error = error - prev_error
        prev_error = error

        if abs(error) > error_threshold:
            u_fuzzy = fuzzy_controller.compute(error, delta_error)
            action = int(np.clip(round((u_fuzzy + 2.0) / 4.0 * 4.0), 0, 4))
        else:
            action = agent.choose_action(state, action_space_size, explore=False)

        state, _, done, truncated, info = env_hybrid.step(action)
        altitudes_hybrid.append(info["altitude"])

    # --- Gráfica comparativa ---
    plt.figure(figsize=(12, 6))
    plt.plot(altitudes_rl, label="Solo RL (Q-Learning)", linewidth=2)
    plt.plot(altitudes_hybrid, label="Híbrido (RL + Difuso PSO)", linewidth=2)
    plt.axhline(
        y=env_cfg["target_altitude"],
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Objetivo ({env_cfg['target_altitude']} m)",
    )
    plt.xlabel("Paso de simulación")
    plt.ylabel("Altitud (m)")
    plt.title("Comparación: Control solo RL vs. Control Híbrido")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg["paths"]["hybrid_comparison_file"], dpi=150)
    plt.close()
    print(f"  Gráfica guardada en: {cfg['paths']['hybrid_comparison_file']}")

    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()