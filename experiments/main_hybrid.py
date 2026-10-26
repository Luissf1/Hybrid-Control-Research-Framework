"""
Main Orchestrator — Full Hybrid Pipeline
=========================================

Executes the three phases of the doctoral thesis pipeline:

    Phase A: PSO optimizes the FuzzyController membership centers.
    Phase B: Tabular Q-Learning trains on the QuadrotorEnv.
    Phase C: Hybrid control — RL drives normally, fuzzy controller takes
             over whenever |error| > threshold.

Generated artifacts (inside results/):
    - best_fuzzy_params.pkl
    - rl_learning_curve.png
    - hybrid_comparison.png
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


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    # ==================================================================
    # 0. Load configuration and prepare output directory
    # ==================================================================
    cfg = load_config()
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)

    print("=" * 60)
    print("HYBRID ADAPTIVE CONTROL SYSTEM (Doctoral Thesis)")
    print("=" * 60)

    # ==================================================================
    # PHASE A — PSO optimizes the fuzzy controller
    # ==================================================================
    print("\n[PHASE A] Optimizing FuzzyController with PSO...")
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
    print(f"  Best MSE    : {optimizer.best_mse:.6f}")
    print(f"  Parameters  : {np.round(best_params, 4)}")

    # Persist the optimized parameters for later reuse
    fuzzy_params_file = cfg["paths"]["fuzzy_params_file"]
    with open(fuzzy_params_file, "wb") as f:
        pickle.dump(best_params, f)
    print(f"  Saved to    : {fuzzy_params_file}")

    # ==================================================================
    # PHASE B — Tabular Q-Learning training
    # ==================================================================
    print("\n[PHASE B] Training tabular Q-Learning agent...")
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
        seed=rl_cfg.get("seed"),
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
                f"  Episode {ep + 1}/{episodes} | "
                f"reward: {total_reward:.2f} | "
                f"epsilon: {agent.epsilon:.4f}"
            )

    # Plot learning curve with a moving average
    window = 20
    moving_avg = np.convolve(
        episode_rewards, np.ones(window) / window, mode="valid"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Episode reward")
    plt.plot(
        range(window - 1, len(episode_rewards)),
        moving_avg,
        color="red",
        linewidth=2,
        label=f"Moving average (window={window})",
    )
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("Learning curve — Tabular Q-Learning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg["paths"]["rl_curve_file"], dpi=150)
    plt.close()
    print(f"  Curve saved to: {cfg['paths']['rl_curve_file']}")

    # ==================================================================
    # PHASE C — Hybrid evaluation (RL + PSO-tuned fuzzy controller)
    # ==================================================================
    print("\n[PHASE C] Evaluating hybrid control (RL + fuzzy)...")

    # Load PSO-tuned fuzzy controller
    fuzzy_controller = FuzzyController()
    with open(fuzzy_params_file, "rb") as f:
        loaded_params = pickle.load(f)
    fuzzy_controller.set_membership_params(loaded_params)

    # Error threshold above which the fuzzy controller takes over
    error_threshold = 2.0

    # --- Baseline: RL only ---
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

    # --- Hybrid: RL with fuzzy safety override ---
    env_hybrid = QuadrotorEnv(
        max_steps=env_cfg["max_steps"],
        target_altitude=env_cfg["target_altitude"],
        tilt_limit=env_cfg["tilt_limit"],
        dt=env_cfg["dt"],
    )
    state, _ = env_hybrid.reset()
    altitudes_hybrid = [env_hybrid.altitude]
    previous_error = env_cfg["target_altitude"] - env_hybrid.altitude
    done = False
    truncated = False

    while not (done or truncated):
        error = env_cfg["target_altitude"] - env_hybrid.altitude
        delta_error = error - previous_error
        previous_error = error

        if abs(error) > error_threshold:
            # Fuzzy controller stabilizes; map its [-2, 2] output to [0, 4]
            u_fuzzy = fuzzy_controller.compute(error, delta_error)
            action = int(np.clip(round((u_fuzzy + 2.0) / 4.0 * 4.0), 0, 4))
        else:
            # RL agent governs close to the reference
            action = agent.choose_action(state, action_space_size, explore=False)

        state, _, done, truncated, info = env_hybrid.step(action)
        altitudes_hybrid.append(info["altitude"])

    # --- Comparative plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(altitudes_rl, label="RL only (Q-Learning)", linewidth=2)
    plt.plot(altitudes_hybrid, label="Hybrid (RL + PSO-tuned fuzzy)", linewidth=2)
    plt.axhline(
        y=env_cfg["target_altitude"],
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Set-point ({env_cfg['target_altitude']} m)",
    )
    plt.xlabel("Simulation step")
    plt.ylabel("Altitude (m)")
    plt.title("Comparison: RL-only vs. Hybrid control")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg["paths"]["hybrid_comparison_file"], dpi=150)
    plt.close()
    print(f"  Comparison saved to: {cfg['paths']['hybrid_comparison_file']}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()