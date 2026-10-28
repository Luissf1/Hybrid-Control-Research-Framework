"""
Ablation Study — Quantifying the Contribution of Each Component
================================================================

Three ablations of the full hybrid pipeline:

    A1. No PSO       : fuzzy controller uses DEFAULT membership params.
    A2. No fuzzy     : Phase C ignores the fuzzy override (RL-only).
    A3. Threshold sweep : error_threshold ∈ {1.0, 2.0, 3.0, 5.0}.

For each ablation, we evaluate Phase C over `n_episodes` runs and
report the Mean Squared Error (MSE) of the altitude tracking. The
MSE is the standard metric for tracking performance.

Outputs:
    - results/ablation_summary.csv
    - results/ablation_comparison.png
"""

import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.rl.q_learning import QLearningAgent
from src.environments.quadrotor_env import QuadrotorEnv
from src.fuzzy.fuzzy_controller import FuzzyController


# ----------------------------------------------------------------------
# Helper: run one Phase C episode and return the tracking MSE
# ----------------------------------------------------------------------
def run_phase_c(agent, fuzzy_controller, env_cfg, error_threshold, action_space_size):
    """
    Run one Phase C episode and return the MSE of altitude tracking.

    If error_threshold is None, the fuzzy override is disabled
    (RL-only baseline). If fuzzy_controller is None, the same.
    """
    env = QuadrotorEnv(
        max_steps=env_cfg["max_steps"],
        target_altitude=env_cfg["target_altitude"],
        tilt_limit=env_cfg["tilt_limit"],
        dt=env_cfg["dt"],
    )
    state, _ = env.reset()
    altitudes = [env.altitude]
    previous_error = env_cfg["target_altitude"] - env.altitude
    done, truncated = False, False

    while not (done or truncated):
        error = env_cfg["target_altitude"] - env.altitude
        delta_error = error - previous_error
        previous_error = error

        use_fuzzy = (
            fuzzy_controller is not None
            and error_threshold is not None
            and abs(error) > error_threshold
        )

        if use_fuzzy:
            u_fuzzy = fuzzy_controller.compute(error, delta_error)
            target_accel = u_fuzzy / 2.0
            accel_table = [1.0, 0.5, 0.0, -0.5, -1.0]
            action = int(np.argmin([abs(target_accel - a) for a in accel_table]))
        else:
            action = agent.choose_action(state, action_space_size, explore=False)

        state, _, done, truncated, info = env.step(action)
        altitudes.append(info["altitude"])

    altitudes = np.array(altitudes)
    target = np.full_like(altitudes, env_cfg["target_altitude"])
    mse = float(np.mean((altitudes - target) ** 2))
    return mse, altitudes


# ----------------------------------------------------------------------
# Helper: train a fresh agent (same seed as main_hybrid.py)
# ----------------------------------------------------------------------
def train_agent(cfg):
    env = QuadrotorEnv(
        max_steps=cfg["environment"]["max_steps"],
        target_altitude=cfg["environment"]["target_altitude"],
        tilt_limit=cfg["environment"]["tilt_limit"],
        dt=cfg["environment"]["dt"],
    )
    agent = QLearningAgent(
        alpha=cfg["rl"]["alpha"],
        gamma=cfg["rl"]["gamma"],
        epsilon=cfg["rl"]["epsilon"],
        epsilon_decay=cfg["rl"]["epsilon_decay"],
        epsilon_min=cfg["rl"]["epsilon_min"],
        seed=cfg["rl"].get("seed"),
    )
    action_space_size = env.action_space.n
    for _ in range(cfg["rl"]["episodes"]):
        state, _ = env.reset()
        done, truncated = False, False
        while not (done or truncated):
            action = agent.choose_action(state, action_space_size)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done, action_space_size)
            state = next_state
        agent.decay_epsilon()
    return agent, action_space_size


# ----------------------------------------------------------------------
# Main ablation driver
# ----------------------------------------------------------------------
def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["environment"]
    print("[ABLATIONS] Training reference agent...")
    agent, action_space_size = train_agent(cfg)

    # --- Fuzzy controller variants ---
    fuzzy_pso = FuzzyController()
    with open(cfg["paths"]["fuzzy_params_file"], "rb") as f:
        best_params = pickle.load(f)
    fuzzy_pso.set_membership_params(best_params)

    fuzzy_default = FuzzyController()   # default memberships, no PSO

    # --- Ablations ---
    # Each entry: (label, fuzzy_controller, error_threshold)
    ablations = [
        ("Full hybrid (PSO fuzzy, thr=2.0)", fuzzy_pso, 2.0),
        ("A1: No PSO (default fuzzy, thr=2.0)", fuzzy_default, 2.0),
        ("A2: No fuzzy (RL only)", None, None),
        ("A3a: thr=1.0", fuzzy_pso, 1.0),
        ("A3b: thr=3.0", fuzzy_pso, 3.0),
        ("A3c: thr=5.0", fuzzy_pso, 5.0),
    ]

    results = []
    curves = {}
    for label, fc, thr in ablations:
        mse, alt = run_phase_c(agent, fc, env_cfg, thr, action_space_size)
        results.append({"ablation": label, "mse": mse, "final_alt": alt[-1]})
        curves[label] = alt
        print(f"  {label:<40s} MSE = {mse:.4f}  final_alt = {alt[-1]:.3f} m")

    # Save summary CSV (great for a thesis appendix)
    df = pd.DataFrame(results)
    df.to_csv("results/ablation_summary.csv", index=False)
    print("\nSaved results/ablation_summary.csv")

    # Comparison plot: altitude curves
    plt.figure(figsize=(12, 6))
    for label, alt in curves.items():
        plt.plot(alt, label=label, linewidth=1.8)
    plt.axhline(
        y=env_cfg["target_altitude"],
        color="black", linestyle="--", linewidth=1,
        label=f"Set-point ({env_cfg['target_altitude']} m)",
    )
    plt.xlabel("Simulation step")
    plt.ylabel("Altitude (m)")
    plt.title("Ablation study: altitude tracking under different controller configurations")
    plt.legend(fontsize=8, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/ablation_comparison.png", dpi=150)
    plt.close()
    print("Saved results/ablation_comparison.png")

    # Bar chart of MSEs (compact table for the thesis)
    plt.figure(figsize=(10, 5))
    plt.barh(df["ablation"], df["mse"], color="steelblue")
    plt.xlabel("Tracking MSE (lower is better)")
    plt.title("Ablation study: tracking MSE by configuration")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/ablation_mse.png", dpi=150)
    plt.close()
    print("Saved results/ablation_mse.png")


if __name__ == "__main__":
    main()